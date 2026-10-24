"""Train the transformer controller: behaviour cloning + DAgger."""
import json, math, time, os
import numpy as np
import torch
import torch.nn as nn

import plant as pl
import controllers as C
import dataset as D
from dataset import H, FEAT, DT, features_row, random_x0, windows
from model import Policy
from plant import P, wrap

torch.manual_seed(0)
np.random.seed(0)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------- policy as a controller
class PolicyCtl:
    """Wraps the torch policy so it can drive the numpy plant like any other controller.

    IMPORTANT: the running window must follow the dataset convention exactly --
    feature row k is (state_k, action_{k-1}), so the last token of the window
    always carries the *current* state and the previous action.
    """
    def __init__(self, model, device="cpu"):
        self.m = model.eval()
        self.dev = device
        self.buf = None
        self.u_prev = 0.0
        self.name = "Transformer"

    def reset(self):
        self.buf = None
        self.u_prev = 0.0

    def __call__(self, x, t=0.0):
        f = features_row(x, self.u_prev)
        if self.buf is None:
            self.buf = np.tile(f, (H, 1)).astype(np.float32)
        else:
            self.buf = np.roll(self.buf, -1, axis=0)
            self.buf[-1] = f
        with torch.no_grad():
            u = float(self.m(torch.from_numpy(self.buf[None]).to(self.dev))[0])
        self.u_prev = u
        return u


def rollout_policy(ctl, x0, T=12.0, noise=0.6, rng=None):
    """Closed-loop rollout of the policy; returns states, applied forces (for DAgger)."""
    n = int(T / DT)
    ctl.reset()
    x = np.array(x0, dtype=float)
    X = np.zeros((n + 1, 4))
    U = np.zeros(n + 1)
    X[0] = x
    for k in range(n):
        u = ctl(x, k * DT)
        a = float(np.clip(u + rng.normal(0, noise), -P["u_max"], P["u_max"]))
        U[k] = a
        x = pl.rk4(x, a, DT)
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]
            x[2] = 0.0
        X[k + 1] = x
    U[n] = U[n - 1]
    return X, U


def relabel(X, expert):
    """Run the expert over the visited state sequence to get its action at each state."""
    expert.reset()
    return np.array([expert(X[k], k * DT) for k in range(len(X))])


# -------------------------------------------------------------- closed-loop eval
def rollout_any(ctl, x0, T=12.0, dist=None):
    """Rollout that works for both classical controllers and PolicyCtl."""
    n = int(T / DT)
    ctl.reset()
    x = np.array(x0, dtype=float)
    X = np.zeros((n + 1, 4)); U = np.zeros(n + 1); X[0] = x
    for k in range(n):
        u = ctl(x, k * DT)
        a = float(np.clip(u + (0.0 if dist is None else dist(k * DT)), -P["u_max"], P["u_max"]))
        U[k] = u
        x = pl.rk4(x, a, DT)
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]; x[2] = 0.0
        X[k + 1] = x
    U[n] = U[n - 1]
    return X, U


def evaluate(ctl, n_ep=40, T=12.0, seed=7, disturb=False):
    rng = np.random.default_rng(seed)
    res = []
    for i in range(n_ep):
        x0 = random_x0(rng, "hard")
        d = None
        if disturb:
            mag = rng.uniform(2.0, 4.5)
            d = (lambda t, m=mag: m if 4.0 <= t < 8.0 else 0.0)
        X, U = rollout_any(ctl, x0, T=T, dist=d)
        th = np.abs(wrap(X[:, 1]))
        late = slice(int(0.75 * T / DT), None)
        ok = bool(th[late].max() < 0.15 and np.abs(X[late, 0]).max() < 1.5) if len(th[late]) else False
        up = np.where(th < 0.1)[0]
        res.append(dict(ok=ok, t_up=float(up[0] * DT) if len(up) else 99.0,
                        th_rms=float(np.sqrt(np.mean(th[late] ** 2))),
                        u_rms=float(np.sqrt(np.mean(U[late] ** 2))),
                        p_max=float(np.abs(X[:, 0]).max()),
                        rail_hit=bool(np.abs(X[:, 0]).max() > P["x_max"] - 1e-6)))
    agg = dict(success=float(np.mean([r["ok"] for r in res])),
               t_up=float(np.mean([r["t_up"] for r in res if r["t_up"] < 90] or [99])),
               th_rms=float(np.mean([r["th_rms"] for r in res])),
               u_rms=float(np.mean([r["u_rms"] for r in res])),
               p_max=float(np.mean([r["p_max"] for r in res])),
               rail_rate=float(np.mean([r["rail_hit"] for r in res])))
    return agg, res


# --------------------------------------------------------------------- training
def train(model, Xtr, Ytr, Xva, Yva, epochs, bs=512, lr=2e-3, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * math.ceil(len(Xtr) / bs), pct_start=0.25)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    Xva_t = torch.from_numpy(Xva); Yva_t = torch.from_numpy(Yva)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            
            # targets are stored normalised by u_max
            loss = lossf(pred / P["u_max"], Ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_t) / P["u_max"], Yva_t).item()
        hist.append(dict(epoch=ep + 1, train=float(tot / len(perm)), val=float(vl)))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  {tag} epoch {ep+1:3d}/{epochs}  train={tot/len(perm):.5f}  val={vl:.5f}")
    return hist


def main():
    t0 = time.time()
    print("[1] generating expert data ...")
    X, Y = D.gen(n_ep=300, T=8.0, noise=1.5, seed=1)
    print("    windows:", X.shape, " t=%.1fs" % (time.time() - t0))
    n = len(X); idx = np.random.permutation(n)
    nva = max(2000, int(0.08 * n))
    va, tr = idx[:nva], idx[nva:]
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    print("    train %d / val %d" % (len(Xtr), len(Xva)))

    model = Policy()
    print("    params:", sum(p.numel() for p in model.parameters()))
    hist = train(model, Xtr, Ytr, Xva, Yva, epochs=24, tag="BC")

    pctl = PolicyCtl(model)
    agg, _ = evaluate(pctl, n_ep=30)
    print("    BC closed-loop:", {k: round(v, 3) for k, v in agg.items()})

    expert = D.build_expert()
    dagger_stats = []
    for rnd in range(2):
        print(f"[2.{rnd+1}] DAgger round {rnd+1}")
        rng = np.random.default_rng(100 + rnd)
        Xs, Ys = [], []
        for i in range(140):
            x0 = random_x0(rng, "hard" if rng.random() < 0.8 else "near")
            Xr, Ur = rollout_policy(pctl, x0, T=8.0, noise=1.2, rng=rng)
            Ue = relabel(Xr, expert)
            F = np.stack([features_row(Xr[k], Ur[k - 1] if k > 0 else 0.0) for k in range(len(Xr))])
            Xw, Yw = windows(F, Ue)
            Xs.append(Xw); Ys.append(Yw)
        Xa, Ya = np.concatenate(Xs), np.concatenate(Ys)
        print("    aggregated new windows:", Xa.shape)
        Xtr = np.concatenate([Xtr, Xa]); Ytr = np.concatenate([Ytr, Ya])
        # keep a bounded replay buffer
        if len(Xtr) > 260000:
            keep = np.random.choice(len(Xtr), 260000, replace=False)
            Xtr, Ytr = Xtr[keep], Ytr[keep]
        h = train(model, Xtr, Ytr, Xva, Yva, epochs=10, tag=f"DAgger{rnd+1}")
        hist += h
        agg, _ = evaluate(pctl, n_ep=30)
        print("    closed-loop:", {k: round(v, 3) for k, v in agg.items()})
        dagger_stats.append(agg)

    print("[3] final evaluation")
    fin, _ = evaluate(pctl, n_ep=60)
    print("    nominal      :", {k: round(v, 3) for k, v in fin.items()})
    fin_d, _ = evaluate(pctl, n_ep=60, disturb=True)
    print("    with 2-4.5N disturbance:", {k: round(v, 3) for k, v in fin_d.items()})

    torch.save(model.state_dict(), os.path.join(OUT, "policy.pt"))
    json.dump(dict(history=hist, final=fin, final_disturbed=fin_d, dagger=dagger_stats),
              open(os.path.join(OUT, "train_log.json"), "w"), indent=1)
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

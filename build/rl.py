"""PPO fine-tuning of the imitation-learned transformer policy.

Question: is a behaviour-cloned transformer doomed to be "as good as the expert"?
This script keeps the network and its BC/DAgger initialisation, and only changes
the objective: maximise closed-loop return instead of matching expert actions.

Reward (per 20 ms step):
    r = cos(theta)                       upright is worth 1, hanging is worth -1
        - 0.15 * p^2                     keep the cart near the middle
        - 0.50 * (u/12)^2                penalise effort (and saturation)
        + 0.50 * 1{|theta| < 0.1}        bonus for actually holding the top
        - 4.0  * 1{rail hit}             hard penalty (once per episode)
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn

import plant as pl
import controllers as C
from dataset import H, FEAT, DT, features_row, random_x0
from model import Policy
from plant import P, wrap

UMAX = P["u_max"]
OUT = os.path.join(os.path.dirname(__file__), "out")
torch.set_num_threads(4)


# ------------------------------------------------------------------ batched plant
def deriv_batch(x, u):
    th = x[:, 1]; v = x[:, 2]; w = x[:, 3]
    s, c = np.sin(th), np.cos(th)
    den = P["m_c"] + P["m_p"] * s * s
    pdd = (u - P["b_c"] * v + P["m_p"] * P["l"] * s * w * w
           + P["b_p"] * c * w - P["m_p"] * P["g"] * s * c) / den
    wdd = (P["g"] * s - c * pdd - P["b_p"] * w / (P["m_p"] * P["l"])) / P["l"]
    return np.stack([v, w, pdd, wdd], axis=1)


def rk4_batch(x, u, sub=2):
    h = DT / sub
    for _ in range(sub):
        k1 = deriv_batch(x, u)
        k2 = deriv_batch(x + 0.5 * h * k1, u)
        k3 = deriv_batch(x + 0.5 * h * k2, u)
        k4 = deriv_batch(x + h * k3, u)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    hit = np.abs(x[:, 0]) > P["x_max"]
    x[hit, 0] = np.sign(x[hit, 0]) * P["x_max"]
    x[hit, 2] = 0.0
    return x, hit


def feat_batch(x, u_prev):
    return np.stack([
        np.sin(x[:, 1]), np.cos(x[:, 1]),
        x[:, 3] / 4.0, x[:, 2] / 4.0,
        x[:, 0] / P["x_max"], u_prev / UMAX,
    ], axis=1).astype(np.float32)


def init_states(n, rng):
    """Same distribution as the evaluation protocol (uniform angle over the circle)."""
    return np.stack([
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(-np.pi, np.pi, n),
        rng.uniform(-0.5, 0.5, n),
        rng.uniform(-1.0, 1.0, n),
    ], axis=1)


# ------------------------------------------------------------------ policy + PPO
class GaussPolicy(nn.Module):
    """BC transformer as the mean of a Gaussian in force space."""
    def __init__(self, base, log_std=-0.6):
        super().__init__()
        self.base = base
        self.log_std = nn.Parameter(torch.tensor(float(log_std)))

    def forward(self, obs):
        return self.base(obs)                       # mean force, bounded by +-12


def evaluate(policy, n_ep=40, T=12.0, seed=11, disturb=None):
    rng = np.random.default_rng(seed)
    X0 = init_states(n_ep, rng)
    x = X0.copy()
    buf = None
    u_prev = np.zeros(n_ep)
    N = int(T / DT)
    th_late = np.zeros(n_ep); u_all = np.zeros(n_ep); p_late = np.zeros(n_ep)
    t_up = np.full(n_ep, np.inf); rail = np.zeros(n_ep, bool)
    for k in range(N):
        f = feat_batch(x, u_prev)
        buf = np.tile(f[:, None, :], (1, H, 1)) if buf is None else \
            np.concatenate([buf[:, 1:, :], f[:, None, :]], axis=1)
        with torch.no_grad():
            u = policy(torch.from_numpy(buf)).numpy()
        u = np.clip(u, -UMAX, UMAX)
        u_prev = u
        if disturb is None:
            d = 0.0
        elif callable(disturb):
            d = disturb(k * DT)
        else:
            d = float(disturb) if 4.0 <= k * DT < 8.0 else 0.0
        x, hit = rk4_batch(x, np.clip(u + d, -UMAX, UMAX))
        rail |= hit
        th = np.abs(wrap(x[:, 1]))
        u_all += u * u
        upd = (t_up == np.inf) & (th < 0.1)
        t_up[upd] = (k + 1) * DT
        if k >= 0.75 * N:
            th_late += th ** 2
            p_late = np.maximum(p_late, np.abs(x[:, 0]))
    n_late = N - int(0.75 * N)
    th_rms = np.sqrt(th_late / n_late)
    ok = (th_rms < 0.15) & (p_late < 1.5)
    t_up = np.minimum(t_up, T)                    # never-reached counts as the horizon
    return dict(success=float(ok.mean()), t_up=float(np.mean(t_up)),
                th_rms=float(np.mean(th_rms)), u_rms=float(np.mean(np.sqrt(u_all / N))),
                p_max=float(np.mean(p_late)), rail=float(rail.mean()))


def main(iterations=150, n_env=48, T=8.0, lr=1.5e-4, seed=0, epochs=2,
         dr=0.0, init="policy.pt", tag="policy_rl"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    base = Policy()
    sd = torch.load(os.path.join(OUT, init))
    sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")} if any(
        k.startswith("base.") for k in sd) else sd
    base.load_state_dict(sd)
    pol = GaussPolicy(base)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    N = int(T / DT)

    # reference: what the imitation policy does on the same eval protocol
    ref = evaluate(pol)
    print("BC init        :", {k: round(v, 4) for k, v in ref.items()}, flush=True)

    best = dict(score=-1e9, state=None, it=0, m=ref)
    hist = []
    t0 = time.time()
    for it in range(1, iterations + 1):
        x = init_states(n_env, rng)
        if dr > 0:                                     # domain randomisation
            dm = np.where(rng.random(n_env) < dr, rng.uniform(1.5, 5.0, n_env), 0.0)
            dt0 = rng.uniform(0.0, T - 2.0, n_env)
            ddur = rng.uniform(1.0, 3.0, n_env)
        else:
            dm = np.zeros(n_env); dt0 = np.zeros(n_env); ddur = np.zeros(n_env)
        buf = None
        u_prev = np.zeros(n_env)
        obs_s = np.zeros((N, n_env, H, FEAT), np.float32)
        act_s = np.zeros((N, n_env), np.float32)
        logp_s = np.zeros((N, n_env), np.float32)
        rew_s = np.zeros((N, n_env), np.float32)
        val_s = np.zeros((N, n_env), np.float32)
        done_s = np.zeros((N, n_env), np.float32)
        std = pol.log_std.exp().item()

        for k in range(N):
            f = feat_batch(x, u_prev)
            buf = np.tile(f[:, None, :], (1, H, 1)) if buf is None else \
                np.concatenate([buf[:, 1:, :], f[:, None, :]], axis=1)
            obs_s[k] = buf
            with torch.no_grad():
                mean = pol(torch.from_numpy(buf)).numpy()
            a = mean + std * rng.standard_normal(n_env).astype(np.float32)
            a_c = np.clip(a, -UMAX, UMAX).astype(np.float32)
            logp = (-0.5 * ((a - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi))
            u_prev = a_c
            dist = np.where((k * DT >= dt0) & (k * DT < dt0 + ddur), dm, 0.0)
            x, hit = rk4_batch(x, np.clip(a_c + dist, -UMAX, UMAX))
            th = wrap(x[:, 1])
            r = (np.cos(th) - 0.15 * x[:, 0] ** 2 - 0.5 * (a_c / UMAX) ** 2
                 + 0.5 * (np.abs(th) < 0.1) - 4.0 * hit)
            act_s[k] = a_c; logp_s[k] = logp; rew_s[k] = r; done_s[k] = 0.0

        # rewards-to-go (no bootstrapping needed: fixed horizon)
        ret = np.zeros_like(rew_s); G = np.zeros(n_env)
        for k in reversed(range(N)):
            G = rew_s[k] + 0.995 * G
            ret[k] = G
        adv = ret - ret.mean()
        adv = adv / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(obs_s.reshape(N * n_env, H, FEAT))
        act_t = torch.from_numpy(act_s.reshape(-1))
        oldlogp_t = torch.from_numpy(logp_s.reshape(-1))
        adv_t = torch.from_numpy(adv.reshape(-1).astype(np.float32))
        ret_t = torch.from_numpy(ret.reshape(-1).astype(np.float32))

        idx = np.arange(N * n_env)
        for ep in range(epochs):
            rng.shuffle(idx)
            for i in range(0, len(idx), 1024):
                b = torch.from_numpy(idx[i:i + 1024])
                mean = pol(obs_t[b])
                std_ = pol.log_std.exp()
                logp = (-0.5 * ((act_t[b] - mean) / std_) ** 2 - pol.log_std
                        - 0.5 * math.log(2 * math.pi))
                ratio = torch.exp(logp - oldlogp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b]
                loss = -torch.min(s1, s2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()

        if it % 8 == 0 or it == 1:
            m = evaluate(pol, n_ep=24, T=12.0)
            md = evaluate(pol, n_ep=24, T=12.0, seed=13, disturb=4.0) if dr > 0 else None
            score = 100 * m["success"] - m["t_up"] - 10 * m["rail"]
            if md is not None:
                score += 60 * md["success"]
            m = dict(m); m["dist"] = md
            hist.append(dict(it=it, eval=m, std=float(pol.log_std.exp().item()),
                             ret=float(ret.mean())))
            dd = "" if md is None else f" | +4N succ={md['success']:.2f} th={md['th_rms']:.3f}"
            print(dd, flush=True)
            print(f"it {it:3d}  R={ret.mean():8.2f}  std={pol.log_std.exp():.2f}  "
                  f"succ={m['success']:.2f} t_up={m['t_up']:5.2f} th={m['th_rms']:.4f} "
                  f"u={m['u_rms']:5.2f} rail={m['rail']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            if score > best["score"]:
                best = dict(score=score, it=it, m=m,
                            state={k: v.clone() for k, v in pol.state_dict().items()})
    torch.save(best["state"], os.path.join(OUT, tag + ".pt"))
    json.dump(dict(history=hist, best_it=best["it"], best_eval=best["m"], bc_eval=ref),
              open(os.path.join(OUT, tag + "_log.json"), "w"), indent=1)
    print("best iteration", best["it"],
          {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best["m"].items()})
    print("total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()

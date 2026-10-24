"""Final comparison: expert (traditional nonlinear) vs BC vs RL vs RL+DR.

All four controllers are evaluated on the identical 60-initial-condition protocol that
is used everywhere else in the report. The two learned policies are exported as int8
blobs (what the browser actually runs), so the numbers here match what the page shows.
"""
import base64, json, os
import numpy as np
import torch

import controllers as C
import train as T
from dataset import H, FEAT, DT, random_x0
from model import Policy
from plant import P, wrap

OUT = os.path.join(os.path.dirname(__file__), "out")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def quantise(model):
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    blob = {}
    for k, v in sd.items():
        if k.endswith("weight") and v.ndim == 2:
            m = np.abs(v).max() or 1.0
            blob[k] = dict(dtype="i8", shape=list(v.shape), scale=float(m / 127),
                           data=b64(np.round(v / m * 127).astype(np.int8)))
        else:
            blob[k] = dict(dtype="f32", shape=list(v.shape), data=b64(v.astype(np.float32)))
    W = {}
    for k, e in blob.items():
        raw = np.frombuffer(base64.b64decode(e["data"]),
                            dtype=np.int8 if e["dtype"] == "i8" else np.float32)
        raw = raw.astype(np.float64).reshape(e["shape"])
        W[k] = raw * e["scale"] if e["dtype"] == "i8" else raw
    return blob, W


def load_quantised(path, strip="base."):
    sd = torch.load(path)
    if any(k.startswith(strip) for k in sd):
        sd = {k[len(strip):]: v for k, v in sd.items() if k.startswith(strip)}
    model = Policy(); model.load_state_dict(sd); model.eval()
    blob, W = quantise(model)
    m2 = Policy(); m2.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in W.items()})
    m2.eval()
    return m2, blob


def measure(ctl, ics, dist=None):
    rows = []
    for x0 in ics:
        d = (lambda t, m=dist: m if 4.0 <= t < 8.0 else 0.0) if dist else None
        X, U = T.rollout_any(ctl, x0, T=12.0, dist=d)
        th = np.abs(wrap(X[:, 1]))
        late = slice(int(9.0 / DT), None)
        up = np.where(th < 0.1)[0]
        rows.append(dict(
            ok=bool(th[late].max() < 0.15 and np.abs(X[late, 0]).max() < 1.5),
            t_up=float(up[0] * DT) if len(up) else 99.0,
            th=float(np.sqrt(np.mean(th[late] ** 2))),
            u=float(np.sqrt(np.mean(U ** 2))),
            tv=float(np.abs(np.diff(U)).sum()),
            du=float(np.abs(np.diff(U)).max()),
            pmax=float(np.abs(X[:, 0]).max()),
            rail=bool(np.abs(X[:, 0]).max() > P["x_max"] - 1e-9)))
    a = {k: float(np.mean([r[k] for r in rows])) for k in ("t_up", "th", "u", "tv", "du", "pmax")}
    a["success"] = float(np.mean([r["ok"] for r in rows]))
    a["rail"] = float(np.mean([r["rail"] for r in rows]))
    return a


def main():
    m_bc, blob_bc = load_quantised(os.path.join(OUT, "policy.pt"), strip="")
    m_rl, blob_rl = load_quantised(os.path.join(OUT, "policy_rl.pt"))
    m_dr, blob_dr = load_quantised(os.path.join(OUT, "policy_rl_dr.pt"))
    json.dump(dict(params=blob_bc), open(os.path.join(OUT, "policy_weights.json"), "w"))
    json.dump(dict(params=blob_rl), open(os.path.join(OUT, "policy_rl_weights.json"), "w"))
    json.dump(dict(params=blob_dr), open(os.path.join(OUT, "policy_dr_weights.json"), "w"))

    rng = np.random.default_rng(2025)
    ics = [random_x0(rng, "hard") for _ in range(60)]
    ctrls = {"expert": C.build("hybrid_lqr"), "bc": T.PolicyCtl(m_bc),
             "rl": T.PolicyCtl(m_rl), "rl_dr": T.PolicyCtl(m_dr)}
    out = {}
    for name, c in ctrls.items():
        a, b = measure(c, ics), measure(c, ics, dist=4.0)
        out[name] = dict(nominal=a, disturbed=b)
        print(f"{name:7s} succ={a['success']:.2f} t_up={a['t_up']:5.2f} th={a['th']:.4f} "
              f"u={a['u']:5.2f} du={a['du']:5.2f} pmax={a['pmax']:5.2f} rail={a['rail']:.2f} "
              f"|| +4N succ={b['success']:.2f} th={b['th']:.4f}")
    json.dump(out, open(os.path.join(OUT, "rl_compare.json"), "w"), indent=1)

    # training curves for the figure
    logs = {}
    for tag, fname in (("policy_rl", "rl_log.json"), ("policy_rl_dr", "policy_rl_dr_log.json")):
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            logs[tag] = json.load(open(p))
    json.dump(logs, open(os.path.join(OUT, "rl_logs.json"), "w"))

    # probes for the in-page JS parity check of the RL blob
    X, _ = __import__("dataset").gen(n_ep=4, T=6.0, noise=1.5, seed=77)
    idx = np.random.RandomState(5).choice(len(X), 5, replace=False)
    probes = torch.from_numpy(X[idx])
    with torch.no_grad():
        ob = m_bc(probes).numpy()
        od = m_dr(probes).numpy()
    json.dump(dict(inputs=[np.round(X[idx][i], 6).tolist() for i in range(5)],
                   expect_bc=[round(float(v), 6) for v in ob],
                   expect_dr=[round(float(v), 6) for v in od]),
              open(os.path.join(OUT, "probes2.json"), "w"))
    print("wrote int8 blobs + rl_compare.json")


if __name__ == "__main__":
    main()

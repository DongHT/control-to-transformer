"""Final comparison of all four controllers on identical random initial conditions."""
import json, os
import numpy as np
import torch

import plant as pl
import controllers as C
import dataset as D
from dataset import H, FEAT, DT, random_x0
from model import Policy
from plant import P, wrap
import train as T

OUT = os.path.join(os.path.dirname(__file__), "out")


def load_policy(quantised=True):
    model = Policy()
    model.load_state_dict(torch.load(os.path.join(OUT, "policy.pt")))
    model.eval()
    if not quantised:
        return T.PolicyCtl(model)
    import base64
    blob = json.load(open(os.path.join(OUT, "policy_weights.json")))["params"]
    W = {}
    for k, e in blob.items():
        raw = np.frombuffer(base64.b64decode(e["data"]),
                            dtype=np.int8 if e["dtype"] == "i8" else np.float32)
        raw = raw.astype(np.float64).reshape(e["shape"])
        W[k] = raw * e["scale"] if e["dtype"] == "i8" else raw
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in W.items()})
    model.eval()
    return T.PolicyCtl(model)


def run_set(ctl, ics, T_=12.0, disturb=None):
    out = []
    for x0 in ics:
        d = None
        if disturb:
            d = (lambda t, m=disturb: m if 4.0 <= t < 8.0 else 0.0)
        X, U = T.rollout_any(ctl, x0, T=T_, dist=d)
        th = np.abs(wrap(X[:, 1]))
        late = slice(int(0.75 * T_ / DT), None)
        up = np.where(th < 0.1)[0]
        out.append(dict(
            ok=bool(th[late].max() < 0.15 and np.abs(X[late, 0]).max() < 1.5),
            t_up=float(up[0] * DT) if len(up) else 99.0,
            th_rms=float(np.sqrt(np.mean(th[late] ** 2))),
            u_rms=float(np.sqrt(np.mean(U[late] ** 2))),
            p_max=float(np.abs(X[:, 0]).max()),
            th_end=float(th[-1]),
        ))
    n = len(out)
    return dict(success=float(np.mean([r["ok"] for r in out])),
                t_up=float(np.mean([r["t_up"] for r in out])),
                th_rms=float(np.mean([r["th_rms"] for r in out])),
                u_rms=float(np.mean([r["u_rms"] for r in out])),
                p_max=float(np.mean([r["p_max"] for r in out])),
                rail=float(np.mean([r["p_max"] > P["x_max"] - 1e-9 for r in out])))


def main():
    rng = np.random.default_rng(2025)
    ics = [random_x0(rng, "hard") for _ in range(60)]
    tf_q = load_policy(True)
    tf_f = load_policy(False)
    ctrls = {"LQR": C.build("lqr"),
             "Energy+LQR": C.build("hybrid_lqr"),
             "Energy+SMC": C.build("hybrid_smc"),
             "Transformer": tf_q}
    rows = {}
    for name, ctl in ctrls.items():
        a = run_set(ctl, ics)
        d = run_set(ctl, ics, disturb=4.0)
        rows[name] = dict(nominal=a, disturbed=d)
        print(f"{name:12s} nominal  succ={a['success']:.2f} t_up={a['t_up']:5.2f} "
              f"th_rms={a['th_rms']:.4f} u_rms={a['u_rms']:5.2f} p_max={a['p_max']:5.2f} rail={a['rail']:.2f}")
        print(f"{'':12s} dist4N   succ={d['success']:.2f} t_up={d['t_up']:5.2f} "
              f"th_rms={d['th_rms']:.4f} u_rms={d['u_rms']:5.2f} p_max={d['p_max']:5.2f} rail={d['rail']:.2f}")
    a = run_set(tf_f, ics)
    print(f"{'TF(float32)':12s} nominal  succ={a['success']:.2f} t_up={a['t_up']:5.2f} "
          f"th_rms={a['th_rms']:.4f} u_rms={a['u_rms']:5.2f} p_max={a['p_max']:5.2f}")
    rows["Transformer_f32_check"] = dict(nominal=a)
    json.dump(rows, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

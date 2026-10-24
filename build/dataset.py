"""Dataset generation for the transformer controller.

The *expert* is the traditional nonlinear controller (energy swing-up + LQR).
Data are (state-history -> expert force) pairs collected from closed-loop rollouts
with exploration noise, i.e. pure behaviour cloning from a nonlinear controller,
plus DAgger rounds using the transformer's own rollouts.
"""
import numpy as np
import plant as pl
import controllers as C
from plant import P, wrap

H = 24                     # context window (steps)  -> 24*0.02 = 0.48 s
FEAT = 6                   # features per step
DT = P["dt"]


def features_row(x, u_prev, p=P):
    pos, th, v, w = x
    return np.array([
        np.sin(th), np.cos(th),
        w / 4.0, v / 4.0, pos / p["x_max"], u_prev / p["u_max"],
    ], dtype=np.float32)


def random_x0(rng, mode="hard"):
    if mode == "hard":
        th0 = rng.uniform(-np.pi, np.pi)
        pos0 = rng.uniform(-1.0, 1.0)
        v0 = rng.uniform(-0.5, 0.5)
        w0 = rng.uniform(-1.0, 1.0)
    else:                                   # near-upright
        th0 = rng.uniform(-0.6, 0.6)
        pos0 = rng.uniform(-0.8, 0.8)
        v0 = 0.0
        w0 = rng.uniform(-0.6, 0.6)
    return np.array([pos0, th0, v0, w0])


def episode(ctrl, x0, T=8.0, noise=1.5, rng=None, dist=None):
    """Roll the expert in closed loop; return (states, features, expert_actions)."""
    n = int(T / DT)
    ctrl.reset()
    x = np.array(x0, dtype=float)
    X = np.zeros((n + 1, 4))
    Uex = np.zeros(n + 1)
    Uapp = np.zeros(n + 1)
    X[0] = x
    for k in range(n):
        u = ctrl(x, k * DT)
        Uex[k] = u
        a = u + (rng.normal(0, noise) if noise else 0.0)
        a = float(np.clip(a, -P["u_max"], P["u_max"]))
        Uapp[k] = a
        if dist is not None:
            a = float(np.clip(a + dist(k * DT), -P["u_max"], P["u_max"]))
        x = pl.rk4(x, a, DT)
        # rail limits: inelastic stop
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]
            x[2] = 0.0
        X[k + 1] = x
    Uex[n] = Uex[n - 1]
    Uapp[n] = Uapp[n - 1]
    F = np.stack([features_row(X[k], Uapp[k - 1]) for k in range(n + 1)])
    return X, F, Uex


def windows(F, Uex, h=H, stride=2):
    n = len(F) - h
    idx = np.arange(0, n, stride)
    Xw = np.stack([F[i:i + h] for i in idx])
    Yw = Uex[idx + h - 1] / P["u_max"]
    return Xw, Yw.astype(np.float32)


def build_expert():
    return C.build("hybrid_lqr")


def gen(n_ep=260, T=8.0, noise=1.5, seed=0, mode_mix=(0.75, 0.25), dist_ep=0.25):
    rng = np.random.default_rng(seed)
    expert = build_expert()
    Xs, Ys = [], []
    for i in range(n_ep):
        mode = "hard" if rng.random() < mode_mix[0] else "near"
        x0 = random_x0(rng, mode)
        d = None
        if rng.random() < dist_ep:            # disturb some episodes (matched force)
            mag = rng.uniform(2.0, 5.0)
            t0 = rng.uniform(1.0, 4.0)
            dur = rng.uniform(1.0, 3.0)
            d = (lambda t, m=mag, a=t0, b=dur: m if a <= t < a + b else 0.0)
        X, F, U = episode(expert, x0, T=T, noise=noise, rng=rng, dist=d)
        Xw, Yw = windows(F, U)
        Xs.append(Xw)
        Ys.append(Yw)
    return np.concatenate(Xs), np.concatenate(Ys)


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, Y = gen(n_ep=20)
    print("windows", X.shape, Y.shape, "in %.1fs" % (time.time() - t0))
    print("action stats: mean %.3f std %.3f min %.3f max %.3f" %
          (Y.mean(), Y.std(), Y.min(), Y.max()))

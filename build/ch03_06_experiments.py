"""第 3–6 讲实验：反馈线性化精度、能量爬升、滑模抗扰、切换瞬态。"""
import json
import os
import numpy as np

import plant as pl
import controllers as C
from dataset import DT, random_x0
from plant import P, wrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
res = {}

# ---------------------------------------------------------------- ch03
# 1) inverse-dynamics accuracy: command pdd=v through the inversion, measure actual
def inv_check():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        x = np.array([rng.uniform(-2, 2), rng.uniform(-3, 3),
                      rng.uniform(-3, 3), rng.uniform(-3, 3)])
        v = rng.uniform(-5, 5)
        u = C.cart_inverse_dynamics(x, v)
        # one tiny integration step, then estimate p_ddot from the plant
        h = 1e-4
        u_cl = np.clip(u, -P["u_max"], P["u_max"])
        x1 = pl.rk4(x, u_cl, h, substeps=1)
        pdd = (x1[2] - x[2]) / h
        errs.append(abs(pdd - np.clip(v, -P["u_max"], P["u_max"])))
    return float(np.max(errs)), float(np.mean(errs))


mx, mean = inv_check()
res["ch03"] = dict(inv_max=mx, inv_mean=mean)

# 2) force-domain (no inversion) vs acceleration-domain energy pump
def energy_bench(with_inv):
    best = None
    for kE in (2.0, 4.0, 6.0, 8.0):
        ok = 0
        for x0 in ([0.0, np.pi - 0.14, 0, 0], [0.5, 2.6, 0, 0], [-0.6, -2.9, 0, 0]):
            ctl = C.EnergyFBL(kE=kE) if with_inv else C.EnergySwingUp(kE=kE)
            X, U = C.rollout(ctl, x0, T=12.0)
            th = np.abs(wrap(X[:, 1]))
            if th[int(9 / DT):].min() < 0.3:
                ok += 1
        if best is None or ok > best[1]:
            best = (kE, ok)
    return dict(best_kE=best[0], reached=best[1], n=3)


res["ch03"]["inv_vs_direct"] = {"inversion": energy_bench(True), "direct": energy_bench(False)}

# ---------------------------------------------------------------- ch04
def energy_curve():
    ctl = C.build("hybrid_lqr")
    X, U = C.rollout(ctl, [0.0, np.pi - 0.14, 0.0, 0.0], T=12.0)
    E = np.array([pl.energy(x) for x in X])
    t = np.arange(len(E)) * DT
    idx = np.linspace(0, len(E) - 1, 400).astype(int)
    th = np.abs(wrap(X[:, 1]))
    up = np.where(th < 0.1)[0]
    return dict(t=[round(float(v), 3) for v in t[idx]],
                E=[round(float(v), 4) for v in E[idx]],
                p=[round(float(v), 3) for v in X[idx, 0]],
                th=[round(float(v), 3) for v in X[idx, 1]],
                u=[round(float(v), 3) for v in U[idx]],
                t_up=float(up[0] * DT) if len(up) else None,
                E_max=float(E.max()))


res["ch04"] = energy_curve()

# kE sweep
sweep = []
for kE in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
    sw = C.Hybrid(C.LQR(), P, swing=C.EnergyFBL(kE=kE), th_in=0.40, w_in=2.5)
    t_ups, pms = [], []
    for x0 in ([0.0, np.pi - 0.14, 0, 0], [0.8, 2.2, 0, 0], [-0.5, -2.6, 0, 0]):
        X, U = C.rollout(sw, x0, T=15.0)
        th = np.abs(wrap(X[:, 1]))
        uu = np.where(th < 0.1)[0]
        t_ups.append(float(uu[0] * DT) if len(uu) else 99.0)
        pms.append(float(np.abs(X[:, 0]).max()))
    sweep.append(dict(kE=kE, t_up=float(np.mean(t_ups)), p_max=float(np.mean(pms))))
res["ch04"]["kE_sweep"] = sweep

# ---------------------------------------------------------------- ch05
def smc_bench():
    rng = np.random.default_rng(2025)
    ics = [random_x0(rng, "hard") for _ in range(40)]
    out = {}
    for name, ctl in (("elqr", C.build("hybrid_lqr")), ("esmc", C.build("hybrid_smc"))):
        rows = []
        for x0 in ics:
            X, U = C.rollout(ctl, x0, T=12.0, dist=lambda t: 4.0 if 4 <= t < 8 else 0.0)
            th = np.abs(wrap(X[:, 1]))
            late = slice(int(9 / DT), None)
            rows.append((float(th[late].mean()), float(np.sqrt(np.mean(U ** 2)))))
        a = np.array(rows)
        out[name] = dict(th_mean=float(a[:, 0].mean()), th_max=float(a[:, 0].max()),
                         u_rms=float(a[:, 1].mean()))
    return out


res["ch05"] = smc_bench()

# disturbance response curves (time series)
def dist_curve(ctl):
    X, U = C.rollout(ctl, [0.0, 0.05, 0.0, 0.0], T=12.0,
                     dist=lambda t: 4.0 if 4 <= t < 8 else 0.0)
    idx = np.linspace(0, len(X) - 1, 400).astype(int)
    return dict(t=[round(float(i * DT), 3) for i in idx],
                th=[round(float(v), 4) for v in X[idx, 1]],
                u=[round(float(v), 3) for v in U[idx]])


res["ch05"]["curves"] = {"elqr": dist_curve(C.build("hybrid_lqr")),
                         "esmc": dist_curve(C.build("hybrid_smc"))}

# boundary-layer sweep for SMC
bl = []
for phi in (0.05, 0.15, 0.25, 0.5):
    ctl = C.Hybrid(C.SMC(phi=phi), P)
    X, U = C.rollout(ctl, [0.0, 0.3, 0.0, 0.0], T=8.0)
    late = slice(int(6 / DT), None)
    bl.append(dict(phi=phi, th_rms=float(np.sqrt(np.mean(X[late, 1] ** 2))),
                   u_rms=float(np.sqrt(np.mean(U[late] ** 2)))))
res["ch05"]["bl_sweep"] = bl

# ---------------------------------------------------------------- ch06
def switch_stats(mode):
    ctl = C.build("hybrid_lqr" if mode == "lqr" else "hybrid_smc")
    X, U = C.rollout(ctl, [0.0, np.pi - 0.14, 0.0, 0.0], T=12.0)
    d = np.abs(np.diff(U))
    return dict(switches=ctl.switches, du_max=float(d.max()),
                t_up=float(next((k * DT for k, v in enumerate(np.abs(wrap(X[:, 1]))) if v < 0.1), np.nan)))


res["ch06"] = {"elqr": switch_stats("lqr"), "esmc": switch_stats("smc")}

# threshold sweep: catch window vs success
th_sweep = []
for th_in in (0.25, 0.35, 0.45, 0.60):
    ok = 0
    for x0 in ([0.0, np.pi - 0.14, 0, 0], [0.8, 2.2, 0, 0], [-0.5, -2.6, 0, 0],
               [0.3, -2.0, 0, 0]):
        ctl = C.Hybrid(C.LQR(), P, swing=C.EnergyFBL(), th_in=th_in, w_in=2.5)
        X, U = C.rollout(ctl, x0, T=15.0)
        th = np.abs(wrap(X[:, 1]))
        if th[int(12 / DT):].max() < 0.05:
            ok += 1
    th_sweep.append(dict(th_in=th_in, ok=ok, n=4))
res["ch06"]["th_sweep"] = th_sweep

json.dump(res, open(os.path.join(OUT, "ch03_06.json"), "w"))
print(json.dumps({k: (list(v.keys()) if isinstance(v, dict) else type(v).__name__)
                  for k, v in res.items()}, ensure_ascii=False))
print("ch03 inv ok:", res["ch03"]["inv_max"], res["ch03"]["inv_mean"])
print("ch04 t_up:", res["ch04"]["t_up"], "E_max:", res["ch04"]["E_max"])
print("ch05:", res["ch05"]["elqr"], res["ch05"]["esmc"])
print("ch06:", res["ch06"])

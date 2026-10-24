"""第 2 讲实验：LQR 设计、吸引域、全局失败、Q/R 权衡。

输出 build/out/ch02.json，供讲义与视频使用。
"""
import json
import os
import numpy as np

import plant as pl
import controllers as C
import train as T
from dataset import random_x0, DT
from plant import P, wrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def settle(x0, K, T_=6.0, tol=0.02):
    """settle time, peak force, saturation fraction for a plain LQR rollout."""
    x = np.array(x0, float)
    n = int(T_ / DT)
    us, ts, ths = [], [], []
    for k in range(n):
        u = float(np.clip(-K @ x, -P["u_max"], P["u_max"]))
        us.append(u); ts.append(k * DT); ths.append(abs(x[1]))
        x = pl.rk4(x, u, DT)
        if abs(x[0]) > P["x_max"]:
            return None
    us = np.array(us); ths = np.array(ths)
    ok = np.where((ths < tol))[0]
    # first time after which it stays inside tol
    st = None
    for i in range(len(ths)):
        if ths[i] < tol and np.all(ths[i:] < tol * 3):
            st = i * DT; break
    return dict(settle=st, max_u=float(np.abs(us).max()),
                sat=float(np.mean(np.abs(us) > 0.98 * P["u_max"])),
                p_max=float(np.abs(x[0])))


def basin(K, th_lim=1.2, w_lim=4.0, nth=61, nw=81, T_=3.0):
    ths = np.linspace(-th_lim, th_lim, nth)
    ws = np.linspace(-w_lim, w_lim, nw)
    ok = np.zeros((nw, nth), bool)
    for j, w0 in enumerate(ws):
        for i, th0 in enumerate(ths):
            x = np.array([0.0, th0, 0.0, w0])
            good = True
            for k in range(int(T_ / DT)):
                u = float(np.clip(-K @ x, -P["u_max"], P["u_max"]))
                x = pl.rk4(x, u, DT)
                if abs(x[0]) > P["x_max"]:
                    good = False; break
            if good and abs(wrap(x[1])) < 0.05 and abs(x[0]) < 1.5:
                ok[j, i] = True
    return ths, ws, ok


def main():
    res = {}

    # ---------- 1. default design
    K, Pi = C.lqr_gain()
    A, B = pl.linearize()
    poles = np.linalg.eigvals(A - B @ K)
    res["design"] = dict(K=[float(v) for v in K[0]],
                         poles=[[float(p.real), float(p.imag)] for p in poles],
                         Q=[1, 40, 1, 2], R=0.05)
    print("K =", np.round(K[0], 4), " poles:", np.round(np.sort_complex(poles), 3))

    # ---------- 2. local response
    for a in (0.1, 0.25, 0.5):
        r = settle([0.0, a, 0.0, 0.0], K)
        res[f"local_{a}"] = r
        print(f"from {a} rad: settle={r['settle']} max_u={r['max_u']:.2f} "
              f"sat={r['sat']:.2f} p_max={r['p_max']:.2f}")

    # ---------- 3. basin
    ths, ws, ok = basin(K)
    res["basin"] = dict(ths=ths.tolist(), ws=ws.tolist(),
                        ok=ok.astype(int).tolist(),
                        frac=float(ok.mean()))
    # boundary: max |theta| reachable for w=0
    j0 = int(np.argmin(np.abs(ws)))
    idx = np.where(ok[j0])[0]
    res["basin"]["th_max_at_w0"] = float(max(abs(ths[idx]))) if len(idx) else 0.0
    print(f"basin (|th|<1.2, |w|<4): {100*ok.mean():.1f}%  th_max(w=0)={res['basin']['th_max_at_w0']:.2f} rad")

    # ---------- 4. global evaluation (60 ics)
    rng = np.random.default_rng(2025)
    ics = [random_x0(rng, "hard") for _ in range(60)]
    ctl = C.build("lqr")
    ok_n, rail, pms, ths_end = 0, 0, [], []
    for x0 in ics:
        X, U = T.rollout_any(ctl, x0, T=12.0)
        th = np.abs(wrap(X[:, 1]))
        late = slice(int(9.0 / DT), None)
        if th[late].max() < 0.15 and np.abs(X[late, 0]).max() < 1.5:
            ok_n += 1
        if np.abs(X[:, 0]).max() > P["x_max"] - 1e-9:
            rail += 1
        pms.append(float(np.abs(X[:, 0]).max())); ths_end.append(float(th[-1]))
    res["global"] = dict(success=ok_n / 60, rail=rail / 60,
                         p_max=float(np.mean(pms)), th_end=float(np.mean(ths_end)))
    print(f"global: success {ok_n/60:.2f}  rail {rail/60:.2f}  p_max {np.mean(pms):.2f}")

    # ---------- 5. Q/R trade-off
    designs = [("温和", [1, 10, 0.3, 0.5], 1.0),
               ("默认", [1, 40, 1, 2], 0.05),
               ("激进", [1, 120, 3, 5], 0.01)]
    rows = []
    for name, Q, R in designs:
        Kx, _ = C.lqr_gain(np.diag([float(v) for v in Q]), np.array([[R]]))
        p = np.linalg.eigvals(A - B @ Kx)
        r = settle([0.0, 0.25, 0.0, 0.0], Kx)
        ths2, ws2, ok2 = basin(Kx)
        rows.append(dict(name=name, Q=Q, R=R, K=[float(v) for v in Kx[0]],
                         poles=[[float(q.real), float(q.imag)] for q in p],
                         settle=(r["settle"] if r else None),
                         max_u=(r["max_u"] if r else None),
                         basin=float(ok2.mean())))
        print(f"{name:4s} K={np.round(Kx[0],2)} settle={r['settle'] if r else 'NA'} "
              f"basin={100*ok2.mean():.0f}%")

    res["designs"] = rows

    # ---------- 6. failure trajectory from hanging (for the video)
    X, U = T.rollout_any(ctl, [0.0, np.pi - 0.1, 0.0, 0.0], T=6.0)
    res["fail_traj"] = dict(t=[float(v) for v in np.arange(len(X)) * DT][::6],
                            th=[float(v) for v in X[:, 1]][::6],
                            p=[float(v) for v in X[:, 0]][::6],
                            u=[float(v) for v in U][::6])

    json.dump(res, open(os.path.join(OUT, "ch02.json"), "w"))
    print("saved", os.path.join(OUT, "ch02.json"))


if __name__ == "__main__":
    main()

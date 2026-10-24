"""Grid-search the swing-up gains so that the hybrid controller reliably catches."""
import numpy as np
import plant as pl
import controllers as C
from plant import P, wrap


def sim(kE, phi, kx, kd, x0, T=25.0, p=P, thi=0.5, wi=2.5, tho=0.85, wo=3.5):
    sw = C.EnergySwingUp(p, kE=kE, phi=phi, kx=kx, kd=kd)
    ctl = C.Hybrid(C.LQR(p), p, swing=sw, th_in=thi, w_in=wi, th_out=tho, w_out=wo)
    X, U = C.rollout(ctl, x0, T=T, p=p)
    th = np.abs(wrap(X[:, 1]))
    up = np.where(th < 0.1)[0]
    t_up = up[0] * p["dt"] if len(up) else np.inf
    tail = slice(int(15 / p["dt"]), None)
    return dict(t_up=t_up, th_end=th[-1], p_end=abs(X[-1, 0]), p_max=np.abs(X[:, 0]).max(),
                u_rms=float(np.sqrt(np.mean(U[tail] ** 2))), sw=ctl.switches, X=X, U=U)


if __name__ == "__main__":
    p = P
    best = []
    for kE in [1.5, 2.2, 3.0, 4.5]:
        for phi in [0.15, 0.35]:
            for kx, kd in [(0.15, 0.25), (0.35, 0.5), (0.6, 0.8)]:
                r = sim(kE, phi, kx, kd, [0.0, np.pi - 0.14, 0.0, 0.0])
                r2 = sim(kE, phi, kx, kd, [0.6, np.pi + 0.6, 0.0, 0.0])
                ok = r["t_up"] < 20 and r["p_max"] < 2.4 and r2["t_up"] < 20 and r2["p_max"] < 2.4
                print(f"kE={kE:3.1f} phi={phi:4.2f} kx={kx:4.2f} kd={kd:4.2f} | "
                      f"A tup={r['t_up']:5.2f} pmax={r['p_max']:5.2f} the={r['th_end']:.3f} | "
                      f"B tup={r2['t_up']:5.2f} pmax={r2['p_max']:5.2f} the={r2['th_end']:.3f} "
                      f"{'  <== OK' if ok else ''}")

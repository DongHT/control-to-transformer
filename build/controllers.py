"""Classical controllers for the cart-pole.

1) LQR          : linear optimal state feedback, valid only near the upright equilibrium
2) ENERGY+LQR   : nonlinear energy-shaping swing-up + LQR stabilization with hysteresis switching
3) ENERGY+SMC   : same swing-up + sliding-mode (robust nonlinear) stabilization
"""
import numpy as np
import plant as pl
from plant import P, wrap


# ---------------------------------------------------------------- LQR
def lqr_gain(Q=None, R=None, p=P):
    from scipy.linalg import solve_continuous_are
    A, B = pl.linearize(p)
    if Q is None:
        Q = np.diag([1.0, 40.0, 1.0, 2.0])
    if R is None:
        R = np.array([[0.05]])
    Pi = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ Pi)
    return K, Pi


class LQR:
    """u = -K x  (cart position referenced to p_ref)"""
    def __init__(self, p=P, p_ref=0.0, Q=None, R=None):
        self.K, self.Pi = lqr_gain(Q, R, p)
        self.p = p
        self.p_ref = p_ref
        self.name = "LQR"

    def reset(self):
        pass

    def __call__(self, x, t=0.0):
        e = np.array([x[0] - self.p_ref, x[1], x[2], x[3]])
        u = -float((self.K @ e).item() if hasattr(self.K @ e, "item") else self.K @ e)
        m = self.p["u_max"]
        return float(np.clip(u, -m, m))


# ------------------------------------------------- energy swing-up
class EnergySwingUp:
    """Astrom-Furuta energy shaping.

        E(x)  = 0.5*mp*l^2*w^2 + mp*g*l*(cos(theta)-1)      (E=0 <=> upright at rest)
        Ed    = target energy (0)
        u     = kE*(E-Ed)*tanh(w*cos(theta)/phi) - kx*p - kd*v

    The tanh() replaces sign() to avoid chattering; the linear terms gently
    regulate the cart so that the rail stays usable.
    """
    def __init__(self, p=P, kE=3.2, phi=0.35, kx=0.55, kd=0.75, Ed=0.0):
        self.p, self.kE, self.phi, self.kx, self.kd, self.Ed = p, kE, phi, kx, kd, Ed
        self.name = "Energy"

    def reset(self):
        pass

    def __call__(self, x, t=0.0):
        pos, th, v, w = x
        E = pl.energy(x, self.p)
        drive = np.tanh(w * np.cos(th) / self.phi)
        u = self.kE * (E - self.Ed) * drive - self.kx * pos - self.kd * v
        m = self.p["u_max"]
        return float(np.clip(u, -m, m))


def cart_inverse_dynamics(x, pdd_cmd, p=P):
    """Exact inversion of the CART channel: return the force u that produces
    p_ddot = pdd_cmd.  (The cart channel is fully actuated, so this is exact and
    non-singular: the denominator m_c + m_p*sin^2(theta) > 0.)"""
    _, th, v, w = x
    s, c = np.sin(th), np.cos(th)
    den = p["m_c"] + p["m_p"] * s * s
    return (pdd_cmd * den + p["b_c"] * v - p["m_p"] * p["l"] * s * w * w
            - p["b_p"] * c * w + p["m_p"] * p["g"] * s * c)


class EnergyFBL:
    """Energy shaping on top of partial feedback linearization.

    Inner loop : u = f^{-1}(x, v)  =>  p_ddot = v exactly
    Outer loop : v = kE*(E - Ed)*sat(w cos(theta)/phi) - kx*p - kd*p_dot

    dE/dt = -m*l*(w cos theta)*v, so with v = kE*(E-Ed) sat(w cos theta / phi)
    the energy error decays exponentially: (E-Ed)dot = -kE*m*l*|w cos theta|*(E-Ed).
    """
    def __init__(self, p=P, kE=6.0, phi=0.20, kx=1.30, kd=1.90, Ed=0.0):
        self.p, self.kE, self.phi, self.kx, self.kd, self.Ed = p, kE, phi, kx, kd, Ed
        self.name = "Energy"

    def reset(self):
        pass

    def __call__(self, x, t=0.0):
        pos, th, v, w = x
        E = pl.energy(x, self.p)
        drive = np.tanh(w * np.cos(th) / self.phi)
        pdd = self.kE * (E - self.Ed) * drive - self.kx * pos - self.kd * v
        u = cart_inverse_dynamics(x, pdd, self.p)
        m = self.p["u_max"]
        return float(np.clip(u, -m, m))


# ---------------------------------------------------- sliding mode
class SMC:
    """Sliding-mode stabilizer around the upright (robust nonlinear controller).

        sliding surface  s = w + lam*(theta - theta_ref)
        reaching law     sdot = -eta*sat(s/phi) - ks*s
        -> required cart acceleration p_ddot (collocated feedback linearization)
        then invert the cart channel to get the force u.

    * theta_ref = -clip(kp*p + kd*v) is a small "lean" bias: it is the only way to
      command the cart position through the same (single) input in a
      time-scale-separated way - the fast pole loop tracks the lean, the slow
      cart loop is then implicitly regulated.
    * the tanh/sat boundary layer removes chattering; eta provides the robustness
      to matched disturbances (extra force on the cart).
    * the 1/cos(theta) singularity of the collocated linearization is bounded.
    """
    def __init__(self, p=P, lam=6.0, eta=12.0, ks=1.0, phi=0.25, kp=0.18, kd=0.22):
        self.p, self.lam, self.eta, self.ks = p, lam, eta, ks
        self.phi, self.kp, self.kd = phi, kp, kd
        self.name = "SMC"

    def reset(self):
        pass

    def __call__(self, x, t=0.0):
        p = self.p
        pos, th, v, w = x
        s, c = np.sin(th), np.cos(th)
        c = np.sign(c) * max(abs(c), 0.25)          # bound the 1/cos singularity
        den = p["m_c"] + p["m_p"] * s * s
        th_ref = -np.clip(self.kp * pos + self.kd * v, -0.25, 0.25)
        sl = w + self.lam * (th - th_ref)
        sat = np.clip(sl / self.phi, -1.0, 1.0)
        pdd = (p["g"] * s + p["l"] * self.lam * w
               + p["l"] * self.eta * sat + p["l"] * self.ks * sl) / c
        u = (pdd * den + p["b_c"] * v - p["m_p"] * p["l"] * s * w * w
             - p["b_p"] * c * w + p["m_p"] * p["g"] * s * c)
        m = p["u_max"]
        return float(np.clip(u, -m, m))


# ------------------------------------------- hybrid nonlinear control
class Hybrid:
    """Energy swing-up + local stabilizer, with hysteresis switching.

    local in  : |theta| < th_in  and |w| < w_in
    local out : |theta| > th_out or  |w| > w_out
    """
    def __init__(self, local, p=P, swing=None, th_in=0.40, w_in=2.5, th_out=0.80, w_out=3.5):
        self.p = p
        self.swing = swing or EnergyFBL(p)
        self.local = local
        self.th_in, self.w_in, self.th_out, self.w_out = th_in, w_in, th_out, w_out
        self.name = "Energy+" + getattr(local, "name", "Local")
        self.mode = "swing"
        self.switches = 0

    def reset(self):
        self.mode = "swing"
        self.switches = 0

    def __call__(self, x, t=0.0):
        th, w = x[1], x[3]
        if self.mode == "swing":
            if abs(th) < self.th_in and abs(w) < self.w_in:
                self.mode = "local"
                self.switches += 1
                self.local.reset()
        else:
            if abs(th) > self.th_out or abs(w) > self.w_out:
                self.mode = "swing"
                self.swing.reset()
        c = self.local if self.mode == "local" else self.swing
        return c(x, t)


def build(name="hybrid_lqr", p=P):
    lqr = LQR(p)
    if name == "lqr":
        return lqr
    if name == "hybrid_lqr":
        return Hybrid(lqr, p)
    if name == "hybrid_smc":
        return Hybrid(SMC(p), p)
    if name == "smc":
        return SMC(p)
    if name == "energy":
        return EnergySwingUp(p)
    raise ValueError(name)


# --------------------------------------------------------- simulation
def rollout(ctrl, x0, T=12.0, dt=P["dt"], dist=None, p=P, rng=None):
    """Closed-loop rollout. dist(t) -> extra force on the cart (matched disturbance)."""
    n = int(T / dt)
    ctrl.reset()
    x = np.array(x0, dtype=float)
    X = np.zeros((n + 1, 4))
    U = np.zeros(n + 1)
    X[0] = x
    for k in range(n):
        u = ctrl(x, k * dt)
        d = 0.0 if dist is None else dist(k * dt)
        u_tot = np.clip(u + d, -p["u_max"], p["u_max"])
        U[k] = u_tot
        x = pl.rk4(x, u_tot, dt, p)
        X[k + 1] = x
    U[n] = U[n - 1]
    return X, U


if __name__ == "__main__":
    p = P
    K, Pi = lqr_gain()
    print("LQR gain K =", np.round(K, 3))
    A, B = pl.linearize()
    Acl = A - B @ K
    print("closed-loop poles:", np.round(np.linalg.eigvals(Acl), 3))

    for name, x0 in [("lqr", [0.0, 0.20, 0.0, 0.0]),
                     ("lqr", [0.0, np.pi - 0.1, 0.0, 0.0]),
                     ("hybrid_lqr", [0.0, np.pi, 0.0, 0.0]),
                     ("hybrid_smc", [0.0, np.pi, 0.0, 0.0])]:
        X, U = rollout(build(name), x0, T=15)
        th = np.abs(wrap(X[:, 1]))
        up = np.where(th < 0.1)[0]
        t_up = (up[0] * p["dt"]) if len(up) else float("nan")
        tail = slice(int(9 / p["dt"]), None)
        print(f"{name:12s} x0_th={x0[1]:.2f}  t_up={t_up:5.2f}s  "
              f"|th|_end={th[-1]:.4f}  |p|_end={abs(X[-1,0]):.3f}  "
              f"|u|rms={np.sqrt(np.mean(U[tail]**2)):5.2f}")

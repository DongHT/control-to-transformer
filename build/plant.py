"""Cart-pole (inverted pendulum on a cart) nonlinear plant.

Model derived from Lagrangian mechanics with:
  - theta measured from the UPRIGHT vertical, positive clockwise
  - pole treated as a point mass m_p at distance l from the pivot
  - viscous friction b_c on the cart rail, b_p at the pivot

State x = [p, theta, p_dot, theta_dot]
Input  u = horizontal force applied to the cart [N]
"""
import numpy as np

PARAMS = dict(
    m_c=1.0,      # cart mass [kg]
    m_p=0.25,     # pole (bob) mass [kg]
    l=0.6,        # distance pivot -> pole CoM [m]
    g=9.81,       # gravity [m/s^2]
    b_c=0.12,     # cart viscous friction [N.s/m]
    b_p=0.004,    # pivot viscous friction [N.m.s/rad]
    u_max=12.0,   # actuator saturation [N]
    dt=0.02,      # control / integration step [s]
    x_max=2.4,    # rail half-length [m]
)
P = PARAMS


def f(x, u, p=P):
    """Continuous-time dynamics xdot = f(x,u)."""
    pos, th, v, w = x
    s, c = np.sin(th), np.cos(th)
    M = p["m_c"] + p["m_p"]
    den = M - p["m_p"] * c * c          # = m_c + m_p*sin^2(theta) > 0
    # Lagrange equations solved for (p_ddot, theta_ddot)
    pdd = (u - p["b_c"] * v + p["m_p"] * p["l"] * s * w * w
           + p["b_p"] * c * w - p["m_p"] * p["g"] * s * c) / den
    wdd = (p["g"] * s - c * pdd - p["b_p"] * w / (p["m_p"] * p["l"])) / p["l"]
    return np.array([v, w, pdd, wdd])


def rk4(x, u, dt, p=P, substeps=2):
    h = dt / substeps
    for _ in range(substeps):
        k1 = f(x, u, p)
        k2 = f(x + 0.5 * h * k1, u, p)
        k3 = f(x + 0.5 * h * k2, u, p)
        k4 = f(x + h * k3, u, p)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def energy(x, p=P):
    """Mechanical energy of the pole measured relative to the upright equilibrium.
    E = 0.5*m*l^2*w^2 + m*g*l*(cos(theta) - 1)  ->  E = 0 at upright rest."""
    _, th, _, w = x
    return 0.5 * p["m_p"] * p["l"] ** 2 * w * w + p["m_p"] * p["g"] * p["l"] * (np.cos(th) - 1.0)


def linearize(p=P):
    """A, B of the linearization at the upright equilibrium (theta=0, all rates 0)."""
    M, mp, l, g = p["m_c"] + p["m_p"], p["m_p"], p["l"], p["g"]
    bc, bp = p["b_c"], p["b_p"]
    # theta=0 -> s=0, c=1, den = m_c
    den = p["m_c"]
    # pdd  = (u - bc*v + bp*w - mp*g*th)/m_c
    # wdd  = (g*th - pdd - bp*w/(mp*l))/l
    # note the two coupling terms that are easy to miss:
    #   d(pdd)/d(th) = -mp*g/m_c          (the pole pulls the cart)
    #   d(wdd)/d(w)  = -(bp/m_c + bp/(mp*l))/l   (pivot friction enters twice)
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -mp * g / den, -bc / den, bp / den],
        [0.0, g / l, bc / (l * den), -(bp / den + bp / (mp * l)) / l],
    ])
    B = np.array([[0.0], [0.0], [1.0 / den], [-1.0 / (l * den)]])
    return A, B


def wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi

"""第 7–10 讲用的理论量：仿真的分布漂移曲线、参数分解等。"""
import json
import os
import numpy as np

import plant as pl
import controllers as C
import train as T
from dataset import DT, random_x0
from plant import P, wrap
from model import Policy
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load_policy(path, strip=""):
    sd = torch.load(path)
    if strip:
        sd = {k[len(strip):]: v for k, v in sd.items() if k.startswith(strip)}
    m = Policy(); m.load_state_dict(sd); m.eval()
    return m


def run_pair(policy_or_none, x0, T_=8.0, noise_eps=0.0, rng=None):
    """run expert and a policy in parallel; return per-step angle deviation."""
    expert = C.build("hybrid_lqr")
    expert.reset()
    ctl = T.PolicyCtl(policy_or_none) if policy_or_none is not None else None
    if ctl:
        ctl.reset()
    x_e = np.array(x0, float)
    x_p = np.array(x0, float)
    dev = []
    for k in range(int(T_ / DT)):
        ue = expert(x_e, k * DT)
        xe = pl.rk4(x_e, ue, DT)
        if abs(xe[0]) > P["x_max"]:
            xe[0] = np.sign(xe[0]) * P["x_max"]; xe[2] = 0.0
        if ctl is None:
            up = ue + (rng.normal(0, noise_eps) if noise_eps else 0.0)
            up = float(np.clip(up, -P["u_max"], P["u_max"]))
        else:
            up = ctl(x_p, k * DT)
        xp = pl.rk4(x_p, np.clip(up, -P["u_max"], P["u_max"]), DT)
        if abs(xp[0]) > P["x_max"]:
            xp[0] = np.sign(xp[0]) * P["x_max"]; xp[2] = 0.0
        x_e, x_p = xe, xp
        dev.append(float(abs(wrap(x_p[1] - x_e[1]))))
    return np.array(dev)


def main():
    rng = np.random.default_rng(0)
    res = {}

    m_bc = load_policy(os.path.join(OUT, "policy.pt"))
    m_dg = load_policy(os.path.join(OUT, "policy.pt"))     # same file: BC+DAgger 最终版
    x0 = [0.0, np.pi - 0.14, 0.0, 0.0]

    d_expert_noise = run_pair(None, x0, noise_eps=0.45, rng=rng)   # 同 ε，但每步被专家拉回
    d_bc = run_pair(m_bc, x0)
    idx = np.linspace(0, len(d_bc) - 1, 300).astype(int)
    res["drift"] = dict(
        t=[round(float(i * DT), 3) for i in idx],
        expert_noise=[round(float(v), 4) for v in d_expert_noise[idx]],
        bc=[round(float(v), 4) for v in d_bc[idx]],
        bc_final=float(d_bc[-1]),
    )
    print("deviation at 8 s: expert+noise %.3f rad, BC %.3f rad" % (d_expert_noise[-1], d_bc[-1]))

    # 误差放大：ε 与首次越界时间
    amp = []
    for eps in (0.1, 0.2, 0.45, 0.8):
        d = run_pair(None, x0, noise_eps=eps, rng=np.random.default_rng(1))
        amp.append(dict(eps=eps, dev_end=float(d[-1])))
    res["amplification"] = amp
    for a in amp:
        print("eps %.2f -> dev_end %.3f" % (a["eps"], a["dev_end"]))

    json.dump(res, open(os.path.join(OUT, "ch07_10.json"), "w"))


if __name__ == "__main__":
    main()

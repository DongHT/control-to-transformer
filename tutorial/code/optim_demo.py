"""
optim_demo.py —— 优化器对比实验（真实计算，不画示意图）

产物 out/optim.json：
  · toy:   二维病态二次型 0.5(x² + κy²) 上五种优化器的真实轨迹
  · real:  同一模型、同一初始化，五种优化器各训 3 epoch 的真实损失曲线
  · step:  第一步时每个参数张量在 SGD 与 Adam 下的有效步长
  · bias:  1−β₁ᵗ 与 1−β₂ᵗ 的修正曲线
  · sched: 本课用的余弦学习率曲线
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "build"))

from tinytf_numpy import TinyTF, mse_loss          # noqa: E402


# --------------------------------------------------------------------------- 优化器
class Opt:
    """统一接口：step(params, grads)"""

    def __init__(self, p, lr, kind, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4, mom=0.9):
        self.lr, self.kind, self.b1, self.b2, self.eps, self.wd, self.mom = \
            lr, kind, b1, b2, eps, wd, mom
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def step(self, p, g, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k in g:
            if self.kind == "sgd":
                p[k] -= lr * g[k]
            elif self.kind == "momentum":
                self.v[k] = self.mom * self.v[k] + g[k]
                p[k] -= lr * self.v[k]
            elif self.kind == "rmsprop":
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
                p[k] -= lr * g[k] / (np.sqrt(self.v[k]) + self.eps)
            elif self.kind in ("adam", "adamw"):
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
                mh = self.m[k] / (1 - self.b1 ** self.t)
                vh = self.v[k] / (1 - self.b2 ** self.t)
                upd = mh / (np.sqrt(vh) + self.eps)
                if self.kind == "adam":       # 耦合式：L2 加进梯度
                    upd = upd + self.wd * p[k]
                else:                          # AdamW：解耦
                    p[k] *= (1 - lr * self.wd) if not k.endswith(("_g", "_b")) else 1.0
                p[k] -= lr * upd


# --------------------------------------------------------------------------- toy 2D
def toy(kappa=50.0, steps=60, start=(1.0, 1.0)):
    """f(x,y)=0.5(x² + κ y²)。返回每种优化器的轨迹（真实迭代）。"""
    def grad(p):
        return np.array([p[0], kappa * p[1]])

    def loss(p):
        return 0.5 * (p[0] ** 2 + kappa * p[1] ** 2)

    runs = {}
    cfg = {"sgd": dict(lr=0.018), "momentum": dict(lr=0.006, mom=0.9),
           "rmsprop": dict(lr=0.15), "adam": dict(lr=0.15)}
    for kind, kw in cfg.items():
        p = {"x": np.array(start, float)}
        o = Opt(p, kind=kind, **kw)
        path, losses = [p["x"].copy()], [loss(p["x"])]
        for _ in range(steps):
            o.step(p, {"x": grad(p["x"])})
            path.append(p["x"].copy())
            losses.append(loss(p["x"]))
        runs[kind] = dict(path=[[float(a), float(b)] for a, b in path],
                          loss=[float(v) for v in losses])
    return runs, kappa


# --------------------------------------------------------------------------- 真实模型
def get_batch(n=8192):
    cache = os.path.join(OUT, "data_cache.npz")
    d = np.load(cache)
    X, Y = d["X"], d["Y"]
    idx = np.random.default_rng(0).choice(len(X), n, replace=False)
    return X[idx].astype(np.float64), Y[idx].astype(np.float64)


def real_runs(epochs=3, bs=256):
    """五种优化器，同一初始化、同一批数据顺序。"""
    X, Y = get_batch()
    base = TinyTF(seed=11).p
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    res = {}
    first_grads = None
    # 名字 → (优化器, 学习率)。SGD 给两个学习率：合适的一个 + 与 Adam 同量级的一个
    for name, kind, lr in (("sgd", "sgd", 1e-3), ("sgd_big", "sgd", 2e-2),
                           ("momentum", "momentum", 1e-3), ("rmsprop", "rmsprop", 2e-3),
                           ("adam", "adam", 2e-3), ("adamw", "adamw", 2e-3)):
        m = TinyTF(seed=11)
        m.p = {k: v.copy() for k, v in base.items()}
        o = Opt(m.p, lr=lr, kind=kind)
        losses, steps = [], 0
        for ep in range(epochs):
            tot, nb = 0.0, 0
            for i in range(0, len(X) - bs, bs):
                idx = order[i:i + bs]
                u, c = m.forward(X[idx], return_cache=True)
                l, dU = mse_loss(u, Y[idx])
                g = m.backward(dU, c)
                if first_grads is None:            # 记录第一步梯度
                    first_grads = {k: v.copy() for k, v in g.items()}
                o.step(m.p, g)
                tot += l
                nb += 1
                steps += 1
            losses.append(tot / nb)
        res[name] = dict(lr=lr, loss=[float(v) for v in losses], steps=steps)
        print(f"  {name:9s} lr={lr:<7g} loss={losses[0]:.4f} → {losses[-1]:.4f}")
    return res, first_grads, base


def step_sizes(grads, base, lr_sgd=0.02, lr_adam=2e-3):
    """第一步：每个张量在 SGD 与 Adam 下的 |Δθ|。"""
    rows = []
    for k, g in grads.items():
        sgd = lr_sgd * float(np.abs(g).mean())
        # Adam 第一步：m=(1-β1)g, v=(1-β2)g², 修正后 ≈ g/|g| （逐元素），平均 ≈ 1
        adam = lr_adam * 1.0
        rows.append(dict(name=k, gnorm=float(np.linalg.norm(g)),
                         gmean=float(np.abs(g).mean()), sgd=sgd, adam=adam))
    rows.sort(key=lambda r: -r["gnorm"])
    return rows


def main():
    print("[toy] 二维病态二次型 κ=50")
    runs, kappa = toy()
    for k, v in runs.items():
        print(f"  {k:9s} loss {v['loss'][0]:.3e} → {v['loss'][-1]:.3e}")

    print("[real] 真实模型，五种优化器各 3 epoch")
    real, grads, base = real_runs()
    steps = step_sizes(grads, base)

    t = np.arange(1, 401)
    bias = dict(t=[int(v) for v in t],
                m=[float(1 - 0.9 ** v) for v in t],
                v=[float(1 - 0.999 ** v) for v in t])

    E = 12
    sched = dict(ep=list(range(E)),
                 cosine=[float(2e-3 * 0.5 * (1 + math.cos(math.pi * e / E))) for e in range(E)])

    json.dump(dict(toy=runs, kappa=kappa, real=real, steps=steps,
                   bias=bias, sched=sched),
              open(os.path.join(OUT, "optim.json"), "w"))
    print("\n[step] 第一步各张量的有效步长（前 6）:")
    for r in steps[:6]:
        print(f"  {r['name']:22s} ‖g‖={r['gnorm']:8.2f}  SGD={r['sgd']:.2e}  Adam={r['adam']:.2e}")
    print("\n→ out/optim.json")


if __name__ == "__main__":
    main()

"""
train_demo.py —— 用同一份数据、同一套初始化，分别用 numpy 与 torch 训练，
然后对比：损失曲线、耗时、闭环成功率、注意力形态。

产物：
  out/train_demo.json    曲线与指标
  out/fig_loss.png       损失曲线（numpy vs torch）
  out/fig_attention.png  注意力热图（摆起 vs 稳摆）
  out/tiny_numpy.npz     numpy 训练好的权重（供文档/视频引用）
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # 仓库根
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(os.path.dirname(HERE), "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BUILD)
sys.path.insert(0, HERE)

from tinytf_numpy import TinyTF as NpTF, mse_loss, AdamW      # noqa: E402
from tinytf_torch import TinyTF as TorchTF, state_from_numpy  # noqa: E402

import torch                                                  # noqa: E402
import dataset as D                                           # noqa: E402


def get_data():
    cache = os.path.join(OUT, "data_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["Y"]
    X, Y = D.gen(n_ep=300, T=8.0, noise=1.5, seed=1)
    np.savez_compressed(cache, X=X, Y=Y)
    return X, Y


def split(X, Y, val_frac=0.08, seed=0):
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    nv = max(2000, int(val_frac * n))
    return X[idx[nv:]], Y[idx[nv:]], X[idx[:nv]], Y[idx[:nv]]


def batches(n, bs, rng):
    order = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        yield order[i:i + bs]


def eval_np(model, X, Y):
    pred = model.forward(X)
    return float(0.5 * np.mean((pred - Y) ** 2))


def main(epochs=12, bs=128, lr=2e-3):
    X, Y = get_data()
    Xtr, Ytr, Xva, Yva = split(X, Y)
    print(f"数据: {Xtr.shape[0]} 训练 / {Xva.shape[0]} 验证, 窗口 {X.shape[1]}×{X.shape[2]}")

    res = {"epochs": epochs, "batch": bs, "lr": lr,
           "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0])}

    # ---------------- numpy 训练
    np_model = NpTF(seed=3)
    opt = AdamW(np_model.p, lr=lr, wd=1e-4)
    rng = np.random.default_rng(0)
    hist_np = []
    t0 = time.time()
    for ep in range(epochs):
        cur_lr = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        tot, nb = 0.0, 0
        for idx in batches(len(Xtr), bs, rng):
            u, c = np_model.forward(Xtr[idx], return_cache=True)
            l, dU = mse_loss(u, Ytr[idx])
            g = np_model.backward(dU, c)
            opt.step(np_model.p, g, lr=cur_lr)
            tot += l
            nb += 1
        hist_np.append(dict(ep=ep + 1, train=tot / nb, val=eval_np(np_model, Xva, Yva)))
        print(f"  [numpy] epoch {ep+1:2d}/{epochs}  train={tot/nb:.5f}  val={hist_np[-1]['val']:.5f}")
    t_np = time.time() - t0

    # ---------------- torch 训练（同一初始化）
    tm = TorchTF()
    tm.load_state_dict(state_from_numpy(np_model.__class__(seed=3).p, dtype=torch.float32))
    opt2 = torch.optim.AdamW(tm.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    Xtr_t, Ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    Xva_t, Yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32)
    hist_t = []
    t0 = time.time()
    for ep in range(epochs):
        cur_lr = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        for gparam in opt2.param_groups:
            gparam["lr"] = cur_lr
        tot, nb = 0.0, 0
        for idx in batches(len(Xtr), bs, rng):
            u = tm(Xtr_t[idx])
            l = 0.5 * ((u - Ytr_t[idx]) ** 2).mean()
            opt2.zero_grad()
            l.backward()
            opt2.step()
            tot += float(l)
            nb += 1
        with torch.no_grad():
            val = float(0.5 * ((tm(Xva_t) - Yva_t) ** 2).mean())
        hist_t.append(dict(ep=ep + 1, train=tot / nb, val=val))
        print(f"  [torch] epoch {ep+1:2d}/{epochs}  train={tot/nb:.5f}  val={val:.5f}")
    t_t = time.time() - t0

    res.update(numpy=hist_np, torch=hist_t, t_numpy=t_np, t_torch=t_t,
               speedup=t_np / t_t, val_numpy=hist_np[-1]["val"], val_torch=hist_t[-1]["val"])

    # ---------------- 保存权重
    np.savez(os.path.join(OUT, "tiny_numpy.npz"), **np_model.p)
    torch.save(tm.state_dict(), os.path.join(OUT, "tiny_torch.pt"))

    # ---------------- 闭环成功率（用 numpy 模型）
    from plant import P, wrap
    import plant as pl
    from dataset import features_row, DT
    T = 24

    def rollout(x0, T_=12.0):
        x = np.array(x0, float)
        buf = None
        uprev = 0.0
        rail = False
        for k in range(int(T_ / DT)):
            f = features_row(x, uprev)
            buf = np.tile(f, (T, 1)) if buf is None else np.concatenate([buf[1:], f[None]])
            u = float(np_model.forward(buf[None].astype(np.float64))[0])
            uprev = u
            x = pl.rk4(x, float(np.clip(u, -P["u_max"], P["u_max"])), DT)
            if abs(x[0]) > P["x_max"]:
                x[0] = np.sign(x[0]) * P["x_max"]
                x[2] = 0.0
                rail = True
        th = abs(wrap(x[1]))
        return th < 0.15 and abs(x[0]) < 1.5 and not rail

    rng = np.random.default_rng(2025)
    ics = [D.random_x0(rng, "hard") for _ in range(30)]
    ok = sum(rollout(x0) for x0 in ics)
    res["success"] = ok / len(ics)
    print(f"\n闭环成功率（30 组随机初值）: {ok}/{len(ics)}")

    json.dump(res, open(os.path.join(OUT, "train_demo.json"), "w"), indent=1)
    print(f"\n耗时: numpy {t_np:.1f}s / torch {t_t:.1f}s → {t_np/t_t:.1f}× 加速")
    print(f"最终验证损失: numpy {hist_np[-1]['val']:.6f} / torch {hist_t[-1]['val']:.6f}")


if __name__ == "__main__":
    main(epochs=int(os.environ.get("EPOCHS", 12)))

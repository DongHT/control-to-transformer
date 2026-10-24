"""
parity_test.py —— 手写反向 vs autograd 的逐位对齐检验

这是整套教程的"可信度基石"：如果纯 numpy 手推的梯度与 torch autograd
差在 1e-6 以上，说明公式或实现有错。

输出：
  1. 前向输出最大偏差
  2. 每个参数张量的梯度最大偏差与相对误差
  3. 有限差分（中心差分）对少数标量的独立交叉验证
"""

import numpy as np
import torch

from tinytf_numpy import TinyTF as NpTF, mse_loss
from tinytf_torch import TinyTF as TorchTF, state_from_numpy


def main():
    rng = np.random.default_rng(0)
    B, T, F = 5, 24, 6
    x = rng.standard_normal((B, T, F)).astype(np.float64)
    target = rng.uniform(-6, 6, size=B)

    # ---------- numpy 版
    m = NpTF(T=T, F=F, seed=7)
    u_np, cache = m.forward(x, return_cache=True)
    loss_np, dU = mse_loss(u_np, target)
    g_np = m.backward(dU, cache)

    # ---------- torch 版（装同一套权重）
    tm = TorchTF(T=T, F=F).double()
    tm.load_state_dict(state_from_numpy(m.p, dtype=torch.float64))
    xt = torch.tensor(x)
    u_t = tm(xt)
    loss_t = 0.5 * ((u_t - torch.tensor(target)) ** 2).mean()
    tm.zero_grad()
    loss_t.backward()

    print("=" * 72)
    print("① 前向输出")
    print(f"   numpy: {np.round(u_np, 6)}")
    print(f"   torch: {np.round(u_t.detach().numpy(), 6)}")
    print(f"   最大偏差 = {np.abs(u_np - u_t.detach().numpy()).max():.3e}")
    print(f"   损失  numpy={loss_np:.10f}  torch={loss_t.item():.10f}")

    print("=" * 72)
    print("② 反向梯度：逐参数对比")
    sd = dict(tm.named_parameters())
    name_map = {"W_in": "inp.weight", "b_in": "inp.bias", "pos": "pos",
                "ln_g": "ln.weight", "ln_b": "ln.bias",
                "w_head": "head.weight", "b_head": "head.bias"}
    for l in range(2):
        k, pre = f"l{l}.", f"blocks.{l}."
        name_map.update({
            k + "ln1_g": pre + "ln1.weight", k + "ln1_b": pre + "ln1.bias",
            k + "W_qkv": pre + "qkv.weight", k + "b_qkv": pre + "qkv.bias",
            k + "W_proj": pre + "proj.weight", k + "b_proj": pre + "proj.bias",
            k + "ln2_g": pre + "ln2.weight", k + "ln2_b": pre + "ln2.bias",
            k + "W1": pre + "fc1.weight", k + "b1": pre + "fc1.bias",
            k + "W2": pre + "fc2.weight", k + "b2": pre + "fc2.bias"})

    worst = 0.0
    print(f"   {'参数':22s} {'max|Δ|':>10s} {'‖g_torch‖':>12s}")
    for k in sorted(g_np):
        gt = sd[name_map[k]].grad.detach().numpy()
        d = np.abs(g_np[k] - gt).max()
        worst = max(worst, d)
        print(f"   {k:22s} {d:10.2e} {np.linalg.norm(gt):12.3f}")
    print(f"\n   全部参数的最大梯度偏差 = {worst:.3e}")

    # ---------- 有限差分交叉验证（与 autograd 无关的第三方证据）
    print("=" * 72)
    print("③ 中心差分交叉验证（数值梯度，步长 1e-6）")
    eps = 1e-6
    for key, idx in (("l0.W1", (3, 7)), ("l1.W_qkv", (11, 5)), ("w_head", (0, 9))):
        p0 = m.p[key][idx]
        m.p[key][idx] = p0 + eps
        lp, _ = mse_loss(m.forward(x), target)
        m.p[key][idx] = p0 - eps
        lm, _ = mse_loss(m.forward(x), target)
        m.p[key][idx] = p0
        num = (lp - lm) / (2 * eps)
        print(f"   {key}{idx}: 数值={num:+.8e}  手写={g_np[key][idx]:+.8e}  "
              f"相对误差={abs(num - g_np[key][idx]) / max(1e-12, abs(num)):.2e}")

    print("=" * 72)
    ok = worst < 1e-6
    print("结论:", "✅ 手写反向与 autograd 一致（<1e-6）" if ok else "❌ 存在偏差，需检查公式")


if __name__ == "__main__":
    main()

"""
tinytf_torch.py —— 与 numpy 版逐位等价的 torch 参考实现

两个版本共享同一套权重布局（见 state_from_numpy），用于：
1. 验证手写反向是否正确（autograd 作为基准）
2. 作为训练/导出用的"生产版"（快 10~30 倍）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d=32, heads=4, ff=64):
        super().__init__()
        self.h, self.nh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, z, mask):
        B, T, d = z.shape
        h = self.ln1(z)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # (B,T,d) -> (B,heads,T,nh)
        q = q.view(B, T, self.h, self.nh).transpose(1, 2)
        k = k.view(B, T, self.h, self.nh).transpose(1, 2)
        v = v.view(B, T, self.h, self.nh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.nh)      # (B,heads,T,T)
        s = s.masked_fill(mask, -1e9)
        a = torch.softmax(s, dim=-1)
        y = (a @ v).transpose(1, 2).reshape(B, T, d)
        z = z + self.proj(y)
        z = z + self.fc2(F.gelu(self.fc1(self.ln2(z))))
        return z


class TinyTF(nn.Module):
    """最小因果 Transformer：d=32, 4 头, ff=64, 2 层, 参数量 18,177"""

    def __init__(self, T=24, F=6, d=32, heads=4, ff=64, layers=2, u_max=12.0):
        super().__init__()
        self.T, self.u_max = T, u_max
        self.inp = nn.Linear(F, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        """x: (B,T,F) -> u: (B,)"""
        z = self.inp(x) + self.pos
        for blk in self.blocks:
            z = blk(z, self.mask)
        r = self.head(self.ln(z[:, -1]))
        return (self.u_max * torch.tanh(r)).squeeze(-1)


# --------------------------------------------------------------------------- 权重互转
def state_from_numpy(p, layers=2, dtype=torch.float32):
    """numpy 参数字典 -> torch state_dict（键名对齐）"""
    sd = {"inp.weight": p["W_in"], "inp.bias": p["b_in"], "pos": p["pos"],
          "ln.weight": p["ln_g"], "ln.bias": p["ln_b"],
          "head.weight": p["w_head"], "head.bias": p["b_head"]}
    for l in range(layers):
        k, pre = f"l{l}.", f"blocks.{l}."
        sd[pre + "ln1.weight"], sd[pre + "ln1.bias"] = p[k + "ln1_g"], p[k + "ln1_b"]
        sd[pre + "qkv.weight"], sd[pre + "qkv.bias"] = p[k + "W_qkv"], p[k + "b_qkv"]
        sd[pre + "proj.weight"], sd[pre + "proj.bias"] = p[k + "W_proj"], p[k + "b_proj"]
        sd[pre + "ln2.weight"], sd[pre + "ln2.bias"] = p[k + "ln2_g"], p[k + "ln2_b"]
        sd[pre + "fc1.weight"], sd[pre + "fc1.bias"] = p[k + "W1"], p[k + "b1"]
        sd[pre + "fc2.weight"], sd[pre + "fc2.bias"] = p[k + "W2"], p[k + "b2"]
    return {k: torch.tensor(v, dtype=dtype) for k, v in sd.items()}


def numpy_from_state(sd, layers=2):
    """torch state_dict -> numpy 参数字典"""
    import numpy as np
    p = {"W_in": sd["inp.weight"].numpy(), "b_in": sd["inp.bias"].numpy(),
         "pos": sd["pos"].numpy(), "ln_g": sd["ln.weight"].numpy(),
         "ln_b": sd["ln.bias"].numpy(), "w_head": sd["head.weight"].numpy(),
         "b_head": sd["head.bias"].numpy()}
    for l in range(layers):
        k, pre = f"l{l}.", f"blocks.{l}."
        p[k + "ln1_g"], p[k + "ln1_b"] = sd[pre + "ln1.weight"].numpy(), sd[pre + "ln1.bias"].numpy()
        p[k + "W_qkv"], p[k + "b_qkv"] = sd[pre + "qkv.weight"].numpy(), sd[pre + "qkv.bias"].numpy()
        p[k + "W_proj"], p[k + "b_proj"] = sd[pre + "proj.weight"].numpy(), sd[pre + "proj.bias"].numpy()
        p[k + "ln2_g"], p[k + "ln2_b"] = sd[pre + "ln2.weight"].numpy(), sd[pre + "ln2.bias"].numpy()
        p[k + "W1"], p[k + "b1"] = sd[pre + "fc1.weight"].numpy(), sd[pre + "fc1.bias"].numpy()
        p[k + "W2"], p[k + "b2"] = sd[pre + "fc2.weight"].numpy(), sd[pre + "fc2.bias"].numpy()
    return p


def count_params(model=None):
    m = model or TinyTF()
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = TinyTF()
    print("torch 版参数量:", count_params(m))
    x = torch.randn(2, 24, 6)
    print("输出:", m(x).detach().numpy().round(4))

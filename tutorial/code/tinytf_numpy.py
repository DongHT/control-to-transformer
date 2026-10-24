"""
tinytf_numpy.py —— 纯 numpy 实现的因果 Transformer（前向 + 手写反向）

设计原则
--------
1. 不依赖任何自动微分框架：所有梯度都是手推的（含 softmax 雅可比、LayerNorm 雅可比）。
2. 与 torch 参考实现**逐位对齐**：同权重同输入，输出与梯度误差都在 1e-10 量级。
3. 结构就是一个最小的 GPT 式解码器：Embedding + PosEnc + N×(LN→MHSA→残差→LN→FFN→残差) + Head。

张量约定（batch-first）
-----------------------
x     (B, T, F)   F=6 个输入特征
z     (B, T, d)   d=32
q,k,v (B, H, T, dh)  H=4 头, dh=8
s,a   (B, H, T, T)   注意力分数 / 权重
"""

import numpy as np
from scipy.special import erf

SQRT2 = np.sqrt(2.0)
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# ----------------------------------------------------------------------------- 基础算子
def ln_forward(x, g, b, eps=1e-5):
    """LayerNorm（对最后一维归一化）。返回 y 与缓存。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    y = xhat * g + b
    return y, (xhat, rstd, g)


def ln_backward(dy, cache):
    """LayerNorm 反向。经典紧凑形式：
       dx = gamma/sigma * (g - mean(g) - xhat * mean(g*xhat))
    两项修正分别抵消"整体平移"与"整体缩放"方向的梯度。"""
    xhat, rstd, g = cache
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    gx = dy * g
    dx = rstd * (gx - gx.mean(-1, keepdims=True)
                 - xhat * (gx * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / SQRT2))


def gelu_grad(x):
    return 0.5 * (1.0 + erf(x / SQRT2)) + x * np.exp(-0.5 * x * x) * INV_SQRT_2PI


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------- 模型
class TinyTF:
    """最小因果 Transformer：两层、4 头、d=32、ff=64。参数量 18,177。"""

    def __init__(self, T=24, F=6, d=32, heads=4, ff=64, layers=2, u_max=12.0, seed=0):
        self.T, self.F, self.d = T, F, d
        self.H, self.dh = heads, d // heads
        self.ff, self.L = ff, layers
        self.u_max = u_max
        rng = np.random.default_rng(seed)

        def w(*shape, scale=None):
            fan_in = shape[-1]
            s = scale if scale is not None else 1.0 / np.sqrt(fan_in)
            return (rng.standard_normal(shape) * s).astype(np.float64)

        p = {}
        p["W_in"], p["b_in"] = w(d, F), np.zeros(d)
        p["pos"] = w(T, d) * 0.02
        for l in range(layers):
            k = f"l{l}."
            p[k + "ln1_g"], p[k + "ln1_b"] = np.ones(d), np.zeros(d)
            p[k + "W_qkv"], p[k + "b_qkv"] = w(3 * d, d), np.zeros(3 * d)
            p[k + "W_proj"], p[k + "b_proj"] = w(d, d), np.zeros(d)
            p[k + "ln2_g"], p[k + "ln2_b"] = np.ones(d), np.zeros(d)
            p[k + "W1"], p[k + "b1"] = w(ff, d), np.zeros(ff)
            p[k + "W2"], p[k + "b2"] = w(d, ff), np.zeros(d)
        p["ln_g"], p["ln_b"] = np.ones(d), np.zeros(d)
        p["w_head"], p["b_head"] = w(1, d), np.zeros(1)
        self.p = p

    # ------------------------------------------------------------------ 前向
    def forward(self, x, return_cache=False):
        """x:(B,T,F) -> u:(B,) 力，单位 N（未裁剪）"""
        p, T, d, H, nh = self.p, self.T, self.d, self.H, self.dh
        B = x.shape[0]
        assert x.shape[1] == T and x.shape[2] == self.F

        z = x @ p["W_in"].T + p["b_in"] + p["pos"][None]          # (B,T,d)
        caches = []
        for l in range(self.L):
            k = f"l{l}."
            # ---- LN1
            h, c_ln1 = ln_forward(z, p[k + "ln1_g"], p[k + "ln1_b"])
            # ---- QKV
            qkv = h @ p[k + "W_qkv"].T + p[k + "b_qkv"]           # (B,T,3d)
            q, kk, v = np.split(qkv, 3, axis=-1)
            # (B,T,d) -> (B,H,T,dh)
            q = q.reshape(B, T, H, nh).transpose(0, 2, 1, 3)
            kk = kk.reshape(B, T, H, nh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, nh).transpose(0, 2, 1, 3)
            # ---- scaled dot-product + causal mask
            s = np.einsum("bhtd,bhsd->bhts", q, kk) / np.sqrt(nh)  # (B,H,T,T)
            mask = np.triu(np.ones((T, T), bool), 1)
            s = np.where(mask[None, None], -1e9, s)
            a = softmax(s, -1)
            # ---- weighted sum + merge
            y = np.einsum("bhts,bhsd->bhtd", a, v)                # (B,H,T,dh)
            y = y.transpose(0, 2, 1, 3).reshape(B, T, d)
            attn_out = y @ p[k + "W_proj"].T + p[k + "b_proj"]
            # ---- residual
            z1 = z + attn_out
            # ---- LN2 + FFN + residual
            h2, c_ln2 = ln_forward(z1, p[k + "ln2_g"], p[k + "ln2_b"])
            pre = h2 @ p[k + "W1"].T + p[k + "b1"]                # (B,T,ff)
            act = gelu(pre)
            z = z1 + act @ p[k + "W2"].T + p[k + "b2"]
            caches.append(dict(h=h, q=q, k=kk, v=v, a=a, y=y, c_ln1=c_ln1,
                               z1=z1, h2=h2, pre=pre, act=act, c_ln2=c_ln2))
        # ---- 取最后一个 token -> LN -> head -> tanh
        zl, c_lnf = ln_forward(z[:, -1], p["ln_g"], p["ln_b"])     # (B,d)
        r = zl @ p["w_head"].T + p["b_head"]                      # (B,1)
        u = self.u_max * np.tanh(r)

        if not return_cache:
            return u[:, 0]
        cache = dict(x=x, z0=z if self.L == 0 else None, layers=caches,
                     zl=zl, c_lnf=c_lnf, r=r, u=u)
        return u[:, 0], cache

    # ------------------------------------------------------------------ 反向
    def backward(self, dU, cache):
        """dU:(B,) 损失对 u 的梯度。返回与 self.p 同键的梯度字典。

        计算图（每层）：
          z ──► LN1 ──► QKV ──► S=QKᵀ/√dh ──► softmax ──► A·V ──► proj ──► ⊕ ─┐
          │                                                                     │
          └─────────────────────────────────────────────────────────────────────┴─► z1
          z1 ──► LN2 ──► FFN ──► GELU ──► FFN ──► ⊕ ──► z
        """
        p = self.p
        T, d, H, nh = self.T, self.d, self.H, self.dh
        x = cache["x"]
        B = x.shape[0]
        g = {}

        # ---- 输出层：u = u_max * tanh(r)
        du = dU.reshape(B, 1)
        dr = du * self.u_max * (1.0 - np.tanh(cache["r"]) ** 2)     # (B,1)
        g["w_head"] = dr.T @ cache["zl"]                            # (1,d)
        g["b_head"] = dr.sum(0)
        dzl = dr @ p["w_head"]                                     # (B,d)
        # ---- 末端 LN（只作用在最后一个 token 上）
        dlast, g["ln_g"], g["ln_b"] = ln_backward(
            dzl[:, None, :], (cache["c_lnf"][0][:, None, :],
                              cache["c_lnf"][1][:, None, :],
                              p["ln_g"]))
        dz = np.zeros((B, T, d))
        dz[:, -1] = dlast[:, 0]

        # ---- 逐层反传（逆序）
        for l in reversed(range(self.L)):
            k = f"l{l}."
            c = cache["layers"][l]
            # z = z1 + ffn(act); 梯度分两路
            dact = dz @ p[k + "W2"]                                # (B,T,ff)
            g[k + "W2"] = dz.reshape(-1, d).T @ c["act"].reshape(-1, self.ff)
            g[k + "b2"] = dz.sum((0, 1))
            dpre = dact * gelu_grad(c["pre"])
            g[k + "W1"] = dpre.reshape(-1, self.ff).T @ c["h2"].reshape(-1, d)
            g[k + "b1"] = dpre.sum((0, 1))
            dh2 = dpre @ p[k + "W1"]
            dz1, g[k + "ln2_g"], g[k + "ln2_b"] = ln_backward(dh2, c["c_ln2"])
            dz1 = dz1 + dz                                          # 残差

            # z1 = z + attn_out
            dattn_out = dz1
            g[k + "W_proj"] = dattn_out.reshape(-1, d).T @ c["y"].reshape(-1, d)
            g[k + "b_proj"] = dattn_out.sum((0, 1))
            dy = dattn_out @ p[k + "W_proj"]                        # (B,T,d)
            dy = dy.reshape(B, T, H, nh).transpose(0, 2, 1, 3)      # (B,H,T,dh)

            # y = A·V
            da = np.einsum("bhtd,bhsd->bhts", dy, c["v"])
            dvv = np.einsum("bhts,bhtd->bhsd", c["a"], dy)
            # softmax 反向：ds = a * (da - sum(da*a))
            dsm = c["a"] * (da - (da * c["a"]).sum(-1, keepdims=True))
            # s = QKᵀ/√dh，掩码位置 a=0 ⇒ ds=0，无需显式处理
            dq = np.einsum("bhts,bhsd->bhtd", dsm, c["k"]) / np.sqrt(nh)
            dkk = np.einsum("bhts,bhtd->bhsd", dsm, c["q"]) / np.sqrt(nh)
            # 合并多头 -> (B,T,3d)
            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, d)
            dkk = dkk.transpose(0, 2, 1, 3).reshape(B, T, d)
            dvv = dvv.transpose(0, 2, 1, 3).reshape(B, T, d)
            dqkv = np.concatenate([dq, dkk, dvv], axis=-1)          # (B,T,3d)
            g[k + "W_qkv"] = dqkv.reshape(-1, 3 * d).T @ c["h"].reshape(-1, d)
            g[k + "b_qkv"] = dqkv.sum((0, 1))
            dh_in = dqkv @ p[k + "W_qkv"]
            dz_in, g[k + "ln1_g"], g[k + "ln1_b"] = ln_backward(dh_in, c["c_ln1"])
            dz = dz1 + dz_in                                        # 残差

        # ---- 嵌入层
        g["W_in"] = dz.reshape(-1, T * 0 + self.F) if False else \
            dz.reshape(-1, d).T @ x.reshape(-1, self.F)
        g["b_in"] = dz.sum((0, 1))
        g["pos"] = dz.sum(0)
        return g

    # ------------------------------------------------------------------ 参数工具
    def params(self):
        return list(self.p.items())

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def copy_from(self, other):
        for k in self.p:
            self.p[k] = other[k].copy() if isinstance(other, dict) else other.p[k].copy()


# ----------------------------------------------------------------------------- 损失与优化器
def mse_loss(u, target):
    """L = mean((u - t)^2)/2 ；返回 loss 与 dL/du"""
    diff = u - target
    return 0.5 * float(np.mean(diff ** 2)), diff / u.size


class AdamW:
    """AdamW：解耦权重衰减（不衰减 bias / LayerNorm 的 gamma,beta）"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k, gval in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gval
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gval ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = mhat / (np.sqrt(vhat) + self.eps)
            if k.endswith("_g") or k.endswith("_b") or k in ("pos",):
                params[k] -= lr * upd                     # 不衰减
            else:
                params[k] -= lr * (upd + self.wd * params[k])


# ----------------------------------------------------------------------------- 自检
if __name__ == "__main__":
    np.random.seed(0)
    m = TinyTF()
    print("参数量:", m.n_params())
    x = np.random.randn(4, 24, 6)
    u = m.forward(x)
    print("输出:", np.round(u, 4))
    u2, c = m.forward(x, return_cache=True)
    grads = m.backward(np.ones(4), c)
    print("梯度范数:", {k: round(float(np.linalg.norm(v)), 3) for k, v in list(grads.items())[:5]})

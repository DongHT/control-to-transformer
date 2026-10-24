"""附录 A 的数据：逐步前向数值、反向梯度、闭环注意力序列。"""
import json
import math
import os
import sys

import numpy as np
import torch

import plant as pl
import controllers as C
import train as T
from dataset import DT, features_row
from model import Policy
from plant import P, wrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
H, FEAT = 24, 6


def load(path):
    m = Policy(); m.load_state_dict(torch.load(path)); m.eval()
    return m


def window_features(x_hist, u_hist):
    return np.stack([features_row(x_hist[k], u_hist[k]) for k in range(len(x_hist))],
                    axis=0).astype(np.float32)


def forward_trace(model, win):
    """manual forward with dumps; win: (H,6)"""
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    x = win.astype(np.float64)
    W_in, b_in, pos = sd["inp.weight"], sd["inp.bias"], sd["pos"]
    z = x @ W_in.T + b_in + pos                      # (H,32)
    out = {"z_last": z[-1][:6].tolist(), "z_norm": float(np.linalg.norm(z[-1]))}

    def ln(v, g, b, eps=1e-5):
        m = v.mean(); var = ((v - m) ** 2).mean()
        return (v - m) / np.sqrt(var + eps) * g + b, float(m), float(var)

    g1, b1 = sd["blocks.0.ln1.weight"], sd["blocks.0.ln1.bias"]
    h1, mu1, var1 = ln(z[-1], g1, b1)
    out["ln1"] = {"mu": mu1, "var": var1,
                  "h_first6": h1[:6].tolist(), "h_norm": float(np.linalg.norm(h1))}

    Wqkv, bqkv = sd["blocks.0.qkv.weight"], sd["blocks.0.qkv.bias"]
    qkv_last = Wqkv @ h1 + bqkv
    out["qkv_last"] = {"first6": qkv_last[:6].tolist(),
                       "q_norm": float(np.linalg.norm(qkv_last[:32])),
                       "k_norm": float(np.linalg.norm(qkv_last[32:64])),
                       "v_norm": float(np.linalg.norm(qkv_last[64:]))}

    # full attention for the last token, head 0
    Hn, pnn = H, 4
    qkv = (z @ Wqkv.T + bqkv)                        # (H,96)
    q = qkv[:, :32].reshape(H, 4, 8)
    k = qkv[:, 32:64].reshape(H, 4, 8)
    v = qkv[:, 64:].reshape(H, 4, 8)
    sc = np.einsum("thd,shd->hts", q, k) / math.sqrt(8)
    mask = np.triu(np.ones((H, H), bool), 1)
    sc = np.where(mask[None], -1e9, sc)
    w = np.exp(sc - sc.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    out["attn"] = {"head0": w[0, -1, :].tolist(),
                   "logits_head0_first6": (sc[0, -1, :6]).tolist(),
                   "row_sum": float(w[0, -1].sum())}
    y = np.einsum("hts,shd->thd", w, v).reshape(H, 32)
    Wp, bp = sd["blocks.0.proj.weight"], sd["blocks.0.proj.bias"]
    z2 = z + (y @ Wp.T + bp)
    out["after_attn"] = {"resid_norm": float(np.linalg.norm(z[-1])),
                         "attn_out_norm": float(np.linalg.norm(y[-1])),
                         "z_norm": float(np.linalg.norm(z2[-1]))}

    g2, b2 = sd["blocks.0.ln2.weight"], sd["blocks.0.ln2.bias"]
    h2, mu2, var2 = ln(z2[-1], g2, b2)
    W1, b1f = sd["blocks.0.fc1.weight"], sd["blocks.0.fc1.bias"]
    W2, b2f = sd["blocks.0.fc2.weight"], sd["blocks.0.fc2.bias"]
    pre = W1 @ h2 + b1f
    act = 0.5 * pre * (1 + np.vectorize(math.erf)(pre / math.sqrt(2)))
    z3 = z2 + (W2 @ act + b2f)
    out["ffn"] = {"pre_first6": pre[:6].tolist(), "act_first6": act[:6].tolist(),
                  "neg_count": int((pre < 0).sum()), "hidden": int(len(pre)),
                  "z_norm": float(np.linalg.norm(z3[-1]))}

    # run the real model to the end for the final numbers
    with torch.no_grad():
        u = float(model(torch.from_numpy(win[None]))[0])
    out["out"] = {"u": u, "u_norm": u / 12.0}
    return out


def backward_trace(model, win, target=-6.0):
    """gradient magnitudes per parameter block + grad at the output."""
    m = Policy(); m.load_state_dict(model.state_dict()); m.train()
    x = torch.from_numpy(win[None])
    u = m(x)
    loss = 0.5 * (u - target) ** 2
    m.zero_grad(); loss.backward()
    g = {}
    for k, p in m.named_parameters():
        g[k] = float(p.grad.norm())
    return {"loss": float(loss), "u": float(u), "grads": g}


def rollout_with_attention(model, x0, T_=12.0):
    """closed loop, recording attention of every step (both layers)."""
    ctl = T.PolicyCtl(model)
    ctl.reset()
    x = np.array(x0, float)
    hist_x, hist_u = [], []
    rec = {"t": [], "th": [], "p": [], "u": [], "attn1": [], "attn2": [], "win": []}
    buf = None
    for k in range(int(T_ / DT)):
        f = features_row(x, ctl.u_prev)
        if buf is None:
            buf = np.tile(f, (H, 1)).astype(np.float32)
        else:
            buf = np.concatenate([buf[1:], f[None]], axis=0)
        u = ctl(x, k * DT)
        # attention from the same forward pass
        att = ctl.m.blocks  # not directly exposed; recompute from weights
        rec["t"].append(round(k * DT, 3))
        rec["th"].append(round(float(wrap(x[1])), 4))
        rec["p"].append(round(float(x[0]), 4))
        rec["u"].append(round(float(u), 3))
        rec["win"].append([round(float(v[1]), 3) for v in buf])   # theta history
        x = pl.rk4(x, u, DT)
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]; x[2] = 0.0
    return rec


def attention_series(model, rec, states):
    """recompute attention rows for the recorded states (cheap, batched)."""
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    def ln(v, g, b, eps=1e-5):
        m = v.mean(); s = np.sqrt(((v - m) ** 2).mean() + eps)
        return (v - m) / s * g + b
    attn1, attn2 = [], []
    for k in range(len(rec["t"])):
        th = np.array(rec["win"][k])
        # rebuild the full window from the recorded theta is lossy; recompute here
        pass
    return attn1, attn2


def main():
    model = load(os.path.join(OUT, "policy.pt"))
    x0 = [0.0, np.pi - 0.14, 0.0, 0.0]

    # ---- a real window from the swing-up phase
    ctl = T.PolicyCtl(model)
    ctl.reset()
    x = np.array(x0, float)
    buf = None
    best = None
    for k in range(int(8.0 / DT)):
        f = features_row(x, ctl.u_prev)
        buf = np.tile(f, (H, 1)).astype(np.float32) if buf is None else \
            np.concatenate([buf[1:], f[None]], axis=0)
        u = ctl(x, k * DT)
        if 1.2 <= k * DT <= 2.4 and best is None:
            best = buf.copy()
        x = pl.rk4(x, u, DT)
    win = best

    fwd = forward_trace(model, win)
    res = {"forward": fwd,
           "backward": backward_trace(model, win, target=fwd["out"]["u"] + 1.0)}
    print("u =", round(res["forward"]["out"]["u"], 3),
          "| loss =", round(res["backward"]["loss"], 4))

    # ---- closed-loop rollout with attention (record everything in one pass, numpy)
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    def ln(v, g, b, eps=1e-5):
        m = v.mean(); s = np.sqrt(((v - m) ** 2).mean() + eps)
        return (v - m) / s * g + b

    def forward_attn(win_np):
        z = win_np @ sd["inp.weight"].T + sd["inp.bias"] + sd["pos"]
        atts = []
        for L in range(2):
            pre = f"blocks.{L}."
            h = np.stack([ln(z[t], sd[pre + "ln1.weight"], sd[pre + "ln1.bias"]) for t in range(H)])
            qkv = h @ sd[pre + "qkv.weight"].T + sd[pre + "qkv.bias"]
            q = qkv[:, :32].reshape(H, 4, 8)
            kk = qkv[:, 32:64].reshape(H, 4, 8)
            v = qkv[:, 64:].reshape(H, 4, 8)
            sc = np.einsum("thd,shd->hts", q, kk) / math.sqrt(8)
            mask = np.triu(np.ones((H, H), bool), 1)
            sc = np.where(mask[None], -1e9, sc)
            w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
            atts.append(w.mean(0)[-1])            # average heads, last token
            y = np.einsum("hts,shd->thd", w, v).reshape(H, 32)
            z = z + (y @ sd[pre + "proj.weight"].T + sd[pre + "proj.bias"])
            h2 = np.stack([ln(z[t], sd[pre + "ln2.weight"], sd[pre + "ln2.bias"]) for t in range(H)])
            act = 0.5 * (h2 @ sd[pre + "fc1.weight"].T + sd[pre + "fc1.bias"])
            act = act * (1 + np.vectorize(math.erf)(act / math.sqrt(2) )) if False else act
            pre_ff = h2 @ sd[pre + "fc1.weight"].T + sd[pre + "fc1.bias"]
            act = 0.5 * pre_ff * (1 + np.vectorize(math.erf)(pre_ff / math.sqrt(2)))
            z = z + (act @ sd[pre + "fc2.weight"].T + sd[pre + "fc2.bias"])
        zl = ln(z[-1], sd["ln.weight"], sd["ln.bias"])
        u = math.tanh(float(zl @ sd["head.weight"][0] + sd["head.bias"][0])) * 12.0
        return u, atts[0], atts[1]

    ctl = T.PolicyCtl(model); ctl.reset()
    x = np.array(x0, float)
    rec = {"t": [], "th": [], "p": [], "u": [], "win": [], "attn": []}
    for k in range(int(12.0 / DT)):
        f = features_row(x, ctl.u_prev)
        buf = np.tile(f, (H, 1)).astype(np.float32) if buf is None or k == 0 else \
            np.concatenate([buf[1:], f[None]], axis=0)
        _u, a1, a2 = forward_attn(buf.astype(np.float64))
        ctl.u_prev = _u
        rec["t"].append(round(k * DT, 3))
        rec["th"].append(round(float(wrap(x[1])), 4))
        rec["p"].append(round(float(x[0])), )
        rec["u"].append(round(float(_u), 3))
        rec["win"].append([round(float(v[1]), 3) for v in buf])       # cos(theta) history
        rec["attn"].append([round(float(v), 4) for v in a2])          # layer-2, last token
        x = pl.rk4(x, np.clip(_u, -P["u_max"], P["u_max"]), DT)
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]; x[2] = 0.0
    res["rollout"] = rec
    json.dump(res, open(os.path.join(OUT, "chA.json"), "w"))
    print("rollout steps:", len(rec["t"]), "-> out/chA.json")
    print("forward keys:", list(res["forward"].keys()))
    print("top grads:", sorted(res["backward"]["grads"].items(), key=lambda kv: -kv[1])[:4])


if __name__ == "__main__":
    main()

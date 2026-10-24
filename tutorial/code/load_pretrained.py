"""
load_pretrained.py —— 把项目里训练好的 torch 权重装进纯 numpy 实现，验证：

1. 推理逐位一致（在真实窗口上比 max|Δu|）
2. 闭环行为一致（numpy 策略跑真实仿真，成功率）
3. 注意力可视化（从 numpy 的 forward cache 里直接取，不依赖任何框架）

产物：
  out/pretrained_report.json
  out/fig_parity.png        逐点一致性散点
  out/fig_attention.png     摆起 / 稳摆 的注意力热图
  out/attention_series.json 闭环中每步的注意力（供视频逐帧动画）
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(os.path.dirname(HERE), "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BUILD)
sys.path.insert(0, HERE)

from tinytf_numpy import TinyTF as NpTF                      # noqa: E402
from tinytf_torch import TinyTF as TorchTF, numpy_from_state  # noqa: E402
from plant import P, wrap                                   # noqa: E402
import plant as pl                                          # noqa: E402
from dataset import features_row, DT, random_x0              # noqa: E402


def load_np_model():
    sd = torch.load(os.path.join(BUILD, "out", "policy.pt"), map_location="cpu")
    np_model = NpTF(seed=0)
    np_model.p = {k: v.astype(np.float64) for k, v in numpy_from_state(sd).items()}
    tm = TorchTF()
    tm.load_state_dict(sd)
    tm.eval()
    return np_model, tm


def check_parity(np_model, tm, X, n=512):
    idx = np.random.default_rng(0).choice(len(X), n, replace=False)
    xb = X[idx].astype(np.float64)
    u_np = np_model.forward(xb)
    with torch.no_grad():
        u_t = tm(torch.tensor(xb, dtype=torch.float32)).numpy()
    d = np.abs(u_np - u_t)
    return dict(n=int(n), max_abs=float(d.max()), mean_abs=float(d.mean()),
                corr=float(np.corrcoef(u_np, u_t)[0, 1]))


def rollout(np_model, x0, T_=12.0, keep_attn=True):
    """numpy 策略的闭环；返回轨迹与每步注意力（第 2 层、最后 token、4 头平均）"""
    x = np.array(x0, float)
    buf, uprev = None, 0.0
    rec = dict(t=[], th=[], p=[], u=[], win=[], attn=[])
    for k in range(int(T_ / DT)):
        f = features_row(x, uprev)
        buf = np.tile(f, (24, 1)) if buf is None else np.concatenate([buf[1:], f[None]])
        _, cache = np_model.forward(buf[None].astype(np.float64), return_cache=True)
        u = float(cache["u"][0, 0])
        a = cache["layers"][-1]["a"][0].mean(0)[-1]          # (T,) 平均 4 头、取最后 token
        uprev = u
        rec["t"].append(round(k * DT, 3))
        rec["th"].append(round(float(wrap(x[1])), 4))
        rec["p"].append(round(float(x[0]), 4))
        rec["u"].append(round(u, 3))
        rec["win"].append([round(float(v[1]), 3) for v in buf])   # cos θ
        rec["attn"].append([round(float(v), 5) for v in a])
        x = pl.rk4(x, float(np.clip(u, -P["u_max"], P["u_max"])), DT)
        if abs(x[0]) > P["x_max"]:
            x[0] = np.sign(x[0]) * P["x_max"]
            x[2] = 0.0
    th = np.abs(np.array(rec["th"]))
    ok = bool(th[int(9 / DT):].max() < 0.15 and abs(rec["p"][-1]) < 1.5)
    return ok, rec


def fig_parity(u_np, u_t, path):
    from PIL import Image, ImageDraw, ImageFont
    W = H = 460
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/System/Library/Fonts/Songti.ttc", 15, index=6)
    L, R, T, B = 60, 20, 20, 50
    lo, hi = -12, 12
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - T - B)
    d.rectangle([L, T, W - R, H - B], outline=(150, 160, 175), width=1)
    d.line([sx(lo), sy(lo), sx(hi), sy(hi)], fill=(120, 130, 145), width=1)
    for i in range(len(u_np)):
        x, y = sx(u_np[i]), sy(u_t[i])
        d.ellipse([x - 2.4, y - 2.4, x + 2.4, y + 2.4], fill=(30, 160, 150))
    for v in (-12, -6, 0, 6, 12):
        d.text((sx(v), H - B + 8), f"{v:g}", font=f, fill=(90, 104, 120), anchor="ma")
        d.text((L - 8, sy(v)), f"{v:g}", font=f, fill=(90, 104, 120), anchor="rm")
    d.text((W / 2, H - 22), "torch 输出 u [N]", font=f, fill=(40, 48, 58), anchor="mm")
    d.text((16, H / 2), "numpy 输出 u [N]", font=f, fill=(40, 48, 58), anchor="mm")
    img.save(path)


def fig_attention(rec, path):
    """左：摆起窗口的注意力；右：稳摆窗口的注意力"""
    from PIL import Image, ImageDraw, ImageFont
    A = np.array(rec["attn"])
    th = np.abs(np.array(rec["th"]))
    i_sw = int(np.argmax((th > 0.6))) if (th > 0.6).any() else 20
    steady_idx = np.where(th < 0.08)[0]
    i_st = int(steady_idx[len(steady_idx) // 2]) if len(steady_idx) else len(th) - 1

    W, H = 900, 420
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/System/Library/Fonts/Songti.ttc", 16, index=6)
    fb = ImageFont.truetype("/System/Library/Fonts/Songti.ttc", 19, index=1)
    for panel, (k, title) in enumerate([(i_sw, "摆起阶段（|θ| 大）"), (i_st, "稳摆阶段（|θ|≈0）")]):
        x0, y0 = 60 + panel * 440, 60
        w, h = 320, 24 * 11
        d.text((x0, y0 - 28), title, font=fb, fill=(30, 36, 44))
        row = A[k] / max(A[k].max(), 1e-9)
        for i in range(24):
            for j in range(24):
                # 横轴=key 位置, 纵轴=query 位置（只画最后一行也行，这里画完整下三角）
                pass
        for j in range(24):
            c = int(240 - row[j] * 190)
            d.rectangle([x0 + j * 13, y0, x0 + j * 13 + 11, y0 + h],
                        fill=(c, 248 - int(row[j] * 40), 245 - int(row[j] * 30)))
        d.text((x0, y0 + h + 8), f"过去 0.48 s → 现在（峰值 @{int(np.argmax(A[k]))}）",
               font=f, fill=(70, 84, 100))
    d.text((W / 2, H - 26), "第 2 层注意力（4 头平均，最后 token 对 24 个历史位置）",
           font=f, fill=(50, 60, 72), anchor="mm")
    img.save(path)


def main():
    np_model, tm = load_np_model()
    X, Y = None, None
    cache = os.path.join(OUT, "data_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        X = d["X"]
    else:
        import dataset as D
        X, Y = D.gen(n_ep=60, T=8.0, noise=1.5, seed=9)

    rep = {"parity": check_parity(np_model, tm, X)}
    print("① 推理一致性:", rep["parity"])

    rng = np.random.default_rng(7)
    ics = [random_x0(rng, "hard") for _ in range(30)]
    oks, recs = [], []
    for x0 in ics:
        ok, rec = rollout(np_model, x0)
        oks.append(ok)
        recs.append(rec)
    rep["success"] = sum(oks) / len(oks)
    print(f"② 闭环成功率（numpy 策略，30 组）: {sum(oks)}/{len(oks)}")

    best = min(recs, key=lambda r: r["t"][-1] if False else 0)
    # 用"从下垂出发"的那条做可视化
    ok, rec_hang = rollout(np_model, [0.0, np.pi - 0.14, 0.0, 0.0])
    rep["hang_to_up_s"] = next((t for t, th in zip(rec_hang["t"], rec_hang["th"])
                                if abs(th) < 0.1), None)
    print(f"③ 从下垂出发: 摆起用时 {rep['hang_to_up_s']} s, 成功={ok}")

    json.dump(rep, open(os.path.join(OUT, "pretrained_report.json"), "w"), indent=1)
    json.dump(rec_hang, open(os.path.join(OUT, "attention_series.json"), "w"))

    idx = np.random.default_rng(0).choice(len(X), 400, replace=False)
    u_np = np_model.forward(X[idx].astype(np.float64))
    with torch.no_grad():
        u_t = tm(torch.tensor(X[idx], dtype=torch.float32)).numpy()
    fig_parity(u_np, u_t, os.path.join(OUT, "fig_parity.png"))
    fig_attention(rec_hang, os.path.join(OUT, "fig_attention.png"))
    print("④ 图已保存: fig_parity.png / fig_attention.png")


if __name__ == "__main__":
    main()

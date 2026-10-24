"""附录 A：Transformer 内部机制全解 —— 8 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import math_, math_w, title, chip, axes, cartpole, curvy
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "chA.json")))
FW, BW, RO = D["forward"], D["backward"], D["rollout"]
ATT = np.array(RO["attn"]); WIN = np.array(RO["win"]); TH = np.array(RO["th"])
PP = np.array(RO["p"]); UU = np.array(RO["u"]); TT = np.array(RO["t"])


def box(cv, x0, y0, x1, y1, label, shape="", col=CY, a=1.0, size=17, shape_size=13):
    cv.rect(x0, y0, x1, y1, col, width=1.2, alpha=0.85 * a, fill_alpha=0.07 * a, radius=7)
    cv.text((x0 + x1) / 2, (y0 + y1) / 2 - 2, label, size=size, color=col, anchor="ma", alpha=a)
    if shape:
        cv.text(x0 + 8, y1 + 16, shape, size=shape_size, color=FAINT, alpha=a)


def s1(cv, t, dur):
    cv.text(640, 250, "倒立摆控制　·　课程附录", size=25, color=CY, anchor="ma", alpha=appear(t, 0.2, 0.9))
    cv.text(640, 344, "附录 A　Transformer 内部机制", size=56, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 404, "前向逐步、反向逐层、注意力与运动对齐", size=30, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((490, 442), (790, 442), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 506, r"\underbrace{24\times6}_{x}\to\underbrace{24\times32}_{z}"
                            r"\to\underbrace{24\times96}_{QKV}\to\cdots\to\underbrace{1}_{u=4.153}",
              20, DIM, anchor="mc")
    cv.text(640, 596, "全部数值取自真实权重与真实闭环轨迹", size=19, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "全链路：21 个算子，18,177 个参数")
    cols = [
        [("输入窗口", "24×6", "", DIM), ("线性嵌入 6→32", "24×32", "224", CY),
         ("+ 位置编码", "24×32", "768", CY), ("LN1", "24×32", "64", CY),
         ("QKV 投影 32→96", "24×96", "3,168", CY), ("分头 4×8", "4×24×8", "", BL)],
        [("因果掩码 + softmax", "4×24×24", "", MG), ("加权求和 AV", "4×24×8", "", MG),
         ("合并 + 输出投影", "24×32", "1,056", CY), ("残差 ⊕", "24×32", "", AM),
         ("LN2", "24×32", "64", CY), ("FFN 32→64", "24×64", "2,112", CY)],
        [("GELU", "24×64", "", GR), ("FFN 64→32", "24×32", "2,080", CY),
         ("残差 ⊕", "24×32", "", AM), ("（以上重复 2 次）", "", "17,088", DIM),
         ("LN + 输出头", "32→1", "97", CY), ("u = 12·tanh(·)", "1", "", MG)],
    ]
    xs = [(96, 396), (446, 746), (796, 1146)]
    y0, bh, gap = 190, 62, 12
    for ci, col in enumerate(cols):
        for ri, (lab, shape, par, c) in enumerate(col):
            aa = appear(t, 0.5 + ci * 0.5 + ri * 0.22, 0.4)
            if aa <= 0:
                continue
            x0, x1 = xs[ci]
            y = y0 + ri * (bh + gap)
            cv.rect(x0, y, x1, y + bh, c, width=1.2, alpha=0.85 * aa, fill_alpha=0.07 * aa, radius=7)
            cv.text((x0 + x1) / 2, y + 26, lab, size=16, color=c, anchor="ma", alpha=aa)
            sub = shape + ("  ·  " + par if par else "")
            if sub:
                cv.text((x0 + x1) / 2, y + 48, sub, size=13, color=FAINT, anchor="ma", alpha=aa)
            if ri < len(col) - 1:
                cv.line(((x0 + x1) / 2, y + bh), ((x0 + x1) / 2, y + bh + gap), c, width=1.2,
                        alpha=0.5 * aa)
    a = appear(t, 3.6, 0.8)
    for ci in range(2):
        cv.arrow((xs[ci][1] + 8, 400), (xs[ci + 1][0] - 8, 400), DIM, width=1.6, alpha=0.8 * a)
    a2 = appear(t, 4.6, 0.8)
    cv.text(96, 640, "每框第二行：张量形状 · 参数量。第 4–16 步整体重复一次，构成两层。",
            size=18, color=FAINT, alpha=a2)


def s3(cv, t, dur):
    title(cv, "前向 ①：嵌入与 LayerNorm")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 176, "当前 token（第 24 步）", size=18, color=FAINT, alpha=a)
    feats = ["sinθ", "cosθ", "θ̇/4", "v/4", "p/2.4", "u/12"]
    for i, f in enumerate(feats):
        x = 110 + i * 96
        box(cv, x, 196, x + 84, 240, f, "", CY, a, size=15)
    a2 = appear(t, 1.6, 0.8)
    cv.text(110, 268, "→ 线性嵌入 6→32，再逐元素加位置编码", size=19, color=INK, alpha=a2)
    zs = FW["z_last"]
    for i, v in enumerate(zs):
        x = 110 + i * 96
        cv.text(x + 42, 320, f"{v:+.3f}", size=17, color=CY, anchor="ma", alpha=a2)
    cv.text(110, 348, f"z 的前 6 维（共 32 维），‖z‖ = {FW['z_norm']:.2f}", size=17, color=FAINT, alpha=a2)

    a3 = appear(t, 3.2, 0.8)
    cv.rect(110, 380, 620, 470, CY, width=1.2, alpha=0.75 * a3, fill_alpha=0.05 * a3, radius=8)
    cv.text(132, 408, "LayerNorm", size=20, color=CY, alpha=a3)
    math_(cv, 132, 450, r"h=\frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}}\odot\gamma+\beta", 20, INK)

    a4 = appear(t, 4.6, 0.8)
    mu, var = FW["ln1"]["mu"], FW["ln1"]["var"]
    cv.text(660, 400, f"μ = {mu:+.3f}", size=20, color=INK, alpha=a4)
    cv.text(660, 436, f"σ² = {var:.3f}", size=20, color=INK, alpha=a4)
    cv.text(660, 472, f"1/σ = {1/math.sqrt(var+1e-5):.2f} 倍放大", size=20, color=AM, alpha=a4)
    a5 = appear(t, 5.8, 0.8)
    cv.text(110, 520, "σ² 只有 0.223：输入都被归一化过，动态范围很小", size=19, color=DIM, alpha=a5)
    cv.text(110, 556, "→ LayerNorm 放大 2.1 倍，否则注意力分数会挤在 0 附近", size=19, color=DIM, alpha=a5)


def s4(cv, t, dur):
    title(cv, "前向 ②：QKV、掩码、softmax")
    a = appear(t, 0.3, 0.8)
    kv = FW["qkv_last"]
    math_(cv, 110, 200, r"[Q;K;V]=hW_{\mathrm{qkv}}^{\top}+b_{\mathrm{qkv}}\in\mathbb{R}^{96}",
          20, INK)
    a2 = appear(t, 1.6, 0.8)
    cv.text(110, 250, f"‖q‖ = {kv['q_norm']:.2f}    ‖k‖ = {kv['k_norm']:.2f}    "
                      f"‖v‖ = {kv['v_norm']:.2f}", size=19, color=CY, alpha=a2)
    a3 = appear(t, 2.8, 0.8)
    math_(cv, 110, 306, r"s_{ts}=\frac{q_t^{\top}k_s}{\sqrt 8}\ (s\le t),\quad "
                        r"a_{ts}=\mathrm{softmax}_s(s_{ts})", 19, INK)
    a4 = appear(t, 4.0, 0.9)
    row = np.array(FW["attn"]["head0"])
    if a4 > 0:
        # attention row bars
        X0, Y0, X1, Y1 = 110, 380, 700, 560
        cv.text(X0, 366, "第 0 头、最后一个 token 的注意力（24 个位置）", size=17, color=FAINT, alpha=a4)
        for i in range(24):
            x = X0 + i * 24.5
            hh = row[i] / max(row.max(), 1e-6) * 150
            cv.rect(x, Y1 - hh, x + 18, Y1, CY, width=1, alpha=0.8 * a4, fill_alpha=0.5 * a4, radius=2)
        cv.line((X0, Y0 + 30), (X1, Y0 + 30), MG, width=1.4, alpha=0.9 * a4, dash=5)
        cv.text(X1 - 4, Y0 + 20, "1/24 = 0.0417（均匀）", size=15, color=MG, anchor="ra", alpha=a4)
        cv.text(X0, Y1 + 22, "过去 ←", size=15, color=FAINT, alpha=a4)
        cv.text(X1, Y1 + 22, "现在", size=15, color=FAINT, anchor="ra", alpha=a4)
    a5 = appear(t, 6.0, 0.8)
    cv.text(760, 400, f"前 6 个位置：{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}…", size=17,
            color=INK, alpha=a5)
    cv.text(760, 440, "与均匀值只差 5%", size=19, color=AM, alpha=a5)
    cv.text(760, 480, "→ 这一头几乎在“平均历史”", size=19, color=DIM, alpha=a5)


def s5(cv, t, dur):
    title(cv, "前向 ③：残差、FFN、输出")
    a = appear(t, 0.3, 0.8)
    aa = FW["after_attn"]
    cv.text(110, 190, "残差前后的量级", size=19, color=FAINT, alpha=a)
    rows = [(f"‖z‖ = {aa['resid_norm']:.2f}", "残差主干", INK),
            (f"‖attn_out‖ = {aa['attn_out_norm']:.2f}", "注意力分支", CY),
            (f"‖z + attn_out‖ = {aa['z_norm']:.2f}", "合并后", AM)]
    for i, (k, v, col) in enumerate(rows):
        cv.text(130, 230 + i * 36, f"{k:18s}  {v}", size=18, color=col, alpha=appear(t, 0.5 + 0.4 * i, 0.5))
    a2 = appear(t, 2.2, 0.8)
    cv.text(110, 360, "注意力只贡献约 1/3 的幅度 → 训练初期不容易崩", size=18, color=DIM, alpha=a2)

    a3 = appear(t, 3.2, 0.8)
    math_(cv, 600, 200, r"\mathrm{FFN}(h)=W_2\,\mathrm{GELU}(W_1h+b_1)+b_2", 19, INK)
    a4 = appear(t, 4.2, 0.9)
    if a4 > 0:
        X0, Y0, X1, Y1 = 600, 260, 1140, 420
        cv.text(X0, Y0 - 12, "GELU：64 个隐单元里有 34 个输入为负", size=17, color=FAINT, alpha=a4)
        sx = lambda v: X0 + (v + 3) / 6 * (X1 - X0)
        sy = lambda v: (Y0 + Y1) / 2 - v / 3 * (Y1 - Y0) / 2
        cv.line((sx(0), Y0), (sx(0), Y1), (90, 108, 130), width=1, alpha=0.6 * a4, dash=5)
        curvy(cv, [(sx(v), sy(0.5 * v * (1 + math.erf(v / math.sqrt(2)))))
                   for v in np.linspace(-3, 3, 200)], GR, 2.2, a4)
        cv.text(sx(-2.4), sy(-0.35), "负半轴被软压到 ~0", size=15, color=FAINT, alpha=a4)

    a5 = appear(t, 6.2, 0.8)
    math_(cv, 600, 470, r"u=12\tanh(w^{\top}\mathrm{LN}(z_{24})+b)=4.153\ \mathrm{N}", 20, CY)
    cv.text(600, 520, "tanh 的导数 1−tanh²：靠近 ±12 时会饱和（梯度消失）", size=17, color=DIM, alpha=a5)


def s6(cv, t, dur):
    title(cv, "反向：梯度怎么流回每一层")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 196, r"\mathcal{L}=\tfrac12(u-u^\star)^2\ \Rightarrow\ "
                        r"\frac{\partial\mathcal{L}}{\partial u}=1.0,\ \mathcal{L}=0.5", 19, INK)
    a2 = appear(t, 1.8, 0.8)
    math_(cv, 110, 248, r"\frac{\partial\mathcal{L}}{\partial s_j}=a_j\Big(g_j-\sum_i a_i g_i\Big)",
          19, CY)
    cv.text(110, 292, "softmax 的雅可比自带“减基线”；注意力越均匀，梯度越小", size=17, color=DIM, alpha=a2)
    a3 = appear(t, 3.2, 0.8)
    math_(cv, 110, 344, r"\frac{\partial\mathcal{L}}{\partial z}=\frac{\gamma}{\sigma}\Big(g-\bar g"
                        r"-\frac{\hat z}{d}\sum_j g_j\hat z_j\Big)", 19, CY)
    cv.text(110, 388, "LN 的梯度正交于“整体平移”与“整体缩放”", size=17, color=DIM, alpha=a3)

    a4 = appear(t, 4.6, 0.9)
    if a4 > 0:
        g = BW["grads"]
        items = [("head.weight", g["head.weight"], MG),
                 ("blocks.0.proj.weight", g["blocks.0.proj.weight"], CY),
                 ("blocks.0.fc2.weight", g["blocks.0.fc2.weight"], CY),
                 ("blocks.0.qkv.weight", g["blocks.0.qkv.weight"], CY),
                 ("blocks.1.fc1.weight", g["blocks.1.fc1.weight"], CY),
                 ("blocks.1.qkv.weight", g["blocks.1.qkv.weight"], CY)]
        cv.text(640, 190, "梯度范数（实测，单样本）", size=17, color=FAINT, alpha=a4)
        mx = max(v for _, v, _ in items)
        for i, (k, v, col) in enumerate(items):
            aa = appear(t, 4.6 + 0.35 * i, 0.5)
            if aa <= 0:
                continue
            y = 228 + i * 46
            cv.text(640, y, k.replace("blocks.", "L").replace(".weight", ""), size=16,
                    color=col, alpha=aa)
            cv.rect(820, y - 14, 820 + v / mx * 300, y + 4, col, width=1, alpha=0.85 * aa,
                    fill_alpha=0.45 * aa, radius=3)
            cv.text(1130, y, f"{v:.1f}", size=15, color=INK, anchor="ra", alpha=aa)
    a5 = appear(t, 7.0, 0.8)
    cv.text(640, 520, "输出头最大（52.2），第 0 层只衰减 4 倍", size=18, color=AM, alpha=a5)
    cv.text(640, 556, "残差是“梯度高速公路”，Pre-LN 让各层输入尺度一致", size=18, color=DIM, alpha=a5)


DT_STEP = 0.02


def s7(cv, t, dur):
    title(cv, "注意力 ↔ 倒立摆：两条时间轴对齐")
    # piecewise slow motion: swing-up (0-3.4 s) at 5x slow, then jump to steady (8.5-11 s)
    if t < 18.0:
        sim_t = max(0.0, (t - 1.0)) * 0.22
        phase = "摆起（慢放 5×）"
    else:
        sim_t = 8.5 + (t - 18.0) * 0.16
        phase = "稳摆（慢放 6×）"
    k = int(min(max(0.0, sim_t / DT_STEP), len(TT) - 1))
    a = appear(t, 0.3, 0.6)
    # cart-pole
    cartpole(cv, 330, 470, TH[k], p=PP[k], u=UU[k], a=a, scale=118, rail=2.4)
    cv.text(330, 520, f"t = {TT[k]:.2f} s    θ = {math.degrees(TH[k]):+.1f}°    "
                      f"p = {PP[k]:+.2f} m    u = {UU[k]:+.2f} N",
            size=18, color=DIM, anchor="ma", alpha=a)

    # attention bars
    row = ATT[k]
    X0, Y0, BW_, BH_ = 690, 210, 460, 150
    cv.text(X0, Y0 - 12, "第 2 层注意力（4 头平均，24 个位置）", size=16, color=FAINT, alpha=a)
    mx = max(row.max(), 1e-6)
    for i in range(24):
        x = X0 + i * 19
        hh = row[i] / mx * BH_
        cv.rect(x, Y0 + BH_ - hh, x + 14, Y0 + BH_, CY if row[i] > 0.0417 else (60, 110, 120),
                width=1, alpha=0.85 * a, fill_alpha=0.55 * a, radius=2)
    cv.line((X0, Y0 + BH_ + 4), (X0 + 24 * 19, Y0 + BH_ + 4), (92, 110, 132), width=1.2, alpha=0.8 * a)
    cv.text(X0, Y0 + BH_ + 28, "过去", size=15, color=FAINT, anchor="la", alpha=a)
    cv.text(X0 + 24 * 19, Y0 + BH_ + 28, "现在", size=15, color=FAINT, anchor="ra", alpha=a)

    # cos(theta) history (input window)
    Y2 = 470
    cv.text(X0, Y2 - 24, "输入窗口里的 cosθ 历史（0.48 s）", size=16, color=FAINT, alpha=a)
    ws = WIN[k]
    sx = lambda i: X0 + i * 19
    sy = lambda v: Y2 + 70 - (v + 1) / 2 * 70
    curvy(cv, [(sx(i), sy(ws[i])) for i in range(len(ws))], AM, 2.2, a)
    cv.line((X0, sy(0)), (X0 + 23 * 19, sy(0)), (90, 108, 130), width=1, alpha=0.5 * a, dash=5)
    cv.circle(sx(23), sy(ws[-1]), 5, AM, width=0, alpha=a, fill_alpha=a)
    cv.text(X0, Y2 + 92, "摆起时这是一段陡坡；稳摆时几乎是一条平线", size=16, color=FAINT, alpha=a)

    # peak marker
    pk = int(np.argmax(row))
    cv.line((X0 + pk * 19 + 7, Y0 + BH_ + 2), (X0 + pk * 19 + 7, Y0 + BH_ - 12), MG, width=2, alpha=a)
    cv.text(X0 + pk * 19 + 7, Y0 + BH_ - 18, f"峰值 @{pk}", size=15, color=MG, anchor="ma", alpha=a)
    cv.text(690, 604, phase + "    峰值位置 = " + str(pk), size=18, color=CY, alpha=a)
    a2 = appear(t, 6.0, 0.8)
    stat = ("统计：摆起峰值 16.6（中段）、稳摆 21.1（最近第 2 步）"
            if abs(TH[k]) < 0.5 else "摆起中：权重铺开，峰值位置逐帧变化（估相位用）")
    cv.text(690, 636, stat, size=16, color=AM, alpha=a2)


def s8(cv, t, dur):
    title(cv, "附录小结")
    items = [("全链路", "21 个算子、18,177 参数，形状可手算核对"),
             ("前向实测", "‖z‖=2.67，LN 放大 2.1×，注意力≈1/24"),
             ("反向实测", "输出头 52.2，第 0 层 13–20，仅衰减 4×"),
             ("为什么稳", "残差给梯度修高速，Pre-LN 保持尺度"),
             ("注意力", "摆起/稳摆几乎相同：整段历史加权"),
             ("一句话", "它不是“挑几步”，而是“估一个相位”")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.6 * i, 0.55)
        if aa <= 0:
            continue
        y = 180 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.8, 0.8)
    cv.text(130, 590, "回到第 8 讲：现在每一层都可以对着数字核对", size=23, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "这一讲是附录，专门拆开第 8 讲那个网络。",
        "从输入窗口到输出力，每一步的张量形状、前向数值、反向梯度，都会给出来。",
        "所有数字都来自真实权重和真实闭环轨迹，不是示意。",
    ], draw=s1, min_dur=13.0),
    dict(name="diagram", say=[
        "先看全链路。",
        "21 个算子：嵌入、位置编码、归一化、QKV、分头、掩码、softmax、加权求和、投影、残差、前馈、输出头。",
        "每个框下面写张量形状，框里写参数量。第四到第 16 步整体重复一次，就是两层。",
    ], draw=s2, min_dur=20.0),
    dict(name="fwd1", say=[
        "前向第一步：嵌入与归一化。",
        {"say": "当前 token 是 6 维特征，线性投影到 $32$ 维，再加位置编码，范数是 $2.67$。",
         "tts": "当前 token 是六维特征，线性投影到三十二维，再加位置编码，范数是二点六七。"},
        {"say": "LayerNorm 的方差只有 $0.223$，所以它把信号放大了 $2.1$ 倍。",
         "tts": "LayerNorm 的方差只有零点二二三，所以它把信号放大了二点一倍。"},
    ], draw=s3, min_dur=19.0),
    dict(name="fwd2", say=[
        "第二步，QKV 与注意力。",
        {"say": "投影到 96 维，切成 4 个头，每头 8 维，然后算点积、除以根号八、加因果掩码、做 softmax。",
         "tts": "投影到九十六维，切成四个头，每头八维，然后算点积、除以根号八、加因果掩码、做 softmax。"},
        {"say": "这是真实的注意力行：前 6 个位置都在 $0.04$ 附近，而均匀值是 0.0417。",
         "tts": "这是真实的注意力行：前六个位置都在零点零四附近，而均匀值是零点零四一七。"},
        "也就是说，它几乎在平均整段历史。",
    ], draw=s4, min_dur=21.0),
    dict(name="fwd3", say=[
        "第 3 步，残差、前馈、输出。",
        {"say": "残差主干范数 $2.67$，注意力分支只贡献 $0.88$，合并后 $2.79$。",
         "tts": "残差主干范数二点六七，注意力分支只贡献零点八八，合并后二点七九。"},
        {"say": "前馈有 64 个隐单元，其中 34 个输入为负，被 GELU 软压到零附近。",
         "tts": "前馈有六十四个隐单元，其中三十四个输入为负，被 GELU 软压到零附近。"},
        {"say": "最后过 tanh 乘 $12$，输出 $4.153$ 牛顿。",
         "tts": "最后过 tanh 乘十二，输出四点一五三牛顿。"},
    ], draw=s5, min_dur=21.0),
    dict(name="backward", say=[
        "现在反过来走一遍。",
        {"say": "损失对输出的梯度，经 tanh 的导数 $1-\\tanh^2$ 进入输出头。",
         "tts": "损失对输出的梯度，经 tanh 的导数一减 tanh 平方，进入输出头。"},
        {"say": "softmax 的雅可比自带减基线；LayerNorm 的梯度正交于平移与缩放——这两点决定了训练的稳定性。",
         "tts": "softmax 的雅可比自带减基线；LayerNorm 的梯度正交于平移与缩放——这两点决定了训练的稳定性。"},
        {"say": "实测梯度：输出头 $52.2$，第零层只有 $13$ 到 $20$，只衰减了 4 倍。",
         "tts": "实测梯度：输出头五十二点二，第零层只有十三到二十，只衰减了四倍。"},
    ], draw=s6, min_dur=23.0),
    dict(name="sync", say=[
        "最后把两条时间轴对起来：左边是倒立摆，右边是同一时刻的注意力。",
        {"say": "下面是输入窗口里的 $\\cos\\theta$ 历史，它其实就是模型“看到”的过去。",
         "tts": "下面是输入窗口里的 cos theta 历史，它其实就是模型看到的过去。"},
        {"say": "摆起阶段，这段历史是一道陡坡，注意力铺满整段——它在估计相位。",
         "tts": "摆起阶段，这段历史是一道陡坡，注意力铺满整段——它在估计相位。"},
        {"say": "稳住之后，历史几乎是一条平线，注意力怎么分都影响不大。",
         "tts": "稳住之后，历史几乎是一条平线，注意力怎么分都影响不大。"},
        "所以它的注意力看起来“没有重点”，但控制是有效的。",
    ], draw=s7, min_dur=32.0),
    dict(name="summary", say=[
        "小结：全链路可以逐层核对，前向与反向都有实测数值。",
        "注意力的形态说明它不是“挑几步”，而是在“估一个相位”。",
    ], draw=s8, min_dur=14.0),
]

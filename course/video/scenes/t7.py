"""Transformer 教程 · 第七集：训练方法（SGD → Momentum → RMSProp → Adam → AdamW）—— 9 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import math_, math_w, title, chip, axes, curvy, code
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
TUT = os.path.join(ROOT, "tutorial", "out")
OP = json.load(open(os.path.join(TUT, "optim.json")))
TOY, REAL = OP["toy"], OP["real"]
STEPS, BIAS, SCHED = OP["steps"], OP["bias"], OP["sched"]
KAPPA = OP["kappa"]

COL = {"sgd": (120, 130, 145), "momentum": BL, "rmsprop": AM, "adam": CY}


def contour(cv, x0, y0, w, h, a=1.0, label=True):
    """画 0.5(x²+κy²) 的等高椭圆。坐标 [-1.2,1.2]×[-2,2] 映射到 (x0,y0,w,h)。"""
    cv.rect(x0, y0, x0 + w, y0 + h, (60, 76, 94), width=1, alpha=0.7 * a, radius=4)
    sx = lambda x: x0 + (x + 1.2) / 2.4 * w
    sy = lambda y: y0 + h - (y + 2.0) / 4.0 * h
    for lv in (0.02, 0.08, 0.25, 0.7, 1.8, 4.5, 10.0):
        for sgn in (1,):
            xs, ys = [], []
            for k in range(73):
                th = 2 * math.pi * k / 72
                xx = math.sqrt(2 * lv) * math.cos(th)
                yy = math.sqrt(2 * lv / KAPPA) * math.sin(th)
                xs.append(sx(xx)); ys.append(sy(yy))
            curvy(cv, list(zip(xs, ys)), (58, 74, 92), 1.0, 0.55 * a)
    if label:
        cv.text(x0 + 8, y0 + 18, "f(x,y)=½(x²+κy²)，κ=50", size=13, color=FAINT, alpha=a)
        cv.text(x0 + w - 8, y0 + h - 12, "x", size=13, color=FAINT, anchor="ra", alpha=a)
    return sx, sy


def path(cv, sx, sy, pts, color, w=2.2, a=1.0, dot=True):
    xy = [(sx(p[0]), sy(p[1])) for p in pts]
    curvy(cv, xy, color, w, a)
    if dot:
        cv.circle(xy[0][0], xy[0][1], 4, color, width=0, alpha=a, fill_alpha=a)
        cv.circle(xy[-1][0], xy[-1][1], 4, color, width=1.4, alpha=a)


def s1(cv, t, dur):
    cv.text(640, 240, "Transformer 从零实现　·　第七集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 334, "训练方法", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 394, "从 SGD 到 AdamW：为什么要两个矩", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 434), (810, 434), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 500, r"\theta_t=\theta_{t-1}-\eta\ \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}",
              26, DIM, anchor="mc")
    cv.text(640, 600, "全部结论来自真实迭代：二维病态曲面 + 本项目的模型", size=18, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "为什么普通梯度下降不够")
    a = appear(t, 0.3, 0.8)
    sx, sy = contour(cv, 110, 200, 520, 380, a)
    a2 = appear(t, 1.4, 0.9)
    if a2 > 0:
        path(cv, sx, sy, TOY["sgd"]["path"], COL["sgd"], 2.0, 0.95 * a2)
        cv.text(120, 596, "SGD：在陡方向来回震荡，在平方向爬得极慢", size=17, color=COL["sgd"], alpha=a2)
    a3 = appear(t, 3.2, 0.8)
    cv.text(700, 210, "病态（ill-conditioned）", size=19, color=MG, alpha=a3)
    cv.text(700, 248, "Hessian 的最大/最小特征值比 = κ = 50", size=16, color=DIM, alpha=a3)
    cv.text(700, 284, "沿 y 方向曲率是 x 方向的 50 倍", size=16, color=DIM, alpha=a3)
    a4 = appear(t, 4.6, 0.8)
    cv.text(700, 344, "于是学习率被迫由最陡方向决定：", size=17, color=INK, alpha=a4)
    cv.text(700, 378, "η 稍大就震荡，稍小就爬不动", size=17, color=INK, alpha=a4)
    a5 = appear(t, 5.8, 0.8)
    chip(cv, 700, 440, "神经网络里这种方向差异是常态", MG, a5, 18)
    a6 = appear(t, 6.8, 0.8)
    cv.text(700, 500, "本模型实测：各层梯度范数从 12 到 70", size=16, color=DIM, alpha=a6)
    cv.text(700, 530, "相差近 6 倍；而同一层内部逐元素还能差更多", size=16, color=DIM, alpha=a6)


def s3(cv, t, dur):
    title(cv, "① 动量：给梯度加惯性")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 190, r"v_t=\beta v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta v_t", 21, INK)
    a2 = appear(t, 1.6, 0.8)
    cv.text(100, 240, "有效学习率 ≈ η/(1−β)：β=0.9 时放大约 10 倍", size=18, color=AM, alpha=a2)
    a3 = appear(t, 2.6, 0.8)
    cv.text(100, 280, "作用：陡方向的正负震荡被平均掉，平方向的分量被累积", size=17, color=DIM, alpha=a3)
    a4 = appear(t, 3.6, 0.9)
    if a4 > 0:
        sx, sy = contour(cv, 100, 330, 420, 300, a4, label=False)
        path(cv, sx, sy, TOY["sgd"]["path"], COL["sgd"], 1.8, 0.75 * a4)
        path(cv, sx, sy, TOY["momentum"]["path"], COL["momentum"], 2.2, a4)
    a5 = appear(t, 5.0, 0.8)
    cv.text(560, 360, "灰 = SGD", size=16, color=COL["sgd"], alpha=a5)
    cv.text(560, 392, "蓝 = Momentum", size=16, color=COL["momentum"], alpha=a5)
    a6 = appear(t, 5.8, 0.8)
    cv.text(560, 440, "同一学习率下：", size=16, color=INK, alpha=a6)
    cv.text(560, 472, "SGD 5.7e-2 → Momentum 4.4e-2", size=15, color=DIM, alpha=a6)
    cv.text(560, 504, "（60 步后损失）", size=14, color=FAINT, alpha=a6)
    a7 = appear(t, 6.8, 0.8)
    cv.text(560, 552, "但动量还是“一个学习率打天下”", size=16, color=MG, alpha=a7)


def s4(cv, t, dur):
    title(cv, "② RMSProp：每个参数自己的步长")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\qquad "
                        r"\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{v_t}+\varepsilon}", 20, INK)
    a2 = appear(t, 1.6, 0.8)
    cv.text(100, 236, "除以 √v：梯度一直很大的参数，步子自动变小；梯度很小的，步子自动变大",
            size=17, color=AM, alpha=a2)
    a3 = appear(t, 2.8, 0.9)
    if a3 > 0:
        rows = STEPS[:6]
        cv.text(100, 300, "本模型第一步：SGD 的步长 vs Adam 的步长", size=16, color=FAINT, alpha=a3)
        mx = max(r["sgd"] for r in rows)
        for i, r in enumerate(rows):
            aa = appear(t, 3.0 + 0.4 * i, 0.5)
            if aa <= 0:
                continue
            y = 336 + i * 42
            cv.text(100, y, r["name"].replace("l0.", "L0.").replace("l1.", "L1."), size=15,
                    color=INK, alpha=aa)
            cv.rect(300, y - 12, 300 + r["sgd"] / mx * 260, y + 2, COL["sgd"], width=0,
                    alpha=0.8 * aa, fill_alpha=0.5 * aa, radius=2)
            cv.text(584, y, f"{r['sgd']:.1e}", size=13, color=COL["sgd"], alpha=aa)
            cv.rect(700, y - 12, 700 + r["adam"] / mx * 260, y + 2, CY, width=0,
                    alpha=0.8 * aa, fill_alpha=0.5 * aa, radius=2)
            cv.text(984, y, f"{r['adam']:.1e}", size=13, color=CY, alpha=aa)
    a4 = appear(t, 6.4, 0.8)
    cv.text(100, 610, "SGD 下 w_head 的步长是 l1.W_qkv 的 31 倍；Adam 下两者完全相同",
            size=17, color=CY, alpha=a4)


def s5(cv, t, dur):
    title(cv, "③ Adam：动量 + 自适应 + 偏差修正")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 182, r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad "
                        r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2", 18, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 100, 232, r"\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}",
          19, AM)
    a3 = appear(t, 2.8, 0.8)
    math_(cv, 100, 292, r"\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}",
          20, CY)
    a4 = appear(t, 3.8, 0.8)
    cv.text(100, 344, "为什么必须做偏差修正：m、v 都从 0 出发，前几步被严重低估", size=17,
            color=DIM, alpha=a4)
    a5 = appear(t, 4.8, 0.9)
    if a5 > 0:
        X0, Y0, X1, Y1 = 620, 180, 1120, 400
        axes(cv, X0, Y0, X1, Y1, a5, "步数 t", "修正因子")
        sx = lambda v: X0 + v / 400 * (X1 - X0)
        sy = lambda v: Y1 - v * (Y1 - Y0)
        for v in (0, 0.5, 1.0):
            cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a5)
            cv.text(X0 + 9, sy(v) + 6, f"{v:g}", size=13, color=FAINT, alpha=a5)
        curvy(cv, [(sx(t2), sy(BIAS["m"][t2 - 1])) for t2 in range(1, 401)], CY, 2.2, a5)
        curvy(cv, [(sx(t2), sy(BIAS["v"][t2 - 1])) for t2 in range(1, 401)], AM, 2.2, a5)
        cv.text(sx(210), sy(0.92), "1−β₁ᵗ（动量）", size=15, color=CY, alpha=a5)
        cv.text(sx(210), sy(0.42), "1−β₂ᵗ（二阶矩）", size=15, color=AM, alpha=a5)
    a6 = appear(t, 6.2, 0.8)
    cv.text(620, 430, "t=1 时 1−β₂ᵗ = 0.001：不修正的话", size=16, color=INK, alpha=a6)
    cv.text(620, 460, "分母 √v 小 31 倍，步长会被放大 31 倍", size=16, color=MG, alpha=a6)
    a7 = appear(t, 7.0, 0.8)
    cv.text(620, 510, "t=100 时 1−β₂ᵗ 才到 0.095", size=15, color=DIM, alpha=a7)


def s6(cv, t, dur):
    title(cv, "四种优化器，同一条等高水平线")
    a = appear(t, 0.3, 0.8)
    sx, sy = contour(cv, 120, 190, 620, 420, a)
    a2 = appear(t, 1.2, 0.9)
    for k in ("sgd", "momentum", "rmsprop", "adam"):
        path(cv, sx, sy, TOY[k]["path"], COL[k], 2.2, a2)
    a3 = appear(t, 3.0, 0.8)
    leg = [("SGD", COL["sgd"], f"{TOY['sgd']['loss'][-1]:.1e}"),
           ("Momentum", COL["momentum"], f"{TOY['momentum']['loss'][-1]:.1e}"),
           ("RMSProp", COL["rmsprop"], f"{TOY['rmsprop']['loss'][-1]:.0e}"),
           ("Adam", COL["adam"], f"{TOY['adam']['loss'][-1]:.1e}")]
    for i, (k, col, v) in enumerate(leg):
        y = 230 + i * 42
        cv.line((820, y - 6), (860, y - 6), col, width=3, alpha=a3)
        cv.text(870, y, k, size=17, color=col, alpha=a3)
        cv.text(1090, y, v, size=15, color=DIM, anchor="ra", alpha=a3)
    cv.text(820, 200, "60 步后的损失", size=15, color=FAINT, alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(820, 430, "RMSProp 在这里最快", size=16, color=AM, alpha=a4)
    cv.text(820, 462, "但它对学习率更敏感", size=16, color=DIM, alpha=a4)
    a5 = appear(t, 6.2, 0.8)
    cv.text(820, 520, "Adam 兼顾两者，", size=16, color=CY, alpha=a5)
    cv.text(820, 552, "是实践中的默认选择", size=16, color=CY, alpha=a5)


def s7(cv, t, dur):
    title(cv, "在本项目模型上的真实对比")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 200, 760, 500
    axes(cv, X0, Y0, X1, Y1, a, "epoch", "训练损失（对数）")
    lo, hi = 1e-2, 100
    sy = lambda v: Y1 - (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (Y1 - Y0)
    sx = lambda ep: X0 + ep / 2.0 * (X1 - X0)
    for v in (1e-2, 1e-1, 1, 10, 100):
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.35 * a)
        cv.text(X0 - 8, sy(v), f"{v:g}", size=13, color=FAINT, anchor="ra", alpha=a)
    for i, ep in enumerate((1, 2, 3)):
        cv.text(sx(ep - 1), Y1 + 22, str(ep), size=14, color=FAINT, anchor="ma", alpha=a)
    names = ["sgd", "sgd_big", "momentum", "rmsprop", "adam", "adamw"]
    cols = {"sgd": COL["sgd"], "sgd_big": MG, "momentum": BL,
            "rmsprop": AM, "adam": CY, "adamw": GR}
    for nm in names:
        L = REAL[nm]["loss"]
        pts = [(sx(i), sy(v)) for i, v in enumerate(L)]
        aa = appear(t, 1.2 + 0.35 * names.index(nm), 0.7)
        if aa <= 0:
            continue
        curvy(cv, pts, cols[nm], 2.4 if nm in ("adam", "adamw") else 1.8, aa)
        for x, y in pts:
            cv.circle(x, y, 3, cols[nm], width=0, alpha=aa, fill_alpha=aa)
    a2 = appear(t, 3.6, 0.8)
    leg = [("Adam / AdamW", CY, "7.0 → 0.047"), ("RMSProp", AM, "57.6 → 0.043"),
           ("SGD η=1e-3", COL["sgd"], "44.2 → 4.47"), ("Momentum η=1e-3", BL, "63.5 → 59.9"),
           ("SGD η=2e-2", MG, "70.7 → 72.0 发散")]
    for i, (k, col, v) in enumerate(leg):
        y = 230 + i * 44
        cv.line((800, y - 6), (840, y - 6), col, width=3, alpha=a2)
        cv.text(850, y, k, size=15, color=col, alpha=a2)
        cv.text(850, y + 22, v, size=13, color=DIM, alpha=a2)
    a3 = appear(t, 6.4, 0.8)
    cv.text(110, 560, "同一个模型、同一初始化、同一批数据顺序 —— 差别全部来自优化器", size=17,
            color=DIM, alpha=a3)


def s8(cv, t, dur):
    title(cv, "④ AdamW：把权重衰减解耦出来")
    a = appear(t, 0.3, 0.8)
    cv.text(100, 190, "Adam + L2（耦合）", size=19, color=MG, alpha=a)
    math_(cv, 100, 232, r"g_t\leftarrow g_t+\lambda\theta_{t-1}\ \Rightarrow\ "
                        r"\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}", 17, MG)
    a2 = appear(t, 2.0, 0.8)
    cv.text(100, 292, "问题：λθ 进了二阶矩，衰减被 √v 归一化，大梯度参数反而衰减更少", size=17,
            color=MG, alpha=a2)
    a3 = appear(t, 3.2, 0.8)
    cv.text(100, 356, "AdamW（解耦）", size=19, color=GR, alpha=a3)
    math_(cv, 100, 398, r"\theta_t=(1-\eta\lambda)\,\theta_{t-1}-\eta"
                        r"\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}", 19, GR)
    a4 = appear(t, 4.6, 0.8)
    cv.text(100, 456, "衰减与自适应步长彻底分开：每个参数按同一比例缩小", size=17, color=DIM, alpha=a4)
    a5 = appear(t, 5.6, 0.8)
    code(cv, 100, 500, [
        "if k.endswith((\"_g\", \"_b\")) or k == \"pos\":",
        "    p[k] -= lr * upd            # 不衰减",
        "else:",
        "    p[k] -= lr * (upd + wd * p[k])   # 解耦衰减",
    ], size=14, alpha=a5)
    a6 = appear(t, 7.0, 0.8)
    cv.text(700, 380, "哪些参数不衰减", size=18, color=AM, alpha=a6)
    for i, s_ in enumerate(["LayerNorm 的 γ、β（它们是归一化参数）",
                            "位置编码（每个位置一个向量）",
                            "所有 bias"]):
        cv.text(700, 418 + i * 34, "· " + s_, size=15, color=DIM, alpha=appear(t, 7.2 + 0.4 * i, 0.5))
    a7 = appear(t, 8.6, 0.8)
    cv.text(700, 546, "衰减它们会破坏归一化的语义", size=16, color=MG, alpha=a7)


def s9(cv, t, dur):
    title(cv, "学习率调度与小结")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 200, 620, 420
    axes(cv, X0, Y0, X1, Y1, a, "epoch", "学习率")
    sx = lambda e: X0 + e / (len(SCHED["ep"]) - 1) * (X1 - X0)
    sy = lambda v: Y1 - v / 2.2e-3 * (Y1 - Y0)
    curvy(cv, [(sx(e), sy(v)) for e, v in zip(SCHED["ep"], SCHED["cosine"])], CY, 2.6, a)
    for e in range(0, len(SCHED["ep"]), 3):
        cv.text(sx(e), Y1 + 20, str(e), size=13, color=FAINT, anchor="ma", alpha=a)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 470, "余弦退火：先大步走，接近收敛时小步微调", size=17, color=DIM, alpha=a2)
    a3 = appear(t, 2.8, 0.8)
    cv.text(110, 506, "本课峰值 2e-3，8 epoch 退到约 0", size=16, color=FAINT, alpha=a3)

    a4 = appear(t, 3.8, 0.8)
    items = [("SGD", "病态方向上震荡，需要手调 η"),
             ("Momentum", "惯性抵消震荡，有效 η≈η/(1−β)"),
             ("RMSProp", "逐参数归一化，对 η 更宽容"),
             ("Adam", "动量 + 归一化 + 偏差修正"),
             ("AdamW", "解耦衰减：LN/位置编码/bias 不衰减")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 3.8 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 200 + i * 56
        cv.text(680, y, k, size=19, color=CY if i >= 3 else DIM, alpha=aa)
        cv.text(680, y + 24, v, size=15, color=INK, alpha=aa)
    a5 = appear(t, 6.6, 0.8)
    chip(cv, 110, 560, "本课全程用 AdamW + 余弦退火，实测 3 epoch 就把损失压到 0.047", GR, a5, 18)


SHOTS = [
    dict(name="title", say=[
        "第七集讲训练方法。核心问题：参数该怎么走。",
        "从最朴素的梯度下降到 AdamW，每一种都给出公式、直觉和真实轨迹。",
    ], draw=s1, min_dur=11.0),
    dict(name="illcond", say=[
        "先说为什么普通梯度下降不够。",
        {"say": "看这条二维曲面：一个方向曲率大，另一个方向小，比值是 $50$。",
         "tts": "看这条二维曲面：一个方向曲率大，另一个方向小，比值是五十。"},
        "学习率被迫由最陡的方向决定，于是最陡方向震荡，最平方向爬得极慢。",
        "神经网络里这种方向差异是常态——本模型各层梯度范数从十二到七十，差近六倍。",
    ], draw=s2, min_dur=19.0),
    dict(name="momentum", say=[
        "第一个改进是动量。",
        {"say": "把梯度做指数加权平均，θ 减去的是平均后的方向。",
         "tts": "把梯度做指数加权平均，theta 减去的是平均后的方向。"},
        "陡方向的正负震荡被平均掉，平方向的分量被累积。",
        {"say": "注意有效学习率被放大了约十倍：η 除以 $1$ 减 β。",
         "tts": "注意有效学习率被放大了约十倍：eta 除以一减 beta。"},
    ], draw=s3, min_dur=18.0),
    dict(name="rmsprop", say=[
        "第二个改进是 RMSProp：让每个参数有自己的步长。",
        "用梯度平方的滑动平均开根号做分母：梯度一直很大的参数步子变小，梯度很小的变大。",
        {"say": "看真实数据：第一步时 SGD 下输出头的步长是某个 qkv 矩阵的三十一倍；Adam 下两者完全相同。",
         "tts": "看真实数据：第一步时 SGD 下输出头的步长是某个 qkv 矩阵的三十一倍；Adam 下两者完全相同。"},
    ], draw=s4, min_dur=18.0),
    dict(name="adam", say=[
        "Adam 把两者合起来：一阶矩加二阶矩。",
        "但有个细节：两个矩都从零出发，前几步被严重低估。",
        {"say": "所以要除以 $1$ 减 β 的 t 次方做偏差修正。",
         "tts": "所以要除以一减 beta 的 t 次方做偏差修正。"},
        {"say": "第一步时二阶矩的修正因子只有 $0.001$：不修正的话，分母小三十一倍，步长会被放大三十一倍。",
         "tts": "第一步时二阶矩的修正因子只有零点零零一：不修正的话，分母小三十一倍，步长会被放大三十一倍。"},
    ], draw=s5, min_dur=20.0),
    dict(name="paths", say=[
        "把四种优化器的真实轨迹画在同一条等高线上。",
        "SGD 之字形；动量直一些；RMSProp 几乎直接冲到底；Adam 介于两者之间。",
        {"say": "六十步后：SGD 五点七乘十的负二次方，动量四点四，RMSProp 到十的负七十七次方。",
         "tts": "六十步后：SGD 五点七乘十的负二次方，动量四点四乘十的负二次方，RMSProp 到了十的负七十七次方。"},
    ], draw=s6, min_dur=17.0),
    dict(name="real", say=[
        "更有说服力的是在真实模型上的对比。",
        {"say": "同一个模型、同一初始化：Adam 三轮就到 $0.047$；SGD 用合适学习率只到 $4.47$。",
         "tts": "同一个模型、同一初始化：Adam 三轮就到零点零四七；SGD 用合适学习率只到四点四七。"},
        {"say": "而 SGD 若用和 Adam 同量级的学习率，直接发散到 $72$。",
         "tts": "而 SGD 若用和 Adam 同量级的学习率，直接发散到七十二。"},
        "这就是为什么实践里默认用 Adam。",
    ], draw=s7, min_dur=19.0),
    dict(name="adamw", say=[
        "最后是 AdamW 的那个 W：解耦权重衰减。",
        "Adam 加 L2 是把 λθ 塞进梯度，于是衰减也被二阶矩归一化了——大梯度参数反而衰减更少。",
        "AdamW 把衰减单独拿出来，每个参数按同一比例缩小。",
        "而且有一类参数不该衰减：LayerNorm 的 γ、β，位置编码，以及所有 bias。",
    ], draw=s8, min_dur=19.0),
    dict(name="summary", say=[
        "最后是学习率调度：余弦退火，先大步走，接近收敛时小步微调。",
        "五种优化器的定位小结在这里。本课全程用 AdamW 加余弦退火。",
    ], draw=s9, min_dur=14.0),
]

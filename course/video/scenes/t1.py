"""Transformer 教程 · 第一部：为什么长这样（背景 → 检索 → 掩码）—— 6 个镜头。"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import math_, math_w, title, chip, axes, curvy, code
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)


def s1(cv, t, dur):
    cv.text(640, 246, "Transformer 从零实现　·　第一部", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 340, "为什么长这样", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 400, "从 RNN 的两个硬伤，到一次软检索", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((480, 440), (800, 440), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 508, r"a_{ts}=\mathrm{softmax}_s\!\left(\frac{q_t^{\top}k_s}{\sqrt{d_h}}\right),"
                            r"\quad y_t=\sum_s a_{ts}v_s", 23, DIM, anchor="mc")
    cv.text(640, 600, "配套代码：tutorial/code/", size=18, color=FAINT, anchor="ma",
            alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "RNN 的两个硬伤")
    a = appear(t, 0.3, 0.8)
    # 串行链
    for i in range(5):
        aa = appear(t, 0.3 + 0.35 * i, 0.4)
        x = 120 + i * 96
        cv.rect(x, 220, x + 70, 268, CY if i else DIM, width=1.2, alpha=0.85 * aa,
                fill_alpha=0.10 * aa, radius=6)
        cv.text(x + 35, 250, f"h{i+1}", size=18, color=CY if i else DIM, anchor="ma", alpha=aa)
        if i:
            cv.arrow((x - 22, 244), (x - 4, 244), DIM, width=1.4, alpha=0.8 * aa)
    cv.text(120, 300, "串行：必须等 h(k−1) 算完 → 不能并行", size=19, color=MG, alpha=a)

    # 梯度指数
    a2 = appear(t, 2.4, 0.9)
    if a2 > 0:
        X0, Y0, X1, Y1 = 120, 360, 560, 560
        axes(cv, X0, Y0, X1, Y1, a2, "反传步数", "‖∂h_t/∂h_s‖（对数）")
        sx = lambda v: X0 + v / 40 * (X1 - X0)
        sy = lambda v: Y1 - (v + 6) / 12 * (Y1 - Y0)
        for v in (-4, 0, 4):
            cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a2)
            cv.text(X0 + 9, sy(v) + 6, f"1e{v}", size=14, color=FAINT, alpha=a2)
        curvy(cv, [(sx(n), sy(max(-6, -0.25 * n))) for n in range(0, 41)], MG, 2.4, a2)
        curvy(cv, [(sx(n), sy(min(6, 0.22 * n))) for n in range(0, 41)], AM, 2.4, 0.9 * a2)
        cv.text(sx(26), sy(-3.4), "σ<1：消失", size=17, color=MG, alpha=a2)
        cv.text(sx(26), sy(3.4), "σ>1：爆炸", size=17, color=AM, alpha=a2)

    a3 = appear(t, 4.4, 0.8)
    cv.text(640, 400, "两个硬伤", size=21, color=INK, alpha=a3)
    cv.text(640, 444, "① 长程依赖：连乘 → 指数衰减/爆炸", size=19, color=DIM, alpha=a3)
    cv.text(640, 480, "② 无法并行：T 步只能串行展开", size=19, color=DIM, alpha=a3)
    a4 = appear(t, 5.6, 0.8)
    chip(cv, 640, 546, "Transformer 的回答：位置之间直接相连 + 全并行", CY, a4, 19)


def s3(cv, t, dur):
    title(cv, "检索视角：注意力就是“按需回看”")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 180, "历史 = 一张键值表", size=19, color=FAINT, alpha=a)
    for i in range(6):
        aa = appear(t, 0.4 + 0.25 * i, 0.4)
        x = 110 + i * 92
        cv.rect(x, 210, x + 76, 244, BL, width=1, alpha=0.8 * aa, fill_alpha=0.10 * aa, radius=4)
        cv.text(x + 38, 232, f"k{i+1}", size=15, color=BL, anchor="ma", alpha=aa)
        cv.rect(x, 250, x + 76, 284, CY, width=1, alpha=0.8 * aa, fill_alpha=0.10 * aa, radius=4)
        cv.text(x + 38, 272, f"v{i+1}", size=15, color=CY, anchor="ma", alpha=aa)

    a2 = appear(t, 2.2, 0.8)
    cv.text(110, 330, "硬检索（不可微）", size=18, color=DIM, alpha=a2)
    math_(cv, 110, 372, r"y=v_{\arg\max_s q^{\top}k_s}", 20, DIM)

    a3 = appear(t, 3.4, 0.9)
    cv.text(110, 440, "软检索（可微）", size=18, color=CY, alpha=a3)
    math_(cv, 110, 486, r"a_s=\frac{e^{q^{\top}k_s/\sqrt{d_h}}}{\sum_{s'}e^{q^{\top}k_{s'}/\sqrt{d_h}}}",
          21, INK)
    math_(cv, 110, 560, r"y=\sum_s a_s v_s", 21, INK)

    a4 = appear(t, 5.0, 0.9)
    if a4 > 0:
        w = np.array([0.05, 0.08, 0.42, 0.25, 0.13, 0.07])
        for i in range(6):
            x = 660 + i * 82
            hh = w[i] * 320
            cv.rect(x, 420 - hh, x + 62, 420, CY, width=1, alpha=0.85 * a4,
                    fill_alpha=0.45 * a4, radius=3)
            cv.text(x + 31, 420 + 24, f"k{i+1}", size=15, color=FAINT, anchor="ma", alpha=a4)
            cv.text(x + 31, 420 - hh - 14, f"{w[i]:.2f}", size=14, color=INK, anchor="ma", alpha=a4)
        cv.text(660, 470, "权重 a_s 由查询与键的相似度决定", size=17, color=DIM, alpha=a4)
        cv.text(660, 502, "全程可微 → 可以学“该看哪里”", size=17, color=AM, alpha=a4)


def s4(cv, t, dur):
    title(cv, "为什么要除以 √d")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 200, r"q^{\top}k=\sum_{i=1}^{d_h}q_ik_i", 21, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 256, r"\mathbb{E}[q^{\top}k]=0,\qquad \mathrm{Var}(q^{\top}k)=d_h", 21, CY)
    a3 = appear(t, 2.8, 0.8)
    cv.text(110, 306, "点积的标准差 = √d；不缩放 → softmax 输入太宽 → 饱和、梯度趋零", size=18,
            color=DIM, alpha=a3)

    a4 = appear(t, 4.0, 0.9)
    if a4 > 0:
        X0, Y0, X1, Y1 = 110, 350, 600, 550
        axes(cv, X0, Y0, X1, Y1, a4, "d_h", "std(q^Tk)")
        sx = lambda v: X0 + v / 16 * (X1 - X0)
        sy = lambda v: Y1 - v / 4.2 * (Y1 - Y0)
        for v in (0, 1, 2, 3, 4):
            cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a4)
            cv.text(X0 + 9, sy(v) + 6, str(v), size=14, color=FAINT, alpha=a4)
        curvy(cv, [(sx(d), sy(math.sqrt(d))) for d in np.linspace(0.05, 16, 120)], MG, 2.2, a4)
        curvy(cv, [(sx(d), sy(1.0)) for d in np.linspace(0.2, 16, 120)], CY, 2.2, a4)
        cv.text(sx(12), sy(3.7), "√d", size=17, color=MG, alpha=a4)
        cv.text(sx(12), sy(1.35), "缩放后 = 1", size=17, color=CY, alpha=a4)

    a5 = appear(t, 6.0, 0.8)
    cv.text(660, 400, "本教程每头 d_h = 8", size=19, color=INK, alpha=a5)
    math_(cv, 660, 446, r"\sqrt{8}=2.83", 21, DIM)
    cv.text(660, 500, "这就是缩放因子的全部来历", size=18, color=DIM, alpha=a5)


def s5(cv, t, dur):
    title(cv, "因果掩码：只许看过去")
    a = appear(t, 0.3, 0.8)
    n = 10
    for i in range(n):
        for j in range(n):
            x = 110 + j * 30
            y = 200 + i * 30
            ok = j <= i
            cv.rect(x, y, x + 26, y + 26, CY if ok else (40, 50, 62), width=0,
                    alpha=(0.55 if ok else 0.9) * a, fill_alpha=(0.32 if ok else 0.5) * a)
    cv.text(110, 176, "上三角被置为 −∞ → 权重严格为 0", size=17, color=FAINT, alpha=a)
    cv.text(110, 200 + n * 30 + 10, "key（历史）→", size=15, color=FAINT, alpha=a)
    cv.text(110, 200 + n * 30 + 34, "↓ query（当前）", size=15, color=FAINT, alpha=a)

    a2 = appear(t, 2.0, 0.8)
    math_(cv, 500, 240, r"M_{ts}=\begin{cases}0,& s\le t\\-\infty,& s>t\end{cases}", 19, INK)
    a3 = appear(t, 3.2, 0.8)
    cv.text(500, 330, "为什么训练时也要掩？", size=20, color=AM, alpha=a3)
    cv.text(500, 372, "部署时第 t 步拿不到第 t+1 个 token。", size=18, color=DIM, alpha=a3)
    cv.text(500, 404, "不掩 → 学到依赖未来的作弊解 → 部署失效。", size=18, color=DIM, alpha=a3)

    a4 = appear(t, 5.0, 0.8)
    code(cv, 500, 480, [
        "# 正确",
        "s = s.masked_fill(mask, -1e9)",
        "# 错误：忘记掩码",
        "s = s                      ",
    ], size=16, hl={1}, hl_color=CY)


def s6(cv, t, dur):
    title(cv, "第一部小结")
    items = [("RNN 的硬伤", "串行 + 梯度指数衰减/爆炸"),
             ("注意力的本质", "软检索：按相似度加权平均历史"),
             ("√d 的来历", "点积方差 = d，缩放让 softmax 不饱和"),
             ("因果掩码", "训推一致：训练时也不许看未来"),
             ("代价", "O(T²) 注意力矩阵；换来 O(1) 路径与全并行")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.62 * i, 0.55)
        if aa <= 0:
            continue
        y = 200 + i * 66
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=23, color=CY, alpha=aa, weight="bold")
        cv.text(330, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.0, 0.8)
    cv.text(130, 588, "第二部：把这个结构的前向与反向，一层一层算出来", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "从这一讲开始，我们把 Transformer 从零实现一遍。",
        "第一部只回答一个问题：它为什么长成这个样子。",
        "答案藏在 RNN 的两个硬伤里。",
    ], draw=s1, min_dur=11.0),
    dict(name="why", say=[
        "先看 RNN。它有两个硬伤。",
        "第一，长程依赖靠梯度连乘，会指数衰减或爆炸。",
        {"say": "第二，h t 依赖 h t 减一，所以一条长度为 $T$ 的序列只能串行展开，无法并行。",
         "tts": "第二，h t 依赖 h t 减一，所以一条长度为 T 的序列只能串行展开，无法并行。"},
        "Transformer 的两个回答是：位置之间直接相连，以及全并行。",
    ], draw=s2, min_dur=18.0),
    dict(name="retrieval", say=[
        "那怎么做到“直接相连”？把注意力看成一个检索过程。",
        "历史就是一张键值表，当前时刻有一个查询。",
        "硬检索是取最相似的那一项，不可微；软检索是按相似度加权平均，处处可微。",
        "权重就是注意力，它由查询和键的点积决定——于是我们可以学“该看哪里”。",
    ], draw=s3, min_dur=19.0),
    dict(name="scale", say=[
        "接下来回答一个细节：为什么除以根号 d。",
        {"say": "假设 q 与 k 各分量独立、零均值、单位方差，点积的方差正好等于 $d_h$。",
         "tts": "假设 q 与 k 各分量独立、零均值、单位方差，点积的方差正好等于 d 下标 h。"},
        {"say": "所以标准差是根号 $d_h$；不缩放，softmax 就会饱和。本教程每头 $d_h$ 等于 $8$，缩放因子 $2.83$。",
         "tts": "所以标准差是根号 d 下标 h；不缩放，softmax 就会饱和。本教程每头 d 下标 h 等于八，缩放因子二点八三。"},
    ], draw=s4, min_dur=18.0),
    dict(name="mask", say=[
        "最后一个约束：因果掩码。",
        "把上三角置为负无穷，未来权重严格为零。",
        "注意训练时也要掩——否则模型会学到依赖未来的解，部署时那些输入根本不存在。",
    ], draw=s5, min_dur=16.0),
    dict(name="summary", say=[
        "第一部到这里：硬伤、检索、缩放、掩码。",
        "下一部把它的前向与反向一层一层算出来，并给出可以逐行验证的实现。",
    ], draw=s6, min_dur=12.0),
]

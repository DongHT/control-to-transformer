"""第 8 讲：因果 Transformer（原理版）—— 7 个镜头。"""
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


def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 8 讲　因果 Transformer", size=60, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "从软检索推导到 int8 部署", size=31, color=DIM, anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"a_t\propto\exp\!\left(\frac{q_t^\top k_t}{\sqrt d}\right),\quad "
                            r"y=\sum a_t v_t", 24, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch08　　代码 · model.py / export.py", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(1.2 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


def s2(cv, t, dur):
    title(cv, "注意力 = 可微的“回看”")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 190, "硬检索（不可微）", size=20, color=FAINT, alpha=a)
    math_(cv, 110, 232, r"y=v_{\arg\max_t\, q^\top k_t}", 22, DIM)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 284, "软检索（可微）", size=20, color=CY, alpha=a2)
    math_(cv, 110, 344, r"a_t=\frac{\exp(q^\top k_t/\sqrt d)}{\sum_{s\le t}\exp(q^\top k_s/\sqrt d)}", 24, INK)
    math_(cv, 110, 420, r"y=\sum_{t}a_t\,v_t", 24, INK)
    a3 = appear(t, 3.4, 0.8)
    cv.text(110, 500, "权重 a_t 就是“注意力”：它由查询和键的相似度决定", size=20, color=DIM, alpha=a3)

    a4 = appear(t, 4.4, 0.9)
    if a4 > 0:
        # attention bars over 24 tokens
        cv.text(700, 190, "24 个 token 上的注意力", size=18, color=FAINT, alpha=a4)
        rng = np.random.default_rng(3)
        w = np.exp(-np.arange(24)[::-1] / 9.0) + 0.12 * rng.random(24)
        w /= w.sum()
        for i in range(24):
            x = 700 + i * 19
            hh = float(w[i]) / w.max() * 190
            cv.rect(x, 430 - hh, x + 14, 430, CY, width=1, alpha=0.8 * a4,
                    fill_alpha=0.55 * a4, radius=2)
        cv.line((700, 432), (700 + 24 * 19, 432), (92, 110, 132), width=1.2, alpha=0.8 * a4)
        cv.text(700, 458, "过去 ←", size=16, color=FAINT, alpha=a4)
        cv.text(700 + 24 * 19, 458, "现在", size=16, color=FAINT, anchor="ra", alpha=a4)


def s3(cv, t, dur):
    title(cv, "为什么除以 √d")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 202, r"q^\top k=\sum_{i=1}^{d}q_i k_i", 22, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 276, r"\mathbb{E}[q^\top k]=0,\qquad \mathrm{Var}(q^\top k)=d", 22, CY)
    a3 = appear(t, 2.8, 0.8)
    cv.text(110, 320, "所以点积的标准差是 √d；不缩放会让 softmax 进入饱和区", size=20, color=DIM, alpha=a3)

    a4 = appear(t, 4.0, 0.9)
    if a4 > 0:
        X0, Y0, X1, Y1 = 110, 360, 640, 540
        axes(cv, X0, Y0, X1, Y1, a4, "d_head", "softmax 输入的标准差")
        sx = lambda v: X0 + (v - 0) / 16.0 * (X1 - X0)
        sy = lambda v: Y1 - v / 4.2 * (Y1 - Y0)
        for v in (0, 1, 2, 3, 4):
            cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a4)
            cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a4)
        curvy(cv, [(sx(d), sy(math.sqrt(d))) for d in np.linspace(0.05, 16, 120)], MG, 2.2, a4)
        curvy(cv, [(sx(d), sy(math.sqrt(d) / math.sqrt(d))) for d in np.linspace(0.2, 16, 120)],
              CY, 2.2, a4)
        cv.text(sx(13), sy(3.9), "√d", size=18, color=MG, alpha=a4)
        cv.text(sx(13), sy(1.25), "归一化后 = 1", size=18, color=CY, alpha=a4)

    a5 = appear(t, 6.0, 0.8)
    cv.text(700, 380, "本课每头 d = 8", size=20, color=INK, alpha=a5)
    math_(cv, 700, 424, r"\sqrt{8}=2.83", 22, DIM)
    cv.text(700, 480, "所以缩放因子就是 2.83", size=19, color=DIM, alpha=a5)


def s4(cv, t, dur):
    title(cv, "因果掩码：训练时也不许看未来")
    a = appear(t, 0.3, 0.8)
    # mask matrix
    n = 12
    for i in range(n):
        for j in range(n):
            x = 110 + j * 26
            y = 200 + i * 26
            col = CY if j <= i else (40, 50, 62)
            cv.rect(x, y, x + 22, y + 22, col, width=0,
                    alpha=(0.55 if j <= i else 0.9) * a, fill_alpha=(0.35 if j <= i else 0.5) * a)
    cv.text(110, 152, "因果掩码（12×12 示意）", size=17, color=FAINT, alpha=a)
    cv.text(110, 176, "上三角被掩掉 → 权重严格为 0", size=17, color=MG, alpha=a)
    a2 = appear(t, 2.0, 0.8)
    math_(cv, 520, 240, r"M_{ts}=\begin{cases}0,& s\le t\\ -\infty,& s>t\end{cases}", 20, INK)
    a3 = appear(t, 3.4, 0.8)
    cv.text(520, 330, "为什么训练时也要掩？", size=20, color=AM, alpha=a3)
    cv.text(520, 372, "部署时第 t 步拿不到第 t+1 个 token", size=19, color=DIM, alpha=a3)
    cv.text(520, 404, "不掩 → 训练学到作弊解 → 部署立刻失效", size=19, color=DIM, alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    chip(cv, 520, 470, "这是最容易被忽略的一条正确性约束", MG, a4, 19)


def s5(cv, t, dur):
    title(cv, "18,177 个参数：一项项数出来")
    rows = [("输入投影 6→32", "224", DIM),
            ("位置编码 24×32", "768", DIM),
            ("×2 层：LN1 / QKV 32→96", "64 / 3168", CY),
            ("×2 层：输出投影 32→32", "1056", CY),
            ("×2 层：LN2 / FFN 32→64→32", "64 / 4192", CY),
            ("末端 LN + 输出头", "64 + 33", AM)]
    for i, (k, v, col) in enumerate(rows):
        aa = appear(t, 0.5 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 200 + i * 52
        cv.text(110, y, k, size=20, color=col, alpha=aa)
        cv.text(880, y, v, size=20, color=INK, anchor="ra", alpha=aa)
        cv.line((110, y + 16), (940, y + 16), (60, 76, 94), width=1, alpha=0.45 * aa)
    a = appear(t, 4.0, 0.8)
    cv.text(110, 540, "单层 8,544 → 两层 17,088", size=21, color=DIM, alpha=a)
    math_(cv, 110, 596, r"224+768+2\times8544+64+33=18{,}177", 24, CY)


def s6(cv, t, dur):
    title(cv, "int8 部署：误差界与实测")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 206, r"s=\frac{\max|W|}{127},\qquad q=\mathrm{round}(W/s),\qquad \hat W=q\,s", 21, INK)
    a2 = appear(t, 1.8, 0.8)
    math_(cv, 110, 262, r"|W-\hat W|\le s/2=\frac{\max|W|}{254}", 22, CY)
    a3 = appear(t, 3.0, 0.8)
    cv.text(110, 320, "多层串联时 LayerNorm 会吸收一部分误差", size=19, color=DIM, alpha=a3)

    a4 = appear(t, 4.2, 0.9)
    cards = [("71 KB → 16.6 KB", "int8（base64 后 29.8 KB）", 4.2, AM),
             ("0.14 N", "单点最大误差 = 1.2% × 12 N", 5.0, CY),
             ("0.32 ms", "浏览器单步前向（50 Hz 占 1.6%）", 5.8, GR)]
    for txt, sub, t0, col in cards:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 380 + (t0 - 4.2) * 66
        cv.text(110, y, txt, size=26, color=col, weight="bold", alpha=aa)
        cv.text(430, y, sub, size=18, color=FAINT, alpha=aa)

    a5 = appear(t, 6.6, 0.8)
    cv.text(700, 380, "闭环影响：", size=19, color=INK, alpha=a5)
    cv.text(700, 414, "成功率 100% 不变", size=19, color=DIM, alpha=a5)
    cv.text(700, 446, "摆起 2.70 → 2.74 s", size=19, color=DIM, alpha=a5)


def s7(cv, t, dur):
    title(cv, "注意力实测：它在做相位估计")
    a = appear(t, 0.3, 0.8)
    rows = [("摆起（|θ|>0.5）", 20.1, 16.6, CY), ("稳摆（|θ|<0.08）", 18.1, 21.1, AM)]
    cv.text(110, 190, "等效跨度（步）", size=18, color=FAINT, alpha=a)
    cv.text(700, 190, "峰值位置（下标，23=最新）", size=18, color=FAINT, alpha=a)
    for i, (k, span, cen, col) in enumerate(rows):
        aa = appear(t, 0.6 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 240 + i * 56
        cv.text(110, y, k, size=19, color=col, alpha=aa)
        cv.rect(330, y - 14, 330 + span / 24 * 300, y + 4, col, width=1,
                alpha=0.8 * aa, fill_alpha=0.35 * aa, radius=3)
        cv.text(650, y, f"{span}", size=18, color=INK, alpha=aa)
        cv.text(780, y, f"{cen}", size=18, color=INK, alpha=aa)

    a2 = appear(t, 3.4, 0.8)
    cv.line((330, 470), (330 + 300, 470), (92, 110, 132), width=1.4, alpha=0.8 * a2)
    cv.text(330, 492, "最旧（下标 0）", size=16, color=FAINT, alpha=a2)
    cv.text(330 + 300, 492, "最新（下标 23）", size=16, color=FAINT, anchor="ra", alpha=a2)
    a3 = appear(t, 4.6, 0.8)
    cv.text(110, 540, "等效跨度 18–20 → 覆盖大半个窗口", size=21, color=CY, alpha=a3)
    cv.text(110, 576, "但峰值会换挡：中段 ↔ 最近第 2 步", size=20, color=DIM, alpha=a3)
    a4 = appear(t, 5.8, 0.8)
    chip(cv, 110, 620, "摆起估相位（看整段斜坡）；稳摆估变化率（锁最近一步）", AM, a4, 18)


SHOTS = [
    dict(name="title", say=[
        "上一讲把问题定义清楚了：从二十四步历史到一个力。",
        "这一讲把这个函数逼近做到底：为什么是注意力、缩放因子从哪来、参数量怎么数、量化误差多少。",
    ], draw=s1, min_dur=11.0),
    dict(name="attn", say=[
        "注意力本质上是一个可微的检索。",
        "硬检索是“取最相似的那一项”，不可微；软检索是“按相似度加权平均”，处处可微。",
        "权重就是注意力，它由查询和键的点积决定。",
    ], draw=s2, min_dur=17.0),
    dict(name="scale", say=[
        "那为什么要除以根号 d？",
        {"say": "假设 q 和 k 的每个分量独立、零均值、单位方差，那么点积的方差正好等于 $d$。",
         "tts": "假设 q 和 k 的每个分量独立、零均值、单位方差，那么点积的方差正好等于 d。"},
        "所以点积的标准差是根号 d；不缩放，softmax 就会进入饱和区，梯度接近零。",
        {"say": "本课每个头 $d=8$，缩放因子就是 $2.83$。",
         "tts": "本课每个头 d 等于八，缩放因子就是二点八三。"},
    ], draw=s3, min_dur=19.0),
    dict(name="mask", say=[
        "第二个关键设计是因果掩码。",
        "把上三角位置在 softmax 之前设为负无穷，未来权重严格为零。",
        "注意训练时也要掩：否则模型会学到一个依赖未来的解，部署时根本拿不到那些输入。",
    ], draw=s4, min_dur=17.0),
    dict(name="params", say=[
        "参数量可以一项项数出来。",
        "输入投影两百二十四，位置编码七百六十八，单层八千五百四十四，两层一万七千零八十八。",
        {"say": "加上末端的归一化和输出头，一共 $18{,}177$，和代码打印完全一致。",
         "tts": "加上末端的归一化和输出头，一共一万八千一百七十七，和代码打印完全一致。"},
    ], draw=s5, min_dur=17.0),
    dict(name="quant", say=[
        "部署到浏览器要量化。",
        {"say": "逐张量 int8：单个权重误差不超过量程除以 $254$。",
         "tts": "逐张量 int8：单个权重误差不超过量程除以二百五十四。"},
        {"say": "实测单点最大误差 $0.14$ 牛顿，只有量程的百分之一点二，闭环看不出差别。",
         "tts": "实测单点最大误差零点一四牛顿，只有量程的百分之一点二，闭环看不出差别。"},
    ], draw=s6, min_dur=18.0),
    dict(name="attnview", say=[
        "最后看它到底在看什么。",
        {"say": "第 2 层的等效跨度是 $18$ 到 $20$ 步，覆盖大半个窗口；但峰值位置会换挡。",
         "tts": "第二层的等效跨度是十八到二十步，覆盖大半个窗口；但峰值位置会换挡。"},
        {"say": "摆起时峰值落在窗口中段，稳摆时锁到最近第 2 步。",
         "tts": "摆起时峰值落在窗口中段，稳摆时锁到最近第二步。"},
        "所以它在摆起阶段估相位，在稳摆阶段估变化率。",
    ], draw=s7, min_dur=17.0),
]

"""Transformer 教程 · 第二部：前向与反向 —— 6 个镜头。"""
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
    cv.text(640, 246, "Transformer 从零实现　·　第二部", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 340, "前向与反向", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 400, "每一层的公式、雅可比，以及 3.0e-14 的对齐", size=28, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 440), (810, 440), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 508, r"\frac{\partial\mathcal{L}}{\partial z}"
                            r"=\frac{\gamma}{\sigma}\Big(g-\bar g-\hat z\,\overline{g\hat z}\Big)",
              24, DIM, anchor="mc")
    cv.text(640, 600, "target: 手写反向 == autograd", size=18, color=FAINT, anchor="ma",
            alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "结构：两层的 Pre-LN 解码器")
    a = appear(t, 0.3, 0.8)
    # 左侧主干
    X = 150
    ys = [176, 250, 424, 498, 572]
    labels = [("输入 x", "B×T×F", DIM), ("嵌入 + 位置编码", "B×T×d", CY)]
    for i, (lab, shp, col) in enumerate(labels):
        aa = appear(t, 0.3 + 0.4 * i, 0.4)
        cv.rect(X - 100, ys[i], X + 160, ys[i] + 50, col, width=1.2, alpha=0.8 * aa,
                fill_alpha=0.07 * aa, radius=6)
        cv.text(X + 30, ys[i] + 24, lab, size=16, color=col, anchor="ma", alpha=aa)
        cv.text(X + 30, ys[i] + 42, shp, size=12, color=FAINT, anchor="ma", alpha=aa)

    # 中间块（两层）
    a2 = appear(t, 1.6, 0.8)
    cv.rect(96, 330, 620, 610, DIM, width=1.0, alpha=0.5 * a2, radius=10)
    cv.text(110, 352, "× 2 层（参数：每层 8,544）", size=15, color=FAINT, alpha=a2)
    inner = [("LN1", CY), ("多头因果注意力", MG), ("⊕ 残差", AM),
             ("LN2", CY), ("FFN + GELU", CY), ("⊕ 残差", AM)]
    for i, (lab, col) in enumerate(inner):
        aa = appear(t, 1.8 + 0.35 * i, 0.4)
        y = 380 + i * 36
        cv.rect(130, y, 520, y + 28, col, width=1, alpha=0.75 * aa, fill_alpha=0.06 * aa, radius=5)
        cv.text(325, y + 19, lab, size=15, color=col, anchor="ma", alpha=aa)
        if i < len(inner) - 1:
            cv.arrow((325, y + 28), (325, y + 35), DIM, width=1.2, alpha=0.5 * aa)

    # 右侧头
    a3 = appear(t, 4.2, 0.8)
    for i, (lab, shp) in enumerate([("取最后 token", "B×d"), ("LN", "B×d"),
                                    ("线性 32→1", "B×1"), ("u = 12·tanh", "B")]):
        y = 330 + i * 62
        cv.rect(680, y, 900, y + 46, CY, width=1.2, alpha=0.8 * a3, fill_alpha=0.06 * a3, radius=6)
        cv.text(790, y + 20, lab, size=15, color=CY, anchor="ma", alpha=a3)
        cv.text(790, y + 38, shp, size=12, color=FAINT, anchor="ma", alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(960, 330, "参数量", size=15, color=FAINT, alpha=a4)
    cv.text(960, 380, "224", size=22, color=INK, alpha=a4)
    cv.text(960, 412, "768", size=22, color=INK, alpha=a4)
    cv.text(960, 444, "×2 8,544", size=22, color=INK, alpha=a4)
    cv.text(960, 476, "97", size=22, color=INK, alpha=a4)
    cv.text(960, 516, "= 18,177", size=24, color=CY, weight="bold", alpha=a4)


def s3(cv, t, dur):
    title(cv, "前向：六行公式走完一层")
    rows = [r"z = xW_{\mathrm{in}}^{\top}+b_{\mathrm{in}}+\mathrm{pos}",
            r"h = \mathrm{LN}_1(z)",
            r"q,k,v = hW_{qkv}^{\top}+b,\quad a=\mathrm{softmax}\Big(\frac{qk^{\top}}{\sqrt{d_h}}+M\Big)",
            r"z \leftarrow z + \big(\mathrm{Concat}(a^{(1..H)}v^{(1..H)})\big)W_o^{\top}",
            r"h_2 = \mathrm{LN}_2(z),\quad z \leftarrow z+W_2\,\mathrm{GELU}(W_1h_2+b_1)+b_2",
            r"u = u_{\max}\tanh\big(w^{\top}\mathrm{LN}(z_{T})+b\big)"]
    for i, tex in enumerate(rows):
        aa = appear(t, 0.4 + 0.75 * i, 0.6)
        if aa <= 0:
            continue
        math_(cv, 110, 190 + i * 74, tex, 19 if i != 2 else 17, INK)
    a = appear(t, 5.2, 0.8)
    cv.text(110, 640, "一行一个运算块；第 4 行有多头拼接，第 5 行是 FFN 与第二个残差", size=17,
            color=FAINT, alpha=a)


def s4(cv, t, dur):
    title(cv, "反向①：两个雅可比决定稳定性")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 180, "softmax", size=20, color=CY, alpha=a)
    math_(cv, 110, 226, r"\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)", 20, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 296, r"\frac{\partial\mathcal{L}}{\partial s_j}"
                        r"=a_j\Big(g_j-\sum_i a_ig_i\Big)", 21, CY)
    a3 = appear(t, 2.6, 0.8)
    cv.text(110, 344, "括号里是“减基线”：所有 g 相同时梯度为零", size=17, color=DIM, alpha=a3)

    a4 = appear(t, 3.6, 0.8)
    cv.text(110, 400, "LayerNorm", size=20, color=MG, alpha=a4)
    math_(cv, 110, 446, r"\frac{\partial\mathcal{L}}{\partial z}=\frac{\gamma}{\sigma}\Big(g-\bar g"
                        r"-\hat z\,\overline{g\hat z}\Big)", 21, MG)
    a5 = appear(t, 4.8, 0.8)
    cv.text(110, 500, "后两项分别抵消“整体平移”与“整体缩放”方向", size=17, color=DIM, alpha=a5)

    a6 = appear(t, 5.8, 0.9)
    cv.rect(700, 180, 1180, 380, CY, width=1.2, alpha=0.7 * a6, fill_alpha=0.05 * a6, radius=8)
    cv.text(722, 212, "两者共同的后果", size=19, color=CY, alpha=a6)
    cv.text(722, 252, "softmax：注意力越均匀，梯度越小", size=17, color=INK, alpha=a6)
    cv.text(722, 288, "LN：梯度天然正交于两个无用方向", size=17, color=INK, alpha=a6)
    cv.text(722, 336, "→ Pre-LN 结构能训得很深", size=18, color=AM, alpha=a6)


def s5(cv, t, dur):
    title(cv, "反向②：注意力与残差")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 196, r"\frac{\partial\mathcal{L}}{\partial V}=A^{\top}G,\qquad"
                        r"\frac{\partial\mathcal{L}}{\partial A}=GV^{\top}", 20, INK)
    a2 = appear(t, 1.8, 0.8)
    math_(cv, 110, 260, r"\frac{\partial\mathcal{L}}{\partial Q}=\frac{1}{\sqrt{d_h}}"
                        r"\frac{\partial\mathcal{L}}{\partial S}K,\qquad"
                        r"\frac{\partial\mathcal{L}}{\partial K}=\frac{1}{\sqrt{d_h}}"
                        r"\Big(\frac{\partial\mathcal{L}}{\partial S}\Big)^{\top}Q", 19, INK)
    a3 = appear(t, 3.2, 0.8)
    cv.text(110, 318, "其中 ∂L/∂S 由上一页的 softmax 雅可比给出（掩码位置自动为 0）", size=17,
            color=DIM, alpha=a3)

    a4 = appear(t, 4.2, 0.8)
    cv.text(110, 384, "残差", size=20, color=AM, alpha=a4)
    math_(cv, 110, 430, r"z_{\text{out}}=z+f(z)\ \Rightarrow\ "
                        r"\frac{\partial\mathcal{L}}{\partial z}"
                        r"=\frac{\partial\mathcal{L}}{\partial z_{\text{out}}}"
                        r"\Big(I+\frac{\partial f}{\partial z}\Big)", 19, AM)
    a5 = appear(t, 5.4, 0.8)
    cv.text(110, 490, "单位阵 I 就是“梯度高速公路”", size=18, color=DIM, alpha=a5)

    a6 = appear(t, 6.2, 0.9)
    if a6 > 0:
        # 梯度量级示意
        vals = [("head", 1.00), ("proj", 0.38), ("fc2", 0.28), ("qkv", 0.26),
                ("L1 qkv", 0.25), ("L0 qkv", 0.26)]
        cv.text(700, 200, "实测梯度范数（相对 head）", size=16, color=FAINT, alpha=a6)
        for i, (k, v) in enumerate(vals):
            y = 240 + i * 44
            cv.text(700, y + 12, k, size=15, color=INK, alpha=a6)
            cv.rect(780, y, 780 + v * 330, y + 20, CY, width=0, alpha=0.8 * a6,
                    fill_alpha=0.45 * a6, radius=3)
            cv.text(1120, y + 12, f"{v:.2f}", size=14, color=DIM, anchor="ra", alpha=a6)
        cv.text(700, 520, "只有 4 倍衰减 → 残差通路有效", size=17, color=AM, alpha=a6)


def s6(cv, t, dur):
    title(cv, "对齐：3.0e-14")
    a = appear(t, 0.3, 0.8)
    rows = [("前向输出 u", "3.1e-15", "float64 舍入量级", CY),
            ("损失 L", "< 1.0e-15", "完全一致", CY),
            ("31 个参数的梯度", "3.0e-14", "手推 == autograd", GR),
            ("中心差分（抽样 3 个元素）", "相对误差 5e-11~8e-9", "与框架无关的第三方验证", AM)]
    cv.text(110, 190, "量", size=16, color=FAINT, alpha=a)
    cv.text(560, 190, "最大绝对偏差", size=16, color=FAINT, alpha=a)
    cv.text(860, 190, "说明", size=16, color=FAINT, alpha=a)
    for i, (k, v, m, col) in enumerate(rows):
        aa = appear(t, 0.6 + 0.55 * i, 0.55)
        if aa <= 0:
            continue
        y = 240 + i * 62
        cv.text(110, y, k, size=19, color=INK, alpha=aa)
        cv.text(560, y, v, size=19, color=col, alpha=aa)
        cv.text(860, y, m, size=16, color=DIM, alpha=aa)
        cv.line((110, y + 18), (1180, y + 18), (60, 76, 94), width=1, alpha=0.4 * aa)
    a2 = appear(t, 3.8, 0.8)
    chip(cv, 110, 520, "如果符号或转置写错，偏差会到 1e-2 甚至形状不匹配", MG, a2, 19)
    a3 = appear(t, 4.8, 0.8)
    cv.text(110, 588, "这就是“实现正确”的定义：不是能跑，而是与 autograd 逐位一致", size=20,
            color=CY, alpha=a3)


SHOTS = [
    dict(name="title", say=[
        "第二部：把这个结构的前向与反向，一层一层算出来。",
        "目标是：手写的反向传播，和 autograd 的结果逐位一致。",
    ], draw=s1, min_dur=11.0),
    dict(name="arch", say=[
        "先看整体。输入是一段窗口，输出一个标量。",
        "中间是两层完全相同的块：先归一化、再做多头因果注意力、加残差；再归一化、做前馈、加残差。",
        {"say": "最后一层之后只取最后一个 token，过归一化和线性层，再乘 $12$ 取 tanh。",
         "tts": "最后一层之后只取最后一个 token，过归一化和线性层，再乘十二取 tanh。"},
        "参数量一万八千一百七十七。",
    ], draw=s2, min_dur=19.0),
    dict(name="forward", say=[
        "前向可以写成六行公式。",
        "嵌入加位置编码；归一化；投影出 QKV 并做带掩码的 softmax；加权求和后投影、加残差。",
        "然后第二个归一化、前馈、第二个残差；最后取最后一个 token 输出。",
        "每一行都是一个可以单独写单元测试的运算块。",
    ], draw=s3, min_dur=18.0),
    dict(name="jac1", say=[
        "反向的第一组关键公式是 softmax 和 LayerNorm 的雅可比。",
        {"say": "softmax 的雅可比展开后自带“减基线”：所有上游梯度相同时，结果为零。",
         "tts": "softmax 的雅可比展开后自带减基线：所有上游梯度相同时，结果为零。"},
        {"say": "LayerNorm 的梯度里有两项修正，分别抵消整体平移与整体缩放。",
         "tts": "LayerNorm 的梯度里有两项修正，分别抵消整体平移与整体缩放。"},
        "这两条决定了 Pre-LN 结构能训得很深。",
    ], draw=s4, min_dur=18.0),
    dict(name="jac2", say=[
        "第二组是注意力本身的反传。",
        {"say": "对 V 是 A 转置乘上游梯度；对 A 是上游梯度乘 V 转置；再经 softmax 得到对 S 的梯度。",
         "tts": "对 V 是 A 转置乘上游梯度；对 A 是上游梯度乘 V 转置；再经 softmax 得到对 S 的梯度。"},
        "Q 和 K 各自由 S 的梯度乘以对方、再除以根号 d 得到。",
        "残差则是一个单位阵：梯度原样穿过，这就是“高速公路”。",
    ], draw=s5, min_dur=19.0),
    dict(name="parity", say=[
        "结果：前向差 $3\\times10^{-15}$，31 个参数张量的梯度差 $3\\times10^{-14}$。",
        "另外用中心差分做了与框架无关的验证，相对误差在 $10^{-11}$ 量级。",
        "这才叫实现正确：不是能跑，而是与 autograd 逐位一致。",
    ], draw=s6, min_dur=16.0),
]

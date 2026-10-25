"""Transformer 教程 · 第五集：反向逐算子（雅可比推导 ↔ 代码）—— 9 个镜头。"""
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
SRC = os.path.join(ROOT, "tutorial", "code", "tinytf_numpy.py")
D = json.load(open(os.path.join(ROOT, "build", "out", "chA.json")))
BW = D["backward"]
_LINES = open(SRC, encoding="utf-8").read().split("\n")
_LINES2 = open(os.path.join(ROOT, "tutorial", "code", "parity_test.py"),
               encoding="utf-8").read().split("\n")


def src(anchor, n, dedent=None):
    for i, ln in enumerate(_LINES):
        if anchor in ln:
            block = _LINES[i:i + n]
            if dedent is None:
                dedent = len(block[0]) - len(block[0].lstrip())
            return [b[dedent:] if len(b) >= dedent else b for b in block]
    return [f"# anchor not found: {anchor}"]


def src2(anchor, n):
    for i, ln in enumerate(_LINES2):
        if anchor in ln:
            block = _LINES2[i:i + n]
            ded = len(block[0]) - len(block[0].lstrip())
            return [b[ded:] if len(b) >= ded else b for b in block]
    return [f"# anchor not found: {anchor}"]


def left(cv, t, tex, t0=0.4, size=18, x=100, y=196, lh=58):
    a = appear(t, t0, 0.6)
    if a > 0:
        for i, tx in enumerate(tex):
            math_(cv, x, y + i * lh, tx, size, INK)


def right(cv, t, snip, t0=1.3, csize=12, x=690, y=190, note=None, note_t=3.4):
    a = appear(t, t0, 0.6)
    if a > 0:
        code(cv, x, y, snip, size=csize, alpha=a)
    if note:
        a2 = appear(t, note_t, 0.7)
        if a2 > 0:
            cv.text(x, y + len(snip) * csize * 1.55 + 26, note, size=16, color=AM, alpha=a2)


def s1(cv, t, dur):
    cv.text(640, 240, "Transformer 从零实现　·　第五集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 334, "反向：逐雅可比", size=62, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 394, "每个梯度怎么来，代码怎么写", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 434), (810, 434), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 500, r"\frac{\partial\mathcal{L}}{\partial z_j}=\sum_i"
                            r"\frac{\partial\mathcal{L}}{\partial y_i}"
                            r"\frac{\partial y_i}{\partial z_j}", 25, DIM, anchor="mc")
    cv.text(640, 600, "目标：所有 31 个参数张量与 autograd 差 < 1e-13", size=18, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "反向的两个原则")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 180, "① 沿计算图逆序，每步只算“局部雅可比 × 上游梯度”", size=21, color=INK, alpha=a)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 236, r"\bar x=\sum_{\text{child}}\ \bar y\,\frac{\partial y}{\partial x},"
                        r"\qquad \bar y\equiv\frac{\partial\mathcal{L}}{\partial y}", 20, CY)
    a3 = appear(t, 2.8, 0.8)
    cv.text(110, 292, "② 前向时把需要的中间量存下来（cache），反向直接用", size=21, color=INK, alpha=a3)

    a4 = appear(t, 4.0, 0.8)
    rows = [("要存的量", "谁在用"),
            ("xhat, rstd", "LN 反向"),
            ("a (注意力权重)", "dS = a·(…)"),
            ("q, k, v", "dQ, dK"),
            ("pre (GELU 前)", "gelu_grad"),
            ("z (残差主干)", "dW₂ = z^T·act")]
    for i, (k, v) in enumerate(rows):
        aa = appear(t, 4.0 + 0.4 * i, 0.5)
        if aa <= 0:
            continue
        y = 360 + i * 42
        cv.text(110, y, k, size=18, color=CY, alpha=aa)
        cv.text(420, y, v, size=18, color=DIM, alpha=aa)
    a5 = appear(t, 7.0, 0.8)
    chip(cv, 110, 640, "省内存的写法是不存、反向重算——本教程选了“全存”，因为模型很小", DIM, a5, 17)


def s3(cv, t, dur):
    title(cv, "① 输出层：从 u 退回 r")
    left(cv, t, [r"\bar u=\frac{\partial\mathcal{L}}{\partial u}=1.0\quad(\mathcal{L}=\tfrac12(u-u^*)^2)",
                 r"\bar r=\bar u\cdot u_{\max}(1-\tanh^2 r)"], t0=0.4, size=18)
    right(cv, t, src("dr = du * self.u_max * (1.0 - np.tanh(cache[\"r\"]) ** 2)", 2),
          note="tanh 导数 = 1−tanh²，|u| 接近 12 时趋零（饱和）")
    a = appear(t, 4.4, 0.8)
    cv.text(100, 380, "实测：u = 4.153 → 1−tanh² ≈ 0.88，尚未饱和", size=18, color=CY, alpha=a)
    a2 = appear(t, 5.4, 0.8)
    cv.text(100, 420, "若输出长期贴近 ±12，这一层梯度会消失 —— 用 tanh 限幅的代价", size=17,
            color=DIM, alpha=a2)


def s4(cv, t, dur):
    title(cv, "② LayerNorm：两项修正从哪来")
    a = appear(t, 0.3, 0.7)
    math_(cv, 100, 186, r"\hat z=\frac{z-\mu}{\sigma},\quad y=\gamma\hat z+\beta,\quad g=\frac{\partial\mathcal{L}}{\partial y}",
          17, DIM)
    a2 = appear(t, 1.4, 0.7)
    math_(cv, 100, 236, r"\frac{\partial\mathcal{L}}{\partial\mu}=\sum_i g_i\Big(\frac{-1}{\sigma}\Big),"
                        r"\qquad \frac{\partial\mathcal{L}}{\partial\sigma^2}=\sum_i g_i(z_i-\mu)\Big(\frac{-1}{2\sigma^3}\Big)",
          16, INK)
    a3 = appear(t, 3.0, 0.7)
    math_(cv, 100, 300, r"\Rightarrow\ \frac{\partial\mathcal{L}}{\partial z_i}=\frac{g_i}{\sigma}"
                        r"-\frac{1}{\sigma d}\sum_j g_j-\frac{\hat z_i}{\sigma d}\sum_j g_j\hat z_j",
          18, CY)
    a4 = appear(t, 4.4, 0.7)
    math_(cv, 100, 360, r"=\frac{\gamma}{\sigma}\Big(g-\bar g-\hat z\,\overline{(g\hat z)}\Big)", 20, AM)
    a5 = appear(t, 5.4, 0.8)
    cv.text(100, 428, "中间两项正是“减整体平移”和“减整体缩放”：它们对应的方向不改变 y", size=16,
            color=DIM, alpha=a5)
    right(cv, t, src("def ln_backward(dy, cache):", 10), t0=2.2, csize=11,
          note="代码直接用化简后的紧凑式，比逐步展开更短、更稳")


def s5(cv, t, dur):
    title(cv, "③ softmax：雅可比与“减基线”")
    left(cv, t, [r"\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)",
                 r"\frac{\partial\mathcal{L}}{\partial s_j}=\sum_i g_i a_i(\delta_{ij}-a_j)"
                 r"=a_j\Big(g_j-\sum_i a_ig_i\Big)"], t0=0.4, size=18)
    right(cv, t, src("dsm = c[\"a\"] * (da - (da * c[\"a\"]).sum(-1, keepdims=True))", 1),
          t0=1.6, csize=12, note="一行就是全部：a·(da − Σ da·a)")
    a = appear(t, 4.6, 0.8)
    cv.text(100, 420, "含义：所有上游梯度相同时，改变分数不改变 softmax 输出 → 梯度为零", size=17,
            color=CY, alpha=a)
    a2 = appear(t, 5.6, 0.8)
    cv.text(100, 460, "注意：注意力越均匀（a 越接近 1/24），这一行给出的梯度越小", size=17,
            color=AM, alpha=a2)
    a3 = appear(t, 6.6, 0.8)
    cv.text(100, 500, "这也是“注意力塌缩”难训练的原因", size=16, color=DIM, alpha=a3)
    a4 = appear(t, 7.4, 0.8)
    chip(cv, 100, 560, "掩码位置 a=0 → dS=0，无需再 masked_fill", MG, a4, 18)


def s6(cv, t, dur):
    title(cv, "④ 注意力：5 个张量的梯度")
    left(cv, t, [r"\bar V=A^{\top}\bar Y,\qquad \bar A=\bar Y V^{\top}",
                 r"\bar S=A\odot\big(\bar A-\textstyle\sum_j A_j\bar A_j\big)",
                 r"\bar Q=\frac{1}{\sqrt{d_h}}\bar S K,\qquad \bar K=\frac{1}{\sqrt{d_h}}\bar S^{\top}Q"],
         t0=0.4, size=17, lh=56)
    right(cv, t, src("da = np.einsum(\"bhtd,bhsd->bhts\", dy, c[\"v\"])", 7), t0=1.4, csize=11,
          note="形状：(B,H,T,dh) 与 (B,H,T,T) 之间来回，错一个 transpose 就全错")
    a = appear(t, 4.6, 0.8)
    cv.text(100, 430, "为什么 A 的梯度是 dY·V^T：Y = A·V 对 A 线性", size=17, color=CY, alpha=a)
    a2 = appear(t, 5.6, 0.8)
    cv.text(100, 470, "为什么 Q 的梯度要乘 K（而不是 K^T）：见 s = q·k^T 的写法", size=17,
            color=DIM, alpha=a2)
    a3 = appear(t, 6.4, 0.8)
    cv.text(100, 510, "多头梯度合并回 (B,T,3d) 后，一次矩阵乘出 W_qkv 的梯度", size=17,
            color=DIM, alpha=a3)


def s7(cv, t, dur):
    title(cv, "⑤ FFN 与 GELU：一个真实的坑")
    left(cv, t, [r"\bar a=\bar z W_2,\qquad \frac{\partial\mathcal{L}}{\partial W_2}=z^{\top}a",
                 r"\bar p=\bar a\odot\mathrm{GELU}'(p),\qquad "
                 r"\frac{\partial\mathcal{L}}{\partial W_1}=p^{\top}h_2"], t0=0.4, size=17, lh=54)
    right(cv, t, src("dact = dz @ p[k + \"W2\"]", 6), t0=1.4, csize=11)
    a = appear(t, 4.2, 0.9)
    cv.text(100, 400, "坑：W₂ 的梯度用 dz（残差主干），不是 dact", size=19, color=MG, alpha=a)
    a2 = appear(t, 5.2, 0.8)
    code(cv, 100, 440, ["g[W2] = dact.T @ act   # (ff,ff)  ← 错",
                        "g[W2] = dz.T   @ act   # (d,ff)   ← 对"], size=13, hl={0}, hl_color=MG)
    a3 = appear(t, 6.4, 0.8)
    cv.text(100, 512, "因为 z = z₁ + W₂·act + b₂：对 W₂ 求导只留下“主干梯度 × 激活值”", size=16,
            color=DIM, alpha=a3)
    a4 = appear(t, 7.2, 0.8)
    cv.text(100, 552, "同理 b₂ 的梯度也是 dz.sum()，不是 dact.sum()", size=16, color=DIM, alpha=a4)


def s8(cv, t, dur):
    title(cv, "⑥ 残差与嵌入层：梯度分叉与汇聚")
    left(cv, t, [r"z_{\text{out}}=z+f(z)\ \Rightarrow\ \bar z=\bar z_{\text{out}}+"
                 r"\bar z_{\text{out}}\frac{\partial f}{\partial z}",
                 r"\frac{\partial\mathcal{L}}{\partial W_{\mathrm{in}}}=\bar z^{\top}x,\qquad "
                 r"\frac{\partial\mathcal{L}}{\partial\mathrm{pos}}=\sum_b\bar z"], t0=0.4, size=17, lh=58)
    right(cv, t, src("g[\"W_in\"] = dz.reshape(-1, d).T @ x.reshape(-1, self.F)", 3), t0=1.4,
          csize=11, note="pos 的梯度按 batch 求和：位置编码被所有样本共享")
    a = appear(t, 4.4, 0.8)
    cv.text(100, 430, "残差让梯度在两处“汇聚”：主干直通 + 分支回传，两者相加", size=17, color=CY, alpha=a)
    a2 = appear(t, 5.4, 0.8)
    cv.text(100, 470, "这也是为什么去掉残差后，第 0 层梯度会小一到两个数量级", size=17,
            color=DIM, alpha=a2)
    a3 = appear(t, 6.4, 0.9)
    if a3 > 0:
        g = BW["grads"]
        items = [("head", g["head.weight"]), ("L0.proj", g["blocks.0.proj.weight"]),
                 ("L0.fc2", g["blocks.0.fc2.weight"]), ("L0.qkv", g["blocks.0.qkv.weight"]),
                 ("L1.qkv", g["blocks.1.qkv.weight"])]
        cv.text(690, 400, "实测梯度范数（相对 head）", size=15, color=FAINT, alpha=a3)
        mx = items[0][1]
        for i, (k, v) in enumerate(items):
            y = 436 + i * 34
            cv.text(690, y + 10, k, size=14, color=INK, alpha=a3)
            cv.rect(790, y, 790 + v / mx * 300, y + 18, CY, width=0, alpha=0.8 * a3,
                    fill_alpha=0.45 * a3, radius=3)
            cv.text(1110, y + 10, f"{v:.0f}", size=13, color=DIM, anchor="ra", alpha=a3)


def s9(cv, t, dur):
    title(cv, "⑦ 验证：3e-14 是怎么量出来的")
    a = appear(t, 0.3, 0.8)
    code(cv, 100, 180, src2("d = np.abs(g_np[k] - gt).max()", 3), size=12)
    a2 = appear(t, 2.2, 0.8)
    cv.text(100, 300, "手写梯度 vs autograd 梯度，逐张量取最大绝对差", size=17, color=DIM, alpha=a2)

    a3 = appear(t, 3.0, 0.8)
    rows = [("前向输出 u", "3.1e-15"), ("损失 L", "< 1.0e-15"),
            ("31 个参数张量", "3.0e-14"), ("中心差分（抽样）", "相对 5e-11 ~ 8e-9")]
    for i, (k, v) in enumerate(rows):
        aa = appear(t, 3.2 + 0.5 * i, 0.55)
        if aa <= 0:
            continue
        y = 360 + i * 46
        cv.text(100, y, k, size=18, color=INK, alpha=aa)
        cv.text(420, y, v, size=18, color=CY if i < 3 else AM, alpha=aa)
    a4 = appear(t, 5.6, 0.9)
    cv.text(700, 360, "写错会怎样", size=18, color=MG, alpha=a4)
    cv.text(700, 400, "符号错 → 偏差 1e-2 量级", size=16, color=DIM, alpha=a4)
    cv.text(700, 432, "转置错 → 形状直接不匹配", size=16, color=DIM, alpha=a4)
    cv.text(700, 464, "漏项 → 训练仍下降但收敛变差", size=16, color=DIM, alpha=a4)
    a5 = appear(t, 6.8, 0.8)
    cv.text(100, 600, "这 3 行数字就是“实现正确”的定义：不是能跑，而是与 autograd 逐位一致",
            size=18, color=CY, alpha=a5)


SHOTS = [
    dict(name="title", say=[
        "第五集：反向。每个雅可比怎么推出来，代码怎么写。",
        "目标很明确：31 个参数张量的梯度，和 autograd 差在一乘十的负十三以内。",
    ], draw=s1, min_dur=11.0),
    dict(name="principles", say=[
        "反向传播只有两个原则。",
        "第一，沿计算图逆序，每一步只算“局部雅可比乘上游梯度”。",
        "第二，前向时把中间量存下来，反向直接用——这就是为什么要有 cache。",
        "存什么？归一化的中间量、注意力权重、q k v、GELU 的输入、残差主干。",
    ], draw=s2, min_dur=19.0),
    dict(name="out", say=[
        "从输出层开始。",
        {"say": "损失对 u 的梯度是一；再乘 tanh 的导数 $1$ 减 tanh 平方，就得到对 r 的梯度。",
         "tts": "损失对 u 的梯度是一；再乘 tanh 的导数一减 tanh 平方，就得到对 r 的梯度。"},
        {"say": "实测 u 等于 $4.153$，导数约 $0.88$，还没有饱和。",
         "tts": "实测 u 等于四点一五三，导数约零点八八，还没有饱和。"},
        "但如果输出长期贴着正负十二，这一层梯度会消失。",
    ], draw=s3, min_dur=18.0),
    dict(name="ln", say=[
        "第二层是 LayerNorm，这是最容易推错的一个。",
        "先对均值求导、对方差求导，再合成对输入 z 的梯度。",
        {"say": "化简之后是 γ 除以 σ，乘上括号里 3 项：g 减均值，再减 z hat 乘 g z hat 的均值。",
         "tts": "化简之后是 gamma 除以 sigma，乘上括号里三项：g 减均值，再减 z hat 乘 g z hat 的均值。"},
        "后面两项正是“减整体平移”和“减整体缩放”：这两个方向不改变输出。",
    ], draw=s4, min_dur=20.0),
    dict(name="softmax", say=[
        "第 3 层是 softmax。",
        "它的雅可比是 a 乘 delta 减 a，展开到链式法则里就得到 a 乘括号 g 减 a 与 g 的内积。",
        "代码就一行：a 乘上 da 减去 da 与 a 的乘积和。",
        "这一项叫减基线：所有上游梯度相同时，梯度为 0。",
    ], draw=s5, min_dur=19.0),
    dict(name="attn", say=[
        "第 4 层是注意力本身，要算 5 个张量。",
        {"say": "对 V 是 A 转置乘上游梯度；对 A 是上游梯度乘 V 转置；再经 softmax 得对 S 的梯度。",
         "tts": "对 V 是 A 转置乘上游梯度；对 A 是上游梯度乘 V 转置；再经 softmax 得对 S 的梯度。"},
        "Q 和 K 各自由 S 的梯度乘以对方、再除以根号 dh。",
        "代码是 7 行 einsum，形状在 (B,H,T,dh) 和 (B,H,T,T) 之间来回——错一个 transpose 就全错。",
    ], draw=s6, min_dur=20.0),
    dict(name="ffn", say=[
        "第 5 层是前馈网络。这里有一个我真实踩过的坑。",
        {"say": "W2 的梯度要用残差主干 dz，不是激活值的梯度 dact。",
         "tts": "W2 的梯度要用残差主干 dz，不是激活值的梯度 dact。"},
        "写错的话形状会从 32 乘 64 变成 64 乘 64，形状检查当场就能抓到。",
        "b2 同理，也是对 dz 求和，而不是对 dact 求和。",
    ], draw=s7, min_dur=19.0),
    dict(name="residual", say=[
        "第 6 层是残差和嵌入层。",
        "残差让梯度在两处汇聚：主干直通的一份，加上分支回传的一份，两者相加。",
        {"say": "实测梯度：输出头最大，第零层只衰减到三分之一左右。",
         "tts": "实测梯度：输出头最大，第零层只衰减到三分之一左右。"},
        "位置编码的梯度要对 batch 求和，因为它被所有样本共享。",
    ], draw=s8, min_dur=18.0),
    dict(name="check", say=[
        "最后是怎么量出那个三乘十的负十四的。",
        "把手写梯度和 autograd 梯度逐张量对比，取最大绝对差。",
        "另外用中心差分做与框架无关的验证，相对误差在十的负负 11 次方到负 9 次方。",
        "如果符号写错，偏差会是 0.01量级；转置写错，形状直接不匹配。",
    ], draw=s9, min_dur=19.0),
]


# ---------- 统一风格的小结镜头（后补，与其他讲保持一致） ----------
def s_summary(cv, t, dur):
    title(cv, "本讲小结")
    for i, (k, v) in enumerate([('起点', '损失对 u 的梯度，经 tanh 的导数往回走'), ('两个雅可比', 'softmax 减基线；LN 正交于平移与缩放'), ('注意力', 'V、A、S、Q、K 五个张量，七行 einsum'), ('一个坑', 'W₂ 的梯度用 dz，不是 dact'), ('验证', '与 autograd 差 3.0e-14，形状检查兜底')]):
        aa = appear(t, 0.5 + 0.62 * i, 0.55)
        if aa <= 0:
            continue
        y = 196 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=23, color=CY, alpha=aa, weight="bold")
        cv.text(310, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.4, 0.8)
    cv.text(130, 580, '下一集：把算子串成一次完整迭代（前向→损失→反向→更新）', size=24, color=AM, alpha=a)


SHOTS.append(dict(name="summary", say=["本集小结。", "反向从输出层开始，经 tanh 的导数一路往回。", "两个关键雅可比：softmax 自带减基线，LayerNorm 正交于平移与缩放。", "注意力要算五个张量的梯度，代码是七行 einsum，形状错一个就不对。", "最容易踩的坑是 W2 的梯度：用残差主干 dz，不是激活值的梯度 dact。", "实测与 autograd 差 3 乘 10 的负 14 次方。下一集，把这一路串成一次完整迭代。"], draw=s_summary, min_dur=14.0))

"""Transformer 教程 · 第六集：串起来（一次迭代 → 训练 → 部署 → 应用）—— 10 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import math_, math_w, title, chip, axes, cartpole, curvy, code
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
TUT = os.path.join(ROOT, "tutorial", "out")
DEMO = json.load(open(os.path.join(TUT, "train_demo.json")))
SER = json.load(open(os.path.join(TUT, "attention_series.json")))
BR = json.load(open(os.path.join(ROOT, "build", "out", "chA.json")))
BW = BR["backward"]["grads"]

STAGES = [("嵌入", "6→32"), ("LN+QKV", "32→96"), ("注意力", "4×24×24"),
          ("投影+残差", "32"), ("FFN+GELU", "32→64→32"), ("头+tanh", "1")]


def pipeline(cv, t, t0, dur, forward=True, a=1.0, y=300):
    """六段流水线，一个小方块随时间沿管道移动。"""
    n = len(STAGES)
    x0, x1 = 110, 1170
    step = (x1 - x0) / n
    for i, (k, shp) in enumerate(STAGES):
        xx = x0 + i * step
        cv.rect(xx + 6, y - 34, xx + step - 18, y + 34, CY if forward else MG,
                width=1.2, alpha=0.75 * a, fill_alpha=0.07 * a, radius=8)
        cv.text(xx + step / 2 - 6, y - 8, k, size=16, color=CY if forward else MG,
                anchor="ma", alpha=a)
        cv.text(xx + step / 2 - 6, y + 16, shp, size=12, color=FAINT, anchor="ma", alpha=a)
        if i < n - 1:
            ax = xx + step - 12
            if forward:
                cv.arrow((ax, y), (ax + 16, y), DIM, width=1.4, alpha=0.7 * a)
            else:
                cv.arrow((ax + 16, y), (ax, y), DIM, width=1.4, alpha=0.7 * a)
    p = min(1.0, max(0.0, (t - t0) / max(dur, 0.01)))
    idx = (1 - p) * (n - 1) if not forward else p * (n - 1)
    mx = x0 + idx * step + step / 2 - 6
    if p > 0:
        cv.circle(mx, y - 62, 7, AM if forward else GR, width=0, alpha=a, fill_alpha=a)
        cv.text(mx, y - 84, "u" if not forward else "张量", size=14,
                color=AM if forward else GR, anchor="ma", alpha=a)
    return p


def s1(cv, t, dur):
    cv.text(640, 240, "Transformer 从零实现　·　第六集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 334, "串起来", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 394, "一次迭代 → 训练 → 部署 → 闭环应用", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 434), (810, 434), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 500, r"\theta\ \xrightarrow{\ \text{前向}\ }u"
                            r"\ \xrightarrow{\ \mathcal{L}\ }"
                            r"\frac{\partial\mathcal{L}}{\partial\theta}"
                            r"\ \xrightarrow{\ \text{AdamW}\ }\theta'", 23, DIM, anchor="mc")
    cv.text(640, 600, "9 个算子串成一条流水线，正着走一遍、反着走一遍", size=18, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "一次迭代的全景")
    a = appear(t, 0.3, 0.8)
    pipeline(cv, t, 1.2, 5.0, forward=True, a=a, y=250)
    cv.text(110, 320, "① 前向：9 个算子依次流过，全部存 cache", size=19, color=CY, alpha=a)

    a2 = appear(t, 6.4, 0.9)
    if a2 > 0:
        cv.text(1150, 320, "→ 损失 L = ½(u − u*)² → ∂L/∂u", size=17, color=AM, anchor="ra", alpha=a2)
    a3 = appear(t, 7.6, 0.9)
    if a3 > 0:
        pipeline(cv, t, 7.6, 4.0, forward=False, a=a3, y=430)
        cv.text(110, 500, "② 反向：沿同一管道逆序，每站算一个局部雅可比 × 上游梯度", size=19,
                color=MG, alpha=a3)
    a4 = appear(t, 11.8, 0.9)
    if a4 > 0:
        cv.text(110, 566, "③ 更新：AdamW 用一阶/二阶矩把梯度变成参数增量 θ ← θ − η(…)",
                size=19, color=GR, alpha=a4)
    a5 = appear(t, 13.4, 0.8)
    chip(cv, 110, 620, "这一次迭代的全部工作 = 前向一遍 + 反向一遍 + 一次参数更新", CY, a5, 18)


def s3(cv, t, dur):
    title(cv, "前向：张量沿管道流一次")
    a = appear(t, 0.3, 0.8)
    pipeline(cv, t, 0.8, 4.2, forward=True, a=a)
    rows = [("x", "(B,24,6)", DIM), ("z", "(B,24,32)", CY), ("h", "(B,24,32)", CY),
            ("q,k,v", "(B,4,24,8)", MG), ("a", "(B,4,24,24)", MG),
            ("y", "(B,24,32)", CY), ("u", "(B,)", AM)]
    for i, (k, shp, col) in enumerate(rows):
        aa = appear(t, 1.0 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        x = 110 + (i % 4) * 270
        y = 430 + (i // 4) * 66
        cv.text(x, y, k, size=19, color=col, alpha=aa)
        cv.text(x + 70, y, shp, size=15, color=FAINT, alpha=aa)
    a2 = appear(t, 5.2, 0.8)
    cv.text(110, 580, "形状每一步都可核对：唯一“变大”的是注意力矩阵 (24×24)，其余都是 d=32",
            size=18, color=DIM, alpha=a2)
    a3 = appear(t, 6.2, 0.8)
    chip(cv, 110, 626, "前向的全部中间量都存进 cache —— 反向要用", CY, a3, 18)


def s4(cv, t, dur):
    title(cv, "反向：梯度沿同一条管道逆流")
    a = appear(t, 0.3, 0.8)
    pipeline(cv, t, 0.8, 4.2, forward=False, a=a)
    rows = [("∂L/∂u", "= 1.0", AM), ("∂L/∂r", "12(1−tanh²r)", AM),
            ("∂L/∂z", "γ/σ·(g − mean·…)", MG), ("∂L/∂S", "a⊙(dA−Σ)", MG),
            ("∂L/∂Q,∂L/∂K", "dS·K/√d", MG), ("∂L/∂W_qkv", "dqkv^T·h", CY),
            ("∂L/∂W_in", "dz^T·x", CY)]
    for i, (k, v, col) in enumerate(rows):
        aa = appear(t, 1.0 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        x = 110 + (i % 4) * 270
        y = 430 + (i // 4) * 66
        cv.text(x, y, k, size=17, color=col, alpha=aa)
        cv.text(x + 130, y, v, size=14, color=FAINT, alpha=aa)
    a2 = appear(t, 5.2, 0.8)
    cv.text(110, 580, "每一站都用 cache 里的前向量：反向不需要重算前向", size=18, color=DIM, alpha=a2)
    a3 = appear(t, 6.2, 0.8)
    chip(cv, 110, 626, "这就是“反向传播”4 个字的全部工作方式", CY, a3, 18)


def s5(cv, t, dur):
    title(cv, "训练循环：3 层嵌套")
    a = appear(t, 0.3, 0.8)
    lines = [
        "def train(model, X, Y, epochs=8, bs=128, lr=2e-3):",
        "    opt = AdamW(model.p, lr=lr)                 # 一阶/二阶矩",
        "    for ep in range(epochs):                    # ① epoch",
        "        lr_ep = lr * 0.5 * (1 + cos(pi * ep / epochs))",
        "        for idx in batches(len(X), bs):         # ② mini-batch",
        "            u, cache = model.forward(X[idx])    # ③ 前向",
        "            loss, dU = mse_loss(u, Y[idx])",
        "            grads = model.backward(dU, cache)",
        "            opt.step(model.p, grads, lr=lr_ep)",
        "        val = evaluate(model, Xva, Yva)         # 每轮验证一次",
        "    return model",
    ]
    ai = None
    for i in range(len(lines)):
        if t > 1.6 + 0.65 * i:
            ai = i
    code(cv, 90, 180, lines, size=14, active=ai, alpha=a)
    a2 = appear(t, 9.0, 0.8)
    cv.text(760, 470, "3 层：epoch → batch → 单步", size=18, color=CY, alpha=a2)
    cv.text(760, 506, "真正的更新发生在最内层", size=18, color=DIM, alpha=a2)
    cv.text(760, 546, f"本课：8 epoch × 408 batch = 3,264 步", size=17, color=DIM, alpha=a2)


def s6(cv, t, dur):
    title(cv, "训练曲线：两条实现重合")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 210, 700, 460
    axes(cv, X0, Y0, X1, Y1, a, "epoch", "损失")
    y0, y1 = 1e-3, 1.0
    sy = lambda v: Y1 - (math.log10(max(v, y0)) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (Y1 - Y0)
    E = DEMO["epochs"]
    sx = lambda ep: X0 + (ep - 1) / max(1, E - 1) * (X1 - X0)
    for v in (1e-3, 1e-2, 1e-1, 1.0):
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.35 * a)
        cv.text(X0 - 8, sy(v), f"{v:g}", size=14, color=FAINT, anchor="ra", alpha=a)
    tr = [h["train"] for h in DEMO["numpy"]]
    va = [h["val"] for h in DEMO["numpy"]]
    prog = min(1.0, max(0.0, (t - 1.0) / 4.0))
    n = max(2, int(len(tr) * prog))
    curvy(cv, [(sx(i + 1), sy(tr[i])) for i in range(n)], CY, 5.0, a)
    for i in range(n - 1):
        if i % 2 == 0:
            cv.line((sx(i + 1), sy(tr[i])),
                    ((sx(i + 1) + sx(i + 2)) / 2, (sy(tr[i]) + sy(tr[i + 1])) / 2),
                    (255, 255, 255), width=2, alpha=a)
    curvy(cv, [(sx(i + 1), sy(va[i])) for i in range(n)], AM, 2.0, 0.9 * a)
    for i in range(n):
        cv.circle(sx(i + 1), sy(va[i]), 3.2, AM, width=1.2, alpha=0.9 * a)

    a2 = appear(t, 4.6, 0.8)
    cv.text(730, 230, "粗青 = numpy 手写反向", size=17, color=CY, alpha=a2)
    cv.text(730, 262, "白划 = torch autograd", size=17, color=INK, alpha=a2)
    cv.text(730, 294, "橙圈 = 验证", size=17, color=AM, alpha=a2)
    a3 = appear(t, 5.4, 0.8)
    cv.text(730, 356, f"耗时 numpy {DEMO['t_numpy']:.0f}s", size=18, color=DIM, alpha=a3)
    cv.text(730, 390, f"torch {DEMO['t_torch']:.0f}s → {DEMO['speedup']:.1f}×", size=18,
            color=DIM, alpha=a3)
    a4 = appear(t, 6.2, 0.8)
    cv.text(730, 450, f"最终验证 {DEMO['val_numpy']:.2e}", size=18, color=GR, alpha=a4)
    cv.text(730, 484, "两条曲线差 1e-7（float32 舍入）", size=15, color=FAINT, alpha=a4)


def s7(cv, t, dur):
    title(cv, "参数更新：AdamW 一步")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad "
                        r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2", 18, INK)
    a2 = appear(t, 1.8, 0.8)
    math_(cv, 100, 240, r"\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}"
                        r"+\lambda\theta_{t-1}\Big)", 19, CY)
    a3 = appear(t, 3.0, 0.8)
    cv.text(100, 300, "一阶矩 = 动量；二阶矩 = 每个参数自己的步长；λ 是解耦权重衰减", size=17,
            color=DIM, alpha=a3)
    a4 = appear(t, 3.8, 0.8)
    code(cv, 100, 356, [
        "for k, g in grads.items():",
        "    m[k] = b1 * m[k] + (1 - b1) * g",
        "    v[k] = b2 * v[k] + (1 - b2) * g ** 2",
        "    mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)",
        "    p[k] -= lr * (mh / (sqrt(vh) + eps) + wd * p[k])",
    ], size=14, active=4 if t > 6.0 else None, alpha=a4)
    a5 = appear(t, 6.6, 0.8)
    cv.text(760, 380, "不衰减的例外", size=18, color=AM, alpha=a5)
    cv.text(760, 416, "LayerNorm 的 γ、β 与位置编码", size=16, color=DIM, alpha=a5)
    cv.text(760, 448, "它们不是“权重”，衰减会破坏归一化", size=16, color=DIM, alpha=a5)


def s8(cv, t, dur):
    title(cv, "部署：从 float32 到 int8")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 190, r"s=\frac{\max|W|}{127},\quad q=\mathrm{round}(W/s),\quad\hat W=q\,s,"
                        r"\quad |W-\hat W|\le\frac{\max|W|}{254}", 18, INK)
    a2 = appear(t, 2.0, 0.8)
    code(cv, 100, 250, [
        "def quantize(W):                      # 逐张量 int8",
        "    s = np.abs(W).max() / 127.0",
        "    q = np.clip(np.round(W / s), -127, 127).astype(np.int8)",
        "    return q, np.float32(s)",
        "",
        "def dequant(q, s): return q.astype(np.float32) * s",
    ], size=14, alpha=a2)
    a3 = appear(t, 4.2, 0.8)
    cards = [("71 KB → 16.6 KB", "体积", 4.2, AM), ("0.14 N", "单点最大误差", 5.0, CY),
             ("0.32 ms", "浏览器单步", 5.8, GR)]
    for txt, sub, t0, col in cards:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 200 + (t0 - 4.2) * 76
        cv.text(760, y, txt, size=24, color=col, weight="bold", alpha=aa)
        cv.text(760, y + 26, sub, size=16, color=FAINT, alpha=aa)
    a4 = appear(t, 6.6, 0.8)
    cv.text(760, 470, "量化后闭环行为不变", size=18, color=CY, alpha=a4)
    cv.text(760, 502, "这是整套流水线的最后一步", size=16, color=DIM, alpha=a4)


def s9(cv, t, dur):
    title(cv, "应用：把它装进控制回路")
    a = appear(t, 0.3, 0.8)
    # 控制回路：传感器 → 窗口 → 模型 → 力 → 对象
    boxes = [("传感器", "θ, θ̇, p, v", 110), ("窗口 24×6", "归一化", 330),
             ("模型", "9 个算子", 550), ("力 u", "±12 N", 770), ("小车", "被控对象", 990)]
    for i, (k, s2_, x) in enumerate(boxes):
        aa = appear(t, 0.5 + 0.5 * i, 0.5)
        cv.rect(x, 210, x + 170, 274, CY if i != 2 else AM, width=1.2, alpha=0.8 * aa,
                fill_alpha=0.07 * aa, radius=8)
        cv.text(x + 85, 238, k, size=17, color=CY if i != 2 else AM, anchor="ma", alpha=aa)
        cv.text(x + 85, 260, s2_, size=12, color=FAINT, anchor="ma", alpha=aa)
        if i < 4:
            cv.arrow((x + 176, 242), (x + 210, 242), DIM, width=1.6, alpha=0.7 * aa)
    a2 = appear(t, 3.2, 0.8)
    cv.arrow((1075, 286), (1075, 340), DIM, width=1.6, alpha=0.8 * a2)
    cv.arrow((1075, 340), (195, 340), DIM, width=1.6, alpha=0.8 * a2)
    cv.arrow((195, 340), (195, 286), CY, width=1.8, alpha=0.8 * a2)
    cv.text(620, 366, "闭环反馈：每 20 ms 走一次", size=17, color=CY, anchor="ma", alpha=a2)

    a3 = appear(t, 4.4, 0.8)
    cartpole(cv, 330, 560, SER["th"][min(int(t * 12), len(SER["th"]) - 1)],
             p=SER["p"][min(int(t * 12), len(SER["p"]) - 1)],
             u=SER["u"][min(int(t * 12), len(SER["u"]) - 1)], a=0.9 * a3, scale=118, rail=2.4)
    a4 = appear(t, 5.6, 0.8)
    cv.text(700, 500, "同一个 18,177 参数的模型", size=18, color=INK, alpha=a4)
    cv.text(700, 534, "在浏览器里以 50 Hz 实时推理", size=17, color=DIM, alpha=a4)
    cv.text(700, 578, "30/30 闭环成功，从下垂摆起 1.9 s", size=18, color=GR, alpha=a4)


def s10(cv, t, dur):
    title(cv, "串联全链路")
    items = [("① 算子", "9 个，各自独立可测"),
             ("② 一次迭代", "前向 → 损失 → 反向 → 更新"),
             ("③ 训练", "epoch → batch → step，3,264 步"),
             ("④ 验证", "两实现曲线重合，差 1e-7"),
             ("⑤ 部署", "int8：16.6 KB，0.32 ms/步"),
             ("⑥ 应用", "闭环 50 Hz，30/30 成功")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 180 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=21, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=20, color=INK, alpha=aa)
    a = appear(t, 4.2, 0.8)
    cv.text(130, 590, "从一台倒立摆，到一个能实时跑的控制器 —— 全部在这 18,177 个参数里", size=22,
            color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "前两集把算子拆开了。这一集把它们串起来。",
        "从一次迭代，到训练，到部署，到真正装进控制回路。",
    ], draw=s1, min_dur=11.0),
    dict(name="overview", say=[
        "先看一次迭代的全景。",
        "9 个算子从前到后走一遍，得到一个力；和标签一比就是损失。",
        "然后沿同一条流水线逆序走一遍，得到每个参数的梯度。",
        "最后 AdamW 把梯度变成参数增量。这就是一次完整的迭代。",
    ], draw=s2, min_dur=20.0),
    dict(name="forward", say=[
        "前向的时候，张量沿管道流一次。",
        {"say": "输入是 $B$ 乘 $24$ 乘 $6$；嵌入后变成 $24$ 乘 $32$；分头后是 $4$ 个头、每头 $8$ 维。",
         "tts": "输入是 B 乘 24 乘 6；嵌入后变成 24 乘 32；分头后是四个头、每头八维。"},
        "唯一“变大”的是注意力矩阵，二十四乘二十四。",
        "所有中间量都存进 cache，因为反向他要用。",
    ], draw=s3, min_dur=18.0),
    dict(name="backward", say=[
        "反向的时候，梯度沿同一条管道逆流。",
        "每个站点都用 cache 里的前向量，算一个局部雅可比，乘上上游梯度。",
        "所以你不需要重算前向——这也是反向传播能高效的原因。",
    ], draw=s4, min_dur=17.0),
    dict(name="loop", say=[
        "训练循环是 3 层嵌套。",
        "最外层是 epoch，中间是 mini-batch，最内层才是一次前向加反向加更新。",
        {"say": "本课是 8 个 epoch 乘 408 个批量，一共 3,264 步。",
         "tts": "本课是八个 epoch 乘四百零八批量，一共三千二百六十四步。"},
        "每轮结束在验证集上评估一次。",
    ], draw=s5, min_dur=18.0),
    dict(name="curve", say=[
        "训练曲线是这样。",
        {"say": "粗青线是 numpy 手写反向，白色虚线是 torch，两条完全重合，差在十的负 7 次方。",
         "tts": "粗青线是 numpy 手写反向，白色虚线是 torch，两条完全重合，差在十的负七次方。"},
        {"say": "代价是速度：numpy 慢 4 倍。",
         "tts": "代价是速度：numpy 慢四倍。"},
    ], draw=s6, min_dur=16.0),
    dict(name="adamw", say=[
        "参数更新用的是 AdamW。",
        "一阶矩是动量，二阶矩给每个参数自己的步长，最后再加一个解耦的权重衰减。",
        "注意有一类参数不衰减：LayerNorm 的 gamma、beta 和位置编码——它们不是权重。",
    ], draw=s7, min_dur=17.0),
    dict(name="deploy", say=[
        "训练完要部署。",
        {"say": "逐张量量化到 int8：体积从七十一 KB 降到十 6.6 KB。",
         "tts": "逐张量量化到 int8：体积从七十一 KB 降到十六点六 KB。"},
        {"say": "单点误差不超过量程除以 $254$，实测 $0.14$ 牛顿，闭环看不出差别。",
         "tts": "单点误差不超过量程除以二百五十四，实测零点一四牛顿，闭环看不出差别。"},
        "浏览器里单步 0.32毫秒。",
    ], draw=s8, min_dur=17.0),
    dict(name="apply", say=[
        "最后是应用：把它装进控制回路。",
        "传感器读 4 个状态，拼成 24 步窗口，模型输出一个力，力作用到小车上，再回到传感器。",
        "每二十毫秒走一次，也就是五十赫兹。",
        {"say": "最终 30 组初值全部成功，从下垂摆起只要 $1.9$ 秒。",
         "tts": "最终三十组初值全部成功，从下垂摆起只要一点九秒。"},
    ], draw=s9, min_dur=19.0),
    dict(name="summary", say=[
        "整条链路就是这样：算子、迭代、训练、验证、部署、应用。",
        "从一台倒立摆到一个能实时跑的控制器，全部信息都在这 18,177 个参数里。",
    ], draw=s10, min_dur=14.0),
]

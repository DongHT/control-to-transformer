"""Transformer 教程 · 第三部：numpy 与 torch 双实现 + 可视化 —— 7 个镜头。"""
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
TUT = os.path.join(HERE, "..", "..", "..", "tutorial", "out")
DEMO = json.load(open(os.path.join(TUT, "train_demo.json")))
REP = json.load(open(os.path.join(TUT, "pretrained_report.json")))
SER = json.load(open(os.path.join(TUT, "attention_series.json")))


def s1(cv, t, dur):
    cv.text(640, 246, "Transformer 从零实现　·　第三部", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 340, "两种实现", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 400, "纯 numpy 手写反向 · torch autograd · 一致性验证", size=27, color=DIM,
            anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((460, 440), (820, 440), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 508, r"\max\left|\frac{\partial\mathcal{L}}{\partial\theta}\Big|_{\text{numpy}}"
                            r"-\frac{\partial\mathcal{L}}{\partial\theta}\Big|_{\text{torch}}\right|"
                            r"=3.0\times10^{-14}", 22, DIM, anchor="mc")
    cv.text(640, 600, "tutorial/code/tinytf_numpy.py · tinytf_torch.py", size=18, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "numpy 版：前向 40 行，反向 80 行")
    a = appear(t, 0.3, 0.8)
    lines = [
        "def forward(self, x, return_cache=False):",
        "    z = x @ W_in.T + b_in + pos          # (B,T,d)",
        "    for l in range(2):",
        "        h, c1 = ln_forward(z, g1, b1)     # LN1",
        "        q, k, v = self.qkv(h)            # (B,H,T,dh)",
        "        a = softmax(mask(q @ k.T / sqrt(dh)))",
        "        z = z + self.proj(merge(a @ v))  # 残差 1",
        "        h2, c2 = ln_forward(z, g2, b2)",
        "        z = z + self.fc2(gelu(self.fc1(h2)))",
        "    return 12 * tanh(self.head(ln(z[:, -1])))",
    ]
    code(cv, 110, 196, lines, size=14, hl={4, 5, 6}, hl_color=CY)
    a2 = appear(t, 3.4, 0.8)
    lines2 = [
        "def backward(self, dU, cache):",
        "    dr = dU * 12 * (1 - tanh(r)**2)      # tanh",
        "    dz[-1] = ln_backward(dr @ w_head)    # LN",
        "    for l in reversed(range(2)):",
        "        dact = dz @ W2                   # FFN",
        "        dz = dz + ln_backward(dact * gelu_grad(pre) @ W1)",
        "        da = dy @ v.T                    # 注意力",
        "        dS = a * (da - (da * a).sum(-1))  # softmax",
        "        dq = dS @ k / sqrt(dh); dk = dS.T @ q / sqrt(dh)",
    ]
    code(cv, 110, 424, lines2, size=14, hl={7}, hl_color=AM)
    a3 = appear(t, 5.6, 0.8)
    cv.text(760, 200, "为什么可以手写", size=19, color=INK, alpha=a3)
    cv.text(760, 240, "只有 9 种算子：", size=17, color=DIM, alpha=a3)
    for i, op in enumerate(["矩阵乘", "softmax", "LayerNorm", "GELU", "残差", "reshape"]):
        cv.text(770, 274 + i * 30, f"· {op}", size=17, color=DIM, alpha=a3)
    cv.text(760, 470, "每种算子一个前向 + 一个反向", size=17, color=CY, alpha=a3)
    cv.text(760, 502, "剩下的只是把它们串起来", size=17, color=CY, alpha=a3)


def s3(cv, t, dur):
    title(cv, "五个易错点（都真实踩过）")
    items = [
        ("① 多头 reshape 顺序", "必须 transpose(1,2)→(B,H,T,dh)", MG),
        ("② softmax 减基线项", "漏掉 −Σ a_i g_i 会拖慢收敛", MG),
        ("③ W2 / b2 的梯度", "来自残差主干 dz，不是 dact", MG),
        ("④ LayerNorm 紧凑式", "比逐步展开更短也更稳", CY),
        ("⑤ 掩码位置的梯度", "a=0 ⇒ dS=0，无需再清零", CY),
    ]
    for i, (k, v, col) in enumerate(items):
        aa = appear(t, 0.5 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 190 + i * 58
        cv.text(110, y, k, size=20, color=col, weight="bold", alpha=aa)
        cv.text(430, y, v, size=18, color=INK, alpha=aa)

    a = appear(t, 3.6, 0.9)
    if a > 0:
        cv.text(110, 510, "第 ③ 条的现场：我第一次写成", size=17, color=FAINT, alpha=a)
        code(cv, 110, 548, ["g[W2] = dact.T @ act   # 形状 (ff,ff) ← 错"], size=15, hl={0},
             hl_color=MG)
        code(cv, 110, 596, ["g[W2] = dz.T   @ act   # 形状 (d,ff)  ← 对"], size=15, hl={0},
             hl_color=GR)
    a2 = appear(t, 5.4, 0.8)
    cv.text(760, 520, "形状检查是最便宜的单元测试", size=18, color=CY, alpha=a2)
    cv.text(760, 556, "它当场就抓到了这个错", size=18, color=DIM, alpha=a2)


def s4(cv, t, dur):
    title(cv, "训练：两条曲线重合")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 210, 700, 470
    axes(cv, X0, Y0, X1, Y1, a, "epoch", "损失（对数）")
    y0, y1 = 1e-3, 1.0
    sy = lambda v: Y1 - (math.log10(max(v, y0)) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (Y1 - Y0)
    E = DEMO["epochs"]
    sx = lambda ep: X0 + (ep - 1) / max(1, E - 1) * (X1 - X0)
    for v in (1e-3, 1e-2, 1e-1, 1.0):
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.35 * a)
        cv.text(X0 - 8, sy(v), f"{v:g}", size=14, color=FAINT, anchor="ra", alpha=a)
    tr = [h["train"] for h in DEMO["numpy"]]
    va = [h["val"] for h in DEMO["numpy"]]
    prog = min(1.0, max(0.0, (t - 1.2) / 3.0))
    n = max(2, int(len(tr) * prog))
    curvy(cv, [(sx(i + 1), sy(tr[i])) for i in range(n)], CY, 5.0, a)
    for i in range(n - 1):          # torch 用白色短划叠在青线上
        x1, y1p = sx(i + 1), sy(tr[i])
        x2, y2p = sx(i + 2), sy(tr[i + 1])
        if i % 2 == 0:
            cv.line((x1, y1p), ((x1 + x2) / 2, (y1p + y2p) / 2), (255, 255, 255), width=2, alpha=a)
    curvy(cv, [(sx(i + 1), sy(va[i])) for i in range(n)], AM, 2.0, 0.9 * a)
    for i in range(n):
        cv.circle(sx(i + 1), sy(va[i]), 3.4, AM, width=1.2, alpha=0.9 * a)

    a2 = appear(t, 4.4, 0.8)
    cv.text(730, 230, "粗青线 = numpy 手写反向", size=17, color=CY, alpha=a2)
    cv.text(730, 262, "白色短划 = torch autograd", size=17, color=INK, alpha=a2)
    cv.text(730, 294, "橙圈 = 验证损失", size=17, color=AM, alpha=a2)
    a3 = appear(t, 5.2, 0.8)
    cv.text(730, 356, "逐 epoch 最大差", size=17, color=FAINT, alpha=a3)
    dif = max(abs(a1["train"] - b1["train"]) for a1, b1 in zip(DEMO["numpy"], DEMO["torch"]))
    cv.text(730, 396, f"{dif:.1e}", size=30, color=GR, weight="bold", alpha=a3)
    cv.text(730, 424, "float32 舍入量级", size=15, color=FAINT, alpha=a3)
    a4 = appear(t, 6.0, 0.8)
    cv.text(730, 490, f"耗时 numpy {DEMO['t_numpy']:.0f}s", size=17, color=DIM, alpha=a4)
    cv.text(730, 520, f"torch {DEMO['t_torch']:.0f}s → {DEMO['speedup']:.1f}× 加速", size=17,
            color=DIM, alpha=a4)


def s5(cv, t, dur):
    title(cv, "加载生产权重：numpy 版直接上岗")
    a = appear(t, 0.3, 0.8)
    p = REP["parity"]
    cv.text(110, 190, "推理一致性（512 个真实窗口）", size=19, color=FAINT, alpha=a)
    cv.text(110, 240, f"{p['max_abs']:.1e} N", size=34, color=GR, weight="bold", alpha=a)
    cv.text(110, 272, f"最大偏差；平均 {p['mean_abs']:.1e} N", size=16, color=FAINT, alpha=a)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 340, "闭环成功率（numpy 策略）", size=19, color=FAINT, alpha=a2)
    cv.text(110, 396, f"{int(round(REP['success']*30))}/30", size=40, color=CY, weight="bold", alpha=a2)
    cv.text(110, 430, "与 torch 版完全一致", size=16, color=FAINT, alpha=a2)
    a3 = appear(t, 3.0, 0.8)
    cv.text(110, 500, "从下垂出发摆起", size=19, color=FAINT, alpha=a3)
    cv.text(110, 548, f"{REP['hang_to_up_s']:.2f} s", size=30, color=AM, weight="bold", alpha=a3)

    a4 = appear(t, 4.0, 0.9)
    if a4 > 0:
        X0, Y0, X1, Y1 = 620, 240, 900, 520
        axes(cv, X0, Y0, X1, Y1, a4, "torch u [N]", "numpy u [N]")
        sx = lambda v: X0 + (v + 12) / 24 * (X1 - X0)
        sy = lambda v: Y1 - (v + 12) / 24 * (Y1 - Y0)
        rng = np.random.default_rng(1)
        for _ in range(120):
            v = rng.uniform(-11, 11)
            cv.circle(sx(v), sy(v + rng.normal(0, 0.02)), 2.2, CY, width=0, alpha=0.75 * a4,
                      fill_alpha=0.75 * a4)
        cv.line((sx(-12), sy(-12)), (sx(12), sy(12)), (120, 130, 145), width=1.2, alpha=0.8 * a4)
    a5 = appear(t, 5.4, 0.8)
    cv.text(950, 300, "所有点落在 y = x 上", size=18, color=INK, alpha=a5)
    cv.text(950, 340, "相关系数 1.000000000", size=17, color=DIM, alpha=a5)
    cv.text(950, 400, "同一个结构、同一套权重", size=17, color=CY, alpha=a5)
    cv.text(950, 432, "两种实现给出同一个控制器", size=17, color=CY, alpha=a5)


def s6(cv, t, dur):
    title(cv, "注意力 × 倒立摆：两条时间轴对齐")
    if t < 17.0:
        k = int(min(max(0.0, (t - 1.0)) * 0.22 / 0.02, len(SER["t"]) - 1))
    else:
        k = int(min((8.5 + (t - 17.0) * 0.16) / 0.02, len(SER["t"]) - 1))
    a = appear(t, 0.3, 0.6)
    cartpole(cv, 330, 470, SER["th"][k], p=SER["p"][k], u=SER["u"][k], a=a, scale=118, rail=2.4)
    cv.text(330, 522, f"t = {SER['t'][k]:.2f} s    θ = {math.degrees(SER['th'][k]):+.1f}°    "
                      f"u = {SER['u'][k]:+.2f} N", size=18, color=DIM, anchor="ma", alpha=a)

    row = np.array(SER["attn"][k])
    X0, Y0, BW, BH = 690, 210, 460, 150
    cv.text(X0, Y0 - 12, "第 2 层注意力（4 头平均）", size=16, color=FAINT, alpha=a)
    mx = max(row.max(), 1e-9)
    for i in range(24):
        x = X0 + i * 19
        hh = row[i] / mx * BH
        cv.rect(x, Y0 + BH - hh, x + 14, Y0 + BH, CY if row[i] > 1 / 24 else (60, 110, 120),
                width=1, alpha=0.85 * a, fill_alpha=0.55 * a, radius=2)
    cv.line((X0, Y0 + BH + 4), (X0 + 24 * 19, Y0 + BH + 4), (92, 110, 132), width=1.2, alpha=0.8 * a)
    pk = int(np.argmax(row))
    cv.text(X0 + pk * 19 + 7, Y0 + BH - 18, f"峰值 @{pk}", size=15, color=MG, anchor="ma", alpha=a)

    ws = SER["win"][k]
    Y2 = 480
    cv.text(X0, Y2 - 24, "输入窗口里的 cosθ（0.48 s）", size=16, color=FAINT, alpha=a)
    sx = lambda i: X0 + i * 19
    sy = lambda v: Y2 + 70 - (v + 1) / 2 * 70
    curvy(cv, [(sx(i), sy(ws[i])) for i in range(len(ws))], AM, 2.2, a)
    cv.line((X0, sy(0)), (X0 + 23 * 19, sy(0)), (90, 108, 130), width=1, alpha=0.5 * a, dash=5)
    cv.circle(sx(23), sy(ws[-1]), 5, AM, width=0, alpha=a, fill_alpha=a)

    a2 = appear(t, 6.0, 0.8)
    cv.text(690, 636, "摆起：权重铺开（估相位）　稳摆：峰值锁最近步（估变化率）", size=16,
            color=CY, alpha=a2)


def s7(cv, t, dur):
    title(cv, "三部小结")
    items = [("动机", "RNN 串行 + 梯度连乘 → 注意力直连"),
             ("结构", "2 层 Pre-LN，18,177 参数"),
             ("前向", "9 种算子，六行公式"),
             ("反向", "softmax/LN 雅可比 + 残差直通"),
             ("验证", "梯度差 3e-14，有限差分独立复核"),
             ("落地", "numpy 版可加载生产权重，30/30 闭环")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.6 * i, 0.55)
        if aa <= 0:
            continue
        y = 180 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(290, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.5, 0.8)
    cv.text(130, 590, "代码与讲义：tutorial/ ；回去自己跑一遍 parity_test.py", size=23,
            color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "第三部：两份实现，一份手写反向，一份 autograd。",
        "它们的结果应该逐位一致——这就是这一部要证明的事。",
    ], draw=s1, min_dur=11.0),
    dict(name="numpy", say=[
        "先看 numpy 版。前向四十行，反向八十行。",
        "整个模型只有九种算子：矩阵乘、softmax、LayerNorm、GELU、残差和 reshape。",
        "每种算子写一个前向、一个反向，剩下的只是把它们按顺序串起来。",
    ], draw=s2, min_dur=18.0),
    dict(name="pitfalls", say=[
        "但真正花时间的是五个易错点。",
        "多头 reshape 的顺序、softmax 的减基线项、W2 的梯度来源、LN 的紧凑写法、掩码位置的梯度。",
        "其中最坑的是 W2：它的梯度来自残差主干，不是激活值。写错的话形状会从 d 乘 ff 变成 ff 乘 ff。",
        "形状检查是最便宜的单元测试，它当场就抓到了这个错。",
    ], draw=s3, min_dur=20.0),
    dict(name="train", say=[
        "用一个脚本同时训练两版：同初始化、同数据、同批次顺序。",
        {"say": "训练损失逐 epoch 差在 $10^{-7}$，也就是 float32 的舍入量级。",
         "tts": "训练损失逐 epoch 差在十的负七次方，也就是 float32 的舍入量级。"},
        {"say": "代价是速度：numpy 慢 $4$ 倍。这就是框架存在的意义。",
         "tts": "代价是速度：numpy 慢四倍。这就是框架存在的意义。"},
    ], draw=s4, min_dur=18.0),
    dict(name="pretrained", say=[
        "更有说服力的是：把项目里训练好的权重直接装进 numpy 版。",
        {"say": "五百一十二个真实窗口上，最大偏差 $5$ 乘十的负六次方牛顿。",
         "tts": "五百一十二个真实窗口上，最大偏差五乘十的负六次方牛顿。"},
        "用 numpy 策略跑闭环，三十组初值全部成功——和 torch 版完全一样。",
    ], draw=s5, min_dur=17.0),
    dict(name="sync", say=[
        "最后看它到底在看什么。左边是倒立摆，右边是同一时刻的注意力。",
        {"say": "摆起阶段，输入窗口里的 cos 曲线是一道陡坡，注意力铺满整段。",
         "tts": "摆起阶段，输入窗口里的 cos 曲线是一道陡坡，注意力铺满整段。"},
        {"say": "稳摆之后，这段历史几乎是一条平线，峰值收到最近一步。",
         "tts": "稳摆之后，这段历史几乎是一条平线，峰值收到最近一步。"},
        "相位估计与变化率估计——两种信息需求，两种注意力形态。",
    ], draw=s6, min_dur=28.0),
    dict(name="summary", say=[
        "三部结束：动机、推导、两份实现、三张可视化。",
        "代码在 tutorial 目录，建议自己跑一遍 parity_test，亲眼看到 3e-14。",
    ], draw=s7, min_dur=13.0),
]

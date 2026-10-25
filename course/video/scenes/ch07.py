"""第 7 讲：数据与模仿（原理版）—— 7 个镜头。"""
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
D = json.load(open(os.path.join(HERE, "..", "..", "..", "build", "out", "ch07_10.json")))["drift"]
DT = np.array(D["t"]); BC = np.array(D["bc0"]); DG = np.array(D["dagger"]); EX = np.array(D["expert_noise"])


def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer", size=25, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 7 讲　数据与模仿", size=64, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把控制器变成函数逼近问题", size=31, color=DIM, anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"\pi_\theta:\ \mathcal{H}_H\to\mathcal{A},\qquad "
                            r"\mathcal{H}_H=\mathbb{R}^{H\times 6}", 22, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch07　　代码 · dataset.py / train.py", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(1.0 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


def s2(cv, t, dur):
    title(cv, "为什么是监督学习，又为什么不是")
    a = appear(t, 0.3, 0.8)
    cv.rect(96, 180, 620, 300, CY, width=1.2, alpha=0.75 * a, fill_alpha=0.05 * a, radius=8)
    cv.text(120, 212, "√ 可以说通的理由", size=21, color=CY, alpha=a)
    cv.text(120, 254, "确定性 MDP 的最优策略是状态的函数：", size=19, color=INK, alpha=a)
    math_(cv, 120, 292, r"\pi^\star(s)=\arg\max_a Q^\star(s,a)", 20, INK)
    a1 = appear(t, 1.8, 0.8)
    cv.rect(96, 330, 1184, 520, MG, width=1.2, alpha=0.75 * a1, fill_alpha=0.05 * a1, radius=8)
    cv.text(120, 362, "× 但它不是普通监督学习：3 个坑", size=21, color=MG, alpha=a1)
    rows = ["① 样本强相关：相邻两帧只差 20 ms",
            "② 训练分布 d_π* ≠ 测试分布 d_πθ",
            "③ 专家有记忆：同一状态，两种模式给不同动作"]
    for i, s in enumerate(rows):
        cv.text(140, 404 + i * 38, s, size=19, color=INK, alpha=appear(t, 2.4 + 0.4 * i, 0.5))
    a2 = appear(t, 4.2, 0.8)
    chip(cv, 110, 566, "结论：输入必须是“一段历史”，不是单帧状态", AM, a2, 20)


def s3(cv, t, dur):
    title(cv, "同一状态，两种历史，两个动作")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 210, 560, 520
    axes(cv, X0, Y0, X1, Y1, a, "θ [rad]", "θ̇ [rad/s]")
    sx = lambda v: X0 + (v + 0.6) / 1.2 * (X1 - X0)
    sy = lambda v: Y1 - (v + 3.0) / 6.0 * (Y1 - Y0)
    for v in (-3, 0, 3):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a)
    for v in (-0.4, 0, 0.4):
        cv.line((sx(v), Y1), (sx(v), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(sx(v), Y1 + 26, f"{v:g}", size=15, color=FAINT, anchor="ma", alpha=a)
    # two paths arriving at the same point (0.1, 0)
    p1 = appear(t, 1.0, 0.8)
    path_a = [(sx(0.5), sy(2.2)), (sx(0.34), sy(1.5)), (sx(0.22), sy(0.8)), (sx(0.14), sy(0.35)),
              (sx(0.1), sy(0.0))]
    curvy(cv, path_a, AM, 2.4, p1)
    path_b = [(sx(-0.35), sy(-1.9)), (sx(-0.15), sy(-1.2)), (sx(0.02), sy(-0.6)),
              (sx(0.08), sy(-0.2)), (sx(0.1), sy(0.0))]
    p2 = appear(t, 1.8, 0.8)
    curvy(cv, path_b, BL, 2.4, p2)
    cv.circle(sx(0.1), sy(0.0), 6, INK, width=0, alpha=a, fill_alpha=a)
    cv.text(sx(0.1) + 14, sy(0.0) + 22, "(0.10, 0)", size=17, color=INK, alpha=a)

    a3 = appear(t, 3.0, 0.8)
    cv.text(610, 240, "同一个点 (θ, θ̇) = (0.10, 0)", size=20, color=INK, alpha=a3)
    cv.text(610, 290, "从上面来（摆起模式）", size=19, color=AM, alpha=a3)
    math_(cv, 610, 330, r"u\approx-6.2\ \mathrm{N}", 20, AM)
    cv.text(610, 390, "从下面来（稳摆模式）", size=19, color=BL, alpha=a3)
    math_(cv, 610, 430, r"u\approx-1.4\ \mathrm{N}", 20, BL)
    a4 = appear(t, 4.6, 0.8)
    cv.text(610, 500, "π* 不是状态的函数，而是历史的函数", size=20, color=MG, alpha=a4)


def s4(cv, t, dur):
    title(cv, "监督目标：我们最小化的其实是条件均值")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 210, r"\hat\theta=\arg\min_\theta\ "
                        r"\mathbb{E}_{s\sim d_{\pi^\star}}\|\pi_\theta(s)-\pi^\star(s)\|^2", 22, INK)
    a2 = appear(t, 1.6, 0.8)
    cv.text(110, 260, "平方损失的最优解 = 条件均值", size=19, color=FAINT, alpha=a2)
    math_(cv, 110, 300, r"\pi_\theta(s)=\mathbb{E}[a\mid s]", 22, CY)

    # bimodal sketch
    a3 = appear(t, 3.0, 0.9)
    if a3 > 0:
        X0, Y0, X1, Y1 = 640, 280, 1140, 470
        axes(cv, X0, Y0, X1, Y1, a3, "动作 a", "密度")
        sx = lambda v: X0 + (v + 2.2) / 4.4 * (X1 - X0)
        sy = lambda v: Y1 - v * (Y1 - Y0)
        for c, h in ((-1.2, 0.85), (1.2, 0.85)):
            pts = [(sx(c + k / 50 * 1.1), sy(h * math.exp(-(k / 50 * 1.1) ** 2 / 0.12)))
                   for k in range(-50, 51)]
            curvy(cv, pts, CY, 2.0, 0.9 * a3)
        cv.line((sx(0), Y1), (sx(0), Y0 + 20), MG, width=2, alpha=0.9 * a3, dash=6)
        cv.text(sx(0), Y0 + 10, "均值 → 夹在两峰之间", size=17, color=MG, anchor="ma", alpha=a3)
        cv.text(sx(-1.2), Y1 - 8, "该泵能量", size=16, color=FAINT, anchor="ma", alpha=a3)
        cv.text(sx(1.2), Y1 - 8, "该刹车", size=16, color=FAINT, anchor="ma", alpha=a3)
    a4 = appear(t, 5.0, 0.8)
    cv.text(110, 400, "本课的处理：标签取未加噪的专家动作，噪声只作用于执行", size=19, color=DIM,
            alpha=a4)
    cv.text(110, 436, "→ 条件分布保持单峰，均值解才是安全的", size=19, color=DIM, alpha=a4)


def s5(cv, t, dur):
    title(cv, "分布漂移：误差为什么会自己长大")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 210, r"\delta x_{t+1}\approx J\,\delta x_t\ \Rightarrow\ "
                        r"\|\delta x_t\|\lesssim\epsilon\,\frac{e^{Lt}-1}{L}", 22, INK)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 262, "本课 L ≈ 4.0（第 1 讲的正特征根），时间常数 0.25 s", size=19, color=DIM, alpha=a2)
    # growth curve
    if a2 > 0:
        X0, Y0, X1, Y1 = 110, 300, 620, 520
        axes(cv, X0, Y0, X1, Y1, a2, "t [s]", "放大倍数")
        sx = lambda v: X0 + v / 2.0 * (X1 - X0)
        sy = lambda v: Y1 - min(v, 30) / 30.0 * (Y1 - Y0)
        for v in (0, 10, 20, 30):
            cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a2)
            cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a2)
        curvy(cv, [(sx(u), sy((math.exp(4.0 * u) - 1))) for u in np.linspace(0, 2, 200)],
              MG, 2.2, a2)
        cv.text(sx(1.2), sy(26), "e^(4t) − 1", size=18, color=MG, alpha=a2)

    a3 = appear(t, 3.4, 0.8)
    math_(cv, 700, 340, r"J_{\mathrm{BC}}\le J^\star+O(\epsilon T^2)", 21, AM)
    math_(cv, 700, 392, r"J_{\mathrm{DAgger}}\le J^\star+O(\epsilon T)", 21, CY)
    a4 = appear(t, 4.6, 0.8)
    cv.text(700, 456, "错误一旦发生，之后每步都在", size=18, color=DIM, alpha=a4)
    cv.text(700, 486, "更陌生的状态上决策 → T²", size=18, color=DIM, alpha=a4)


def s6(cv, t, dur):
    title(cv, "本用例的 3 条曲线")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 220, 780, 520
    axes(cv, X0, Y0, X1, Y1, a, "t [s]", "与专家的角度偏差 [rad]")
    sx = lambda v: X0 + v / 8.0 * (X1 - X0)
    sy = lambda v: Y1 - math.log10(max(v, 1e-4) / 1e-4) / math.log10(3.0 / 1e-4) * (Y1 - Y0)
    for v in (1e-3, 1e-2, 1e-1, 1.0):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a)
        lab = f"{v:g}" if v >= 1e-2 else "0.001"
        cv.text(X0 + 9, sy(v) + 6, lab, size=14, color=FAINT, alpha=a)
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.3 * a, dash=6)
    for v in (0, 2, 4, 6, 8):
        cv.line((sx(v), Y1), (sx(v), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(sx(v), Y1 + 26, str(v), size=15, color=FAINT, anchor="ma", alpha=a)
    prog = min(1.0, max(0.0, (t - 1.0) / 3.5))
    n = max(2, int(len(DT) * prog))
    curvy(cv, [(sx(DT[i]), sy(min(BC[i], 2.9))) for i in range(n)], MG, 2.4, a)
    curvy(cv, [(sx(DT[i]), sy(min(EX[i], 2.9))) for i in range(n)], AM, 2.0, 0.9 * a)
    curvy(cv, [(sx(DT[i]), sy(max(DG[i], 1e-4))) for i in range(n)], CY, 2.4, a)
    for i, (lab, col, x0) in enumerate([("纯 BC", MG, X1 - 210),
                                       ("专家 + 0.45 N 噪声", AM, X1 - 210),
                                       ("DAgger", CY, X1 - 210)]):
        yy = Y0 + 22 + i * 30
        cv.line((x0, yy - 6), (x0 + 26, yy - 6), col, width=2.6, alpha=a)
        cv.text(x0 + 34, yy, lab, size=17, color=col, alpha=a)
    a2 = appear(t, 4.6, 0.8)
    cv.text(820, 250, "0.10 s", size=34, color=MG, weight="bold", alpha=a2)
    cv.text(820, 280, "纯 BC 越过 0.15 rad 的时刻", size=17, color=FAINT, alpha=a2)
    a3 = appear(t, 5.4, 0.8)
    cv.text(820, 360, "0.0016 rad", size=30, color=CY, weight="bold", alpha=a3)
    cv.text(820, 388, "DAgger 末偏差（与专家同步）", size=17, color=FAINT, alpha=a3)
    a4 = appear(t, 6.2, 0.8)
    cv.text(820, 466, "同样的 0.45 N 误差，", size=18, color=DIM, alpha=a4)
    cv.text(820, 496, "每步被纠正就不会跑偏", size=18, color=DIM, alpha=a4)


def s7(cv, t, dur):
    title(cv, "本讲小结")
    items = [("监督前提", "确定性策略 = 状态的函数"),
             ("但需记忆", "专家有模式 → 输入必须是历史"),
             ("窗口", "H=24（0.48 s）≈ 1/3 摆动周期"),
             ("BC 目标", "在专家分布上最小化，测试分布却不是"),
             ("漂移定量", "ε(e^{Lt}−1)/L，L≈4；BC O(T²)、DAgger O(T)"),
             ("本用例", "BC 0.10 s 跑偏；DAgger 0.0016 rad")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.62 * i, 0.55)
        if aa <= 0:
            continue
        y = 186 + i * 58
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.8, 0.8)
    cv.text(130, 580, "下一讲：这个“历史 → 动作”的函数，用什么结构去逼近", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "前六讲里，控制律是我们自己写的。从这一讲开始，把它变成一个函数逼近问题。",
        "但先说清楚：这到底是不是一个普通的监督学习问题。",
    ], draw=s1, min_dur=11.0),
    dict(name="why", say=[
        "可以说通的理由很直接：确定性系统里，最优策略就是状态的函数。",
        "但它不是普通监督学习，有 3 个坑。",
        "样本强相关；训练分布是专家的、测试分布是策略自己的；还有第 3 条——专家有记忆。",
    ], draw=s2, min_dur=18.0),
    dict(name="memory", say=[
        "看这条相平面上的同一个点：角度 0.1，角速度 0。",
        "从上面来的轨迹处在摆起模式，它给的力是−6.2 牛顿。",
        "从下面来的处在稳摆模式，同样的状态只给−1.4 牛顿。",
        "所以策略不是状态的函数，而是历史的函数——这是必须用窗口的原因。",
    ], draw=s3, min_dur=19.0),
    dict(name="objective", say=[
        "再看我们到底在最小化什么。",
        "平方损失的最优解是条件均值。",
        "如果同一状态下专家动作是双峰的（一个要泵能量、一个要刹车），均值会落在两峰之间——那是错的动作。",
        "本课的处理是：标签取未加噪的专家动作，噪声只作用于执行，这样条件分布保持单峰。",
    ], draw=s4, min_dur=20.0),
    dict(name="drift", say=[
        "现在解释最重要的现象：为什么拟合得很好，闭环却会跑偏。",
        {"say": "闭环误差满足一个指数放大：$\epsilon$ 乘上 $(e^{Lt}-1)/L$，本课的 $L$ 约等于 $4$。",
         "tts": "闭环误差满足一个指数放大：epsilon 乘上 e 的 L t 次方减一，再除以 L；本课的 L 约等于四。"},
        "于是理论上行为克隆（BC，Behavior Cloning）的代价按 T 平方涨，而 DAgger（Dataset Aggregation，数据集聚合）只按 T 涨。",
    ], draw=s5, min_dur=19.0),
    dict(name="curves", say=[
        "用本用例的 3 条曲线验证。",
        {"say": "纯 BC 在 $0.10$ 秒就越过 $0.15$ 弧度，8 秒末偏差 $0.652$。",
         "tts": "纯 BC 在零点一秒就越过零点一五弧度，八秒末偏差零点六五二。"},
        {"say": "DAgger 版一路贴着专家，末偏差 $0.0016$ 弧度。",
         "tts": "DAgger 版一路贴着专家，末偏差零点零零一六弧度。"},
        "作为对照，同样 0.45 牛顿的误差，如果每步被专家纠正，系统并不会跑偏。",
    ], draw=s6, min_dur=20.0),
    dict(name="summary", say=[
        "总结：监督前提成立，但分布不成立；窗口解决记忆，DAgger 解决分布。",
        "下一讲讲用什么结构去逼近这个函数。",
    ], draw=s7, min_dur=13.0),
]

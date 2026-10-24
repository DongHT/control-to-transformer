"""第 9 讲：DAgger 与评测（原理版）—— 7 个镜头。"""
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
    cv.text(640, 352, "第 9 讲　DAgger 与评测", size=60, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "把 T² 降成 T，并给出可验收的统计", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"O(\epsilon T^2)\ \longrightarrow\ O(\epsilon T)", 26, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch09　　代码 · train.py / eval_all.py", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(1.1 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


def s2(cv, t, dur):
    title(cv, "把分布漂移写成公式")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 206, r"\epsilon_{\mathrm{BC}}=\mathbb{E}_{s\sim d_{\pi^\star}}"
                        r"\|\pi_\theta(s)-\pi^\star(s)\|", 21, CY)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 258, r"\epsilon_{\mathrm{test}}=\mathbb{E}_{s\sim d_{\pi_\theta}}"
                        r"\|\pi_\theta(s)-\pi^\star(s)\|", 21, MG)
    a3 = appear(t, 2.8, 0.8)
    cv.text(110, 302, "下标不同，这就是全部问题", size=20, color=DIM, alpha=a3)

    a4 = appear(t, 4.0, 0.9)
    if a4 > 0:
        X0, Y0, X1, Y1 = 110, 360, 700, 560
        axes(cv, X0, Y0, X1, Y1, a4, "步数 t", "累计代价")
        sx = lambda v: X0 + v / 600 * (X1 - X0)
        sy = lambda v: Y1 - min(v, 1.05) / 1.05 * (Y1 - Y0)
        curvy(cv, [(sx(t2), sy((t2 / 600) ** 2)) for t2 in range(0, 601, 4)], MG, 2.4, a4)
        curvy(cv, [(sx(t2), sy(0.55 * t2 / 600)) for t2 in range(0, 601, 4)], CY, 2.4, a4)
        cv.text(sx(430), sy(0.92), "BC: εT²", size=19, color=MG, alpha=a4)
        cv.text(sx(430), sy(0.30), "DAgger: εT", size=19, color=CY, alpha=a4)
    a5 = appear(t, 5.6, 0.8)
    math_(cv, 760, 400, r"T=600\ \text{步},\ \epsilon\approx0.038", 20, INK)
    cv.text(760, 450, "→ 理论量级 T² ≈ 3.6×10⁵", size=19, color=DIM, alpha=a5)
    cv.text(760, 486, "与实测 0/30 一致", size=19, color=MG, alpha=a5)


def s3(cv, t, dur):
    title(cv, "DAgger：把“训练集外”变成“训练集内”")
    a = appear(t, 0.3, 0.8)
    steps = [("①", "当前策略闭环跑", "记录它实际到达的 ŝ"),
             ("②", "专家在 ŝ 上标注", "a* = π*(ŝ)"),
             ("③", "并入数据集", "D ← D ∪ {(ŝ, a*)}"),
             ("④", "重新训练", "重复 N 轮")]
    for i, (n, s1_, s2_) in enumerate(steps):
        aa = appear(t, 0.5 + 0.6 * i, 0.55)
        if aa <= 0:
            continue
        y = 210 + i * 88
        cv.circle(130, y - 8, 16, CY, width=1.2, alpha=0.8 * aa)
        cv.text(130, y, n, size=20, color=CY, anchor="ma", alpha=aa)
        cv.text(180, y, s1_, size=21, color=INK, alpha=aa)
        cv.text(480, y, s2_, size=18, color=FAINT, alpha=aa)
    a2 = appear(t, 3.4, 0.8)
    cv.rect(660, 200, 1180, 420, CY, width=1.2, alpha=0.75 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(682, 232, "关键区别", size=20, color=CY, alpha=a2)
    cv.text(682, 272, "BC：状态分布来自专家", size=18, color=INK, alpha=a2)
    cv.text(682, 306, "DAgger：状态分布来自策略自己", size=18, color=INK, alpha=a2)
    math_(cv, 682, 366, r"d_{\pi_\theta^{(i)}}\to d_{\pi_\theta}\ \Rightarrow\ "
                        r"\epsilon_{\mathrm{test}}\to\epsilon_{\mathrm{BC}}", 18, DIM)
    a3 = appear(t, 5.0, 0.8)
    cv.text(660, 480, "代价：专家必须能对任意状态标注", size=19, color=AM, alpha=a3)
    cv.text(660, 512, "（本课满足：专家是解析控制器）", size=18, color=FAINT, alpha=a3)


def s4(cv, t, dur):
    title(cv, "为什么能把 T² 降成 T")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 210, r"\mathbb{E}_{s\sim d_\pi}[\ell(s,\pi)]\ \le\ \epsilon_N"
                        r"+O\!\left(\frac{1}{N}\right)", 22, INK)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 262, "no-regret 归约：N 轮平均策略的代价只差 ε_N + O(1/N)", size=19, color=DIM, alpha=a2)
    a3 = appear(t, 3.0, 0.9)
    # table of stages
    rows = [("BC", "--", "1.45e−3", "0/30", MG),
            ("DAgger 1", "26,460", "3.44e−3", "30/30", CY),
            ("DAgger 2", "26,460", "3.38e−3", "30/30", GR)]
    cv.text(110, 330, "阶段", size=17, color=FAINT, alpha=a3)
    cv.text(330, 330, "新增窗口", size=17, color=FAINT, alpha=a3)
    cv.text(560, 330, "验证损失", size=17, color=FAINT, alpha=a3)
    cv.text(800, 330, "闭环成功", size=17, color=FAINT, alpha=a3)
    for i, (k, w, v, s, col) in enumerate(rows):
        aa = appear(t, 3.2 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 376 + i * 50
        cv.text(110, y, k, size=20, color=col, alpha=aa)
        cv.text(330, y, w, size=19, color=INK, alpha=aa)
        cv.text(560, y, v, size=19, color=INK, alpha=aa)
        cv.text(800, y, s, size=20, color=col, alpha=aa)
    a4 = appear(t, 5.4, 0.8)
    cv.text(110, 560, "损失上升不是退步：验证集还是旧分布，而策略的分布变宽了", size=19,
            color=AM, alpha=a4)


def s5(cv, t, dur):
    title(cv, "评测的统计学：30/30 到底是多少")
    a = appear(t, 0.3, 0.8)
    bars = [("30/30", 1.00, 0.886, 1.000, GR),
            ("24/30", 0.80, 0.627, 0.905, AM),
            ("0/30", 0.00, 0.000, 0.114, MG)]
    cv.text(110, 200, "95% Wilson 区间", size=18, color=FAINT, alpha=a)
    for i, (lab, p, lo, hi, col) in enumerate(bars):
        aa = appear(t, 0.6 + 0.55 * i, 0.55)
        if aa <= 0:
            continue
        y = 260 + i * 92
        x0, x1 = 300, 900
        sx = lambda v: x0 + v * (x1 - x0)
        cv.line((x0, y), (x1, y), (70, 86, 104), width=1, alpha=0.7 * aa)
        cv.rect(sx(lo), y - 14, sx(hi), y + 14, col, width=1.2, alpha=0.85 * aa,
                fill_alpha=0.22 * aa, radius=4)
        cv.circle(sx(p), y, 6, col, width=0, alpha=aa, fill_alpha=aa)
        cv.text(110, y, lab, size=21, color=col, alpha=aa)
        cv.text(sx(hi) + 16, y, f"[{lo:.3f}, {hi:.3f}]", size=18, color=DIM, alpha=aa)
    a2 = appear(t, 3.4, 0.8)
    cv.text(110, 560, "30/30 与 24/30 的区间有重叠 → n=30 不能宣称“显著更好”", size=19,
            color=AM, alpha=a2)
    a3 = appear(t, 4.6, 0.8)
    cv.text(110, 604, "区分 80% 与 100% 约需每组 22 个样本；本课报告 60 组", size=19,
            color=DIM, alpha=a3)


def s6(cv, t, dur):
    title(cv, "三个偏差：别让数字骗自己")
    a = appear(t, 0.3, 0.8)
    items = [("① 检查点选择", "用测试集挑检查点 = 偷看答案", MG),
             ("② 初值分布", "整圈均匀 vs 集中在直立，成功率差一倍", AM),
             ("③ 指标口径", "失败记 99 s 还是只统计成功回合", BL)]
    for i, (k, v, col) in enumerate(items):
        aa = appear(t, 0.6 + 0.7 * i, 0.55)
        if aa <= 0:
            continue
        y = 220 + i * 96
        cv.text(110, y, k, size=23, color=col, weight="bold", alpha=aa)
        cv.text(110, y + 34, v, size=19, color=INK, alpha=aa)
    a2 = appear(t, 3.6, 0.8)
    cv.rect(110, 520, 1180, 600, CY, width=1.2, alpha=0.7 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(134, 552, "本课做法：24 组验证初值挑检查点，60 组测试初值报告成绩", size=20,
            color=CY, alpha=a2)


def s7(cv, t, dur):
    title(cv, "本讲小结")
    items = [("形式化", "ε_BC 在 d_π* 上，ε_test 在 d_πθ 上"),
             ("累积界", "BC O(εT²)，DAgger O(εT)"),
             ("修法", "用策略自己的状态 + 专家标注"),
             ("本用例", "两轮各 26,460 窗口，0/30 → 30/30"),
             ("统计学", "30/30 的 95% 区间是 [0.886, 1.000]"),
             ("三个偏差", "检查点、初值分布、指标口径")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.6 * i, 0.55)
        if aa <= 0:
            continue
        y = 186 + i * 58
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=21, color=INK, alpha=aa)
    a = appear(t, 4.6, 0.8)
    cv.text(130, 580, "下一讲：换掉目标函数，用策略梯度直接优化闭环表现", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲末尾我们算过：行为克隆的误差按 T 平方累积。",
        "这一讲讲 DAgger 为什么能把它降到 T，以及怎么给出能被验收的统计。",
    ], draw=s1, min_dur=11.0),
    dict(name="formula", say=[
        "先把问题写成公式。",
        {"say": "行为克隆最小化的是专家状态分布上的误差；但闭环里承受的是策略自己分布上的误差。",
         "tts": "行为克隆最小化的是专家状态分布上的误差；但闭环里承受的是策略自己分布上的误差。"},
        {"say": "把误差累积求和，就得到 T 平方的代价上界；本课 T 是六百步，量级就是十万。",
         "tts": "把误差累积求和，就得到 T 平方的代价上界；本课 T 是六百步，量级就是十万。"},
    ], draw=s2, min_dur=18.0),
    dict(name="dagger", say=[
        "DAgger 的做法是把“训练集外”变成“训练集内”。",
        "用当前策略跑闭环，记录它实际到达的状态；让专家在这些状态上标注动作；并入数据集重训。",
        "和 BC 的唯一区别是状态分布来自策略自己，于是测试分布和训练分布逐步重合。",
        "代价是：专家必须能对任意状态标注，而不只是演示。",
    ], draw=s3, min_dur=19.0),
    dict(name="noregret", say=[
        "理论上这是一个 no-regret 归约。",
        "N 轮平均策略的代价，只比最优差一个 ε_N 加一阶小项。",
        {"say": "本课的实测：两轮、各两万六千个窗口，成功率从零到百分之百。",
         "tts": "本课的实测：两轮、各两万六千个窗口，成功率从零到百分之百。"},
        "注意验证损失反而上升——因为验证集还是旧分布，而策略的分布变宽了。",
    ], draw=s4, min_dur=18.0),
    dict(name="stats", say=[
        "数字怎么报？成功率必须给区间。",
        {"say": "三十次全成功的百分之九十五 Wilson 区间是 $0.886$ 到 $1$。",
         "tts": "三十次全成功的百分之九十五 Wilson 区间是零点八八六到一。"},
        "它和 24/30 的区间有重叠，所以在三十个样本下不能宣称显著更好。",
    ], draw=s5, min_dur=18.0),
    dict(name="bias", say=[
        "还有三个偏差必须防。",
        "检查点选择：用测试集挑检查点等于偷看答案。",
        "初值分布与指标口径：不写清这两条，数字之间不可比。",
    ], draw=s6, min_dur=16.0),
    dict(name="summary", say=[
        "总结：误差累积有定量界，DAgger 把二次降成线性，评测要给区间。",
        "下一讲换掉目标函数，直接优化闭环表现。",
    ], draw=s7, min_dur=12.0),
]

"""Transformer/学习教程 · 第八集：DAgger 从原理到实现 —— 9 个镜头。"""
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
BUILD = os.path.join(ROOT, "build")
TL = json.load(open(os.path.join(BUILD, "out", "train_log.json")))
DRIFT = json.load(open(os.path.join(BUILD, "out", "ch07_10.json")))["drift"]

_LINES = open(os.path.join(BUILD, "train.py"), encoding="utf-8").read().split("\n")


def src(anchor, n, dedent=None):
    for i, ln in enumerate(_LINES):
        if anchor in ln:
            block = _LINES[i:i + n]
            if dedent is None:
                dedent = len(block[0]) - len(block[0].lstrip())
            return [b[dedent:] if len(b) >= dedent else b for b in block]
    return [f"# anchor not found: {anchor}"]


def s1(cv, t, dur):
    cv.text(640, 246, "学习控制 · 从原理到实现　·　第八集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 340, "DAgger：让专家标注策略自己的状态", size=52, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 402, "从 O(εT²) 到 O(εT)：原理、算法、代码", size=28, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 440), (810, 440), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 506, r"d_{\pi^\star}\ \longrightarrow\ d_{\pi_\theta}", 30, DIM, anchor="mc")
    cv.text(640, 600, "Dataset Aggregation　数据集聚合", size=18, color=FAINT, anchor="ma",
            alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "问题：BC 为什么会越走越偏")
    a = appear(t, 0.3, 0.8)
    X0, Y0, X1, Y1 = 110, 200, 720, 470
    axes(cv, X0, Y0, X1, Y1, a, "t [s]", "与专家的角度偏差 [rad]（对数）")
    lo, hi = 1e-4, 3.0
    sy = lambda v: Y1 - (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / \
        (math.log10(hi) - math.log10(lo)) * (Y1 - Y0)
    sx = lambda v: X0 + v / 8.0 * (X1 - X0)
    for v in (1e-3, 1e-2, 0.1, 1):
        cv.line((X0, sy(v)), (X1, sy(v)), (78, 96, 116), width=1, alpha=0.3 * a)
        cv.text(X0 - 8, sy(v), f"{v:g}", size=13, color=FAINT, anchor="ra", alpha=a)
    T = np.array(DRIFT["t"])
    n = len(T)
    curvy(cv, [(sx(T[i]), sy(max(DRIFT["bc0"][i], lo))) for i in range(n)], MG, 2.2, a)
    curvy(cv, [(sx(T[i]), sy(max(DRIFT["dagger"][i], lo))) for i in range(n)], CY, 2.2, a)
    cv.text(sx(4.4), sy(0.6), "纯 BC", size=17, color=MG, alpha=a)
    cv.text(sx(4.4), sy(0.002), "BC + DAgger", size=17, color=CY, alpha=a)

    a2 = appear(t, 3.2, 0.8)
    cv.text(760, 220, "为什么", size=19, color=INK, alpha=a2)
    cv.text(760, 258, "训练分布 = 专家走过的状态", size=16, color=DIM, alpha=a2)
    cv.text(760, 288, "测试分布 = 策略自己到达的状态", size=16, color=DIM, alpha=a2)
    a3 = appear(t, 4.4, 0.8)
    cv.text(760, 340, "一旦偏离，就进入“没学过”的区域", size=16, color=MG, alpha=a3)
    cv.text(760, 370, "误差按 T² 累积", size=16, color=MG, alpha=a3)
    a4 = appear(t, 5.6, 0.8)
    chip(cv, 760, 430, "纯 BC：0.10 s 就超出 0.15 rad", MG, a4, 17)


def s3(cv, t, dur):
    title(cv, "原理：两个分布，两个界")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"\epsilon_{\mathrm{BC}}=\mathbb{E}_{s\sim d_{\pi^\star}}"
                        r"\big\|\pi_\theta(s)-\pi^\star(s)\big\|", 19, CY)
    a2 = appear(t, 1.4, 0.8)
    math_(cv, 100, 240, r"\epsilon_{\mathrm{test}}=\mathbb{E}_{s\sim d_{\pi_\theta}}"
                        r"\big\|\pi_\theta(s)-\pi^\star(s)\big\|", 19, MG)
    a3 = appear(t, 2.6, 0.8)
    cv.text(100, 292, "下标不同，就是全部问题", size=17, color=DIM, alpha=a3)
    a4 = appear(t, 3.6, 0.8)
    math_(cv, 100, 348, r"J_{\mathrm{BC}}\le J^\star+O(\epsilon T^2),\qquad "
                        r"J_{\mathrm{DAgger}}\le J^\star+O(\epsilon T)", 19, INK)
    a5 = appear(t, 5.0, 0.8)
    cv.text(100, 404, "no-regret 归约：训练分布逐步逼近测试分布，二次项消失", size=16,
            color=DIM, alpha=a5)
    a6 = appear(t, 6.0, 0.9)
    cv.rect(640, 180, 1180, 420, CY, width=1.2, alpha=0.7 * a6, fill_alpha=0.05 * a6, radius=8)
    cv.text(662, 212, "一句话记住", size=18, color=CY, alpha=a6)
    cv.text(662, 252, "BC：在“专家去过的”状态上最小化误差", size=16, color=INK, alpha=a6)
    cv.text(662, 284, "DAgger：在“我会去的”状态上最小化误差", size=16, color=INK, alpha=a6)
    cv.text(662, 330, "后者的状态集，正是前者漏掉的那部分", size=16, color=AM, alpha=a6)


def s4(cv, t, dur):
    title(cv, "算法：四步循环")
    steps = [("① 用当前策略跑闭环", "允许探索噪声，记录实际到达的状态"),
             ("② 请专家在这些状态上标注", "a* = π*(ŝ)，专家是解析控制器"),
             ("③ 并入数据集", "D ← D ∪ {(ŝ, a*)}"),
             ("④ 重新训练", "回到 ①，重复 N 轮")]
    for i, (k, v) in enumerate(steps):
        aa = appear(t, 0.4 + 0.7 * i, 0.55)
        if aa <= 0:
            continue
        y = 200 + i * 92
        cv.circle(130, y - 8, 16, CY, width=1.2, alpha=0.8 * aa)
        cv.text(130, y, k[0], size=20, color=CY, anchor="ma", alpha=aa)
        cv.text(180, y, k[2:], size=21, color=INK, alpha=aa)
        cv.text(180, y + 30, v, size=16, color=FAINT, alpha=aa)

    a2 = appear(t, 3.6, 0.9)
    if a2 > 0:
        cv.rect(640, 200, 1180, 330, AM, width=1.2, alpha=0.7 * a2, fill_alpha=0.05 * a2, radius=8)
        cv.text(662, 232, "可选加固：β 混合", size=18, color=AM, alpha=a2)
        math_(cv, 662, 276, r"a=\beta\,a^\star+(1-\beta)\,\pi_\theta(s),\quad \beta:1\to0", 18, INK)
        cv.text(662, 310, "前几轮多一点专家动作，保证数据质量", size=15, color=DIM, alpha=a2)
    a3 = appear(t, 5.2, 0.8)
    cv.text(640, 380, "关键成本：专家必须能对“任意状态”标注", size=18, color=MG, alpha=a3)
    cv.text(640, 412, "人做专家时，这才是 DAgger 的主要开销", size=16, color=DIM, alpha=a3)


def s5(cv, t, dur):
    title(cv, "实现 ①：让策略去“陌生的地方”")
    code(cv, 90, 180, src("def rollout_policy(ctl, x0, T=12.0, noise=0.6, rng=None):", 12),
         size=13, hl={6, 7}, hl_color=CY)
    a = appear(t, 3.0, 0.8)
    cv.text(90, 470, "两个细节决定成败", size=18, color=INK, alpha=a)
    a2 = appear(t, 3.6, 0.8)
    cv.text(110, 508, "· 探索噪声 noise=0.6 N：不加噪声，策略永远走同一条轨迹", size=16,
            color=DIM, alpha=a2)
    cv.text(110, 540, "· 撞墙钳位：位置钳位 + 速度清零，与真实导轨一致", size=16, color=DIM, alpha=a2)
    a3 = appear(t, 4.8, 0.8)
    chip(cv, 90, 600, "这一步产生的是“策略的状态”，不是专家的", CY, a3, 18)


def s6(cv, t, dur):
    title(cv, "实现 ②：专家在策略的状态上标注")
    code(cv, 90, 180, src("def relabel(X, expert):", 3), size=15, hl={2}, hl_color=CY)
    a = appear(t, 2.6, 0.8)
    cv.text(90, 300, "四行代码，是 DAgger 的全部“魔法”", size=18, color=INK, alpha=a)
    a2 = appear(t, 3.4, 0.9)
    if a2 > 0:
        # 示意：策略轨迹与专家在同样状态上的动作
        cv.text(90, 356, "同一批状态 ŝ，专家给出的动作与策略不同", size=16, color=DIM, alpha=a2)
        X0, Y0 = 120, 400
        for i in range(24):
            x = X0 + i * 26
            hh = 20 + 26 * math.exp(-((i - 9) ** 2) / 30.0)
            cv.rect(x, Y0 + 60 - hh, x + 18, Y0 + 60, CY, width=1, alpha=0.75 * a2,
                    fill_alpha=0.4 * a2, radius=2)
            hh2 = 20 + 26 * math.exp(-((i - 13) ** 2) / 40.0)
            cv.rect(x, Y0 + 130 - hh2, x + 18, Y0 + 130, AM, width=1, alpha=0.75 * a2,
                    fill_alpha=0.4 * a2, radius=2)
        cv.text(X0, Y0 + 20, "策略在 ŝ 上的动作", size=14, color=CY, alpha=a2)
        cv.text(X0, Y0 + 90, "专家在 ŝ 上的动作 a*（标签）", size=14, color=AM, alpha=a2)
    a3 = appear(t, 5.4, 0.8)
    cv.text(760, 400, "注意：不是重放专家的轨迹，", size=16, color=INK, alpha=a3)
    cv.text(760, 430, "而是把专家“请到”策略到达的状态上", size=16, color=INK, alpha=a3)
    cv.text(760, 470, "这就是分布 d 的迁移", size=16, color=CY, alpha=a3)


def s7(cv, t, dur):
    title(cv, "实现 ③：主循环与每轮数据量")
    code(cv, 90, 178, src("for rnd in range(2):", 16), size=12, hl={3, 9}, hl_color=CY)
    a = appear(t, 3.2, 0.8)
    rounds = TL["dagger"]
    rows = [("BC", "0/30", "—"),
            ("DAgger 第 1 轮", "30/30", f"{rounds[0]['t_up']:.2f} s"),
            ("DAgger 第 2 轮", "30/30", f"{rounds[1]['t_up']:.2f} s")]
    cv.text(760, 200, "训练结果", size=18, color=FAINT, alpha=a)
    for i, (k, s_, t_) in enumerate(rows):
        aa = appear(t, 3.4 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 240 + i * 52
        cv.text(760, y, k, size=17, color=INK, alpha=aa)
        cv.text(980, y, s_, size=17, color=(MG if i == 0 else GR), alpha=aa)
        cv.text(1090, y, t_, size=17, color=DIM, alpha=aa)
    a2 = appear(t, 5.2, 0.8)
    cv.text(760, 420, "每轮新增 26,460 个窗口", size=16, color=DIM, alpha=a2)
    cv.text(760, 452, "两轮、约 10 分钟 CPU", size=16, color=DIM, alpha=a2)
    a3 = appear(t, 6.0, 0.8)
    chip(cv, 760, 510, "成功率 0/30 → 30/30", GR, a3, 18)


def s8(cv, t, dur):
    title(cv, "一个反直觉的现象")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 200, "验证损失：变差了", size=19, color=FAINT, alpha=a)
    a2 = appear(t, 1.0, 0.8)
    cv.text(110, 246, "1.45e-3", size=34, color=MG, weight="bold", alpha=a2)
    cv.text(230, 246, "→", size=26, color=DIM, alpha=a2)
    cv.text(290, 246, "3.38e-3", size=34, color=AM, weight="bold", alpha=a2)
    a3 = appear(t, 2.2, 0.8)
    cv.text(110, 290, "（涨了 2.3 倍）", size=16, color=DIM, alpha=a3)

    a4 = appear(t, 3.0, 0.8)
    cv.text(110, 360, "闭环成功率：从 0 到 100%", size=19, color=FAINT, alpha=a4)
    a5 = appear(t, 3.6, 0.8)
    cv.text(110, 416, "0/30", size=36, color=MG, weight="bold", alpha=a5)
    cv.text(240, 416, "→", size=26, color=DIM, alpha=a5)
    cv.text(300, 416, "30/30", size=36, color=GR, weight="bold", alpha=a5)

    a6 = appear(t, 4.8, 0.9)
    cv.rect(620, 190, 1180, 430, CY, width=1.2, alpha=0.7 * a6, fill_alpha=0.05 * a6, radius=8)
    cv.text(642, 222, "为什么损失会“变差”", size=18, color=CY, alpha=a6)
    cv.text(642, 262, "验证集还是旧分布（专家状态）", size=16, color=INK, alpha=a6)
    cv.text(642, 294, "而新策略的分布更宽、更难拟合", size=16, color=INK, alpha=a6)
    cv.text(642, 336, "所以离线指标不再是能力的度量", size=16, color=AM, alpha=a6)
    a7 = appear(t, 6.4, 0.8)
    chip(cv, 620, 480, "验收指标只能是闭环指标", MG, a7, 18)


def s9(cv, t, dur):
    title(cv, "本集小结")
    items = [("问题", "BC 的测试分布 ≠ 训练分布"),
             ("原理", "no-regret 归约：O(εT²) → O(εT)"),
             ("算法", "四步循环：跑 → 标注 → 并入 → 重训"),
             ("实现", "rollout_policy + relabel + 重训，约 40 行"),
             ("本用例", "两轮把 0/30 提到 30/30"),
             ("教训", "损失变差也可能是进步，要看闭环")]
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
    a = appear(t, 4.4, 0.8)
    cv.text(130, 580, "下一集：不再模仿，直接用回报优化策略（PPO）", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "这一集讲 DAgger，从原理到实现。",
        "它是模仿学习里最实用的一个修正：让专家去标注“策略自己走出来的状态”。",
        "全称 Dataset Aggregation，中文叫数据集聚合。",
    ], draw=s1, min_dur=12.0),
    dict(name="problem", say=[
        "先看它要解决什么问题。",
        {"say": "这张图是两条轨迹与专家的角度偏差：纯 BC 在 $0.10$ 秒就越过 $0.15$ 弧度，DAgger 版一路贴着专家。",
         "tts": "这张图是两条轨迹与专家的角度偏差：纯 BC 在零点一秒就越过零点一五弧度，DAgger 版一路贴着专家。"},
        "原因很简单：训练分布是专家走过的状态，测试分布是策略自己到达的状态。",
        "一旦偏离，就进入了没学过的区域，误差按 T 平方累积。",
    ], draw=s2, min_dur=18.0),
    dict(name="theory", say=[
        "原理可以写成两个期望。",
        {"say": "行为克隆最小化的是专家分布上的误差，而闭环里承受的是策略分布上的误差。",
         "tts": "行为克隆最小化的是专家分布上的误差，而闭环里承受的是策略分布上的误差。"},
        "把误差沿时间求和，就得到 T 平方与 T 两个界。",
        "DAgger 之所以能去掉二次项，是因为它让训练分布逐步逼近测试分布。",
    ], draw=s3, min_dur=18.0),
    dict(name="algo", say=[
        "算法只有四步，循环执行。",
        "用当前策略跑闭环、记录它实际到达的状态；请专家在这些状态上标注动作；并入数据集；重新训练。",
        {"say": "可选加固是 β 混合：前几轮让专家动作占一点比例，保证早期数据质量。",
         "tts": "可选加固是 beta 混合：前几轮让专家动作占一点比例，保证早期数据质量。"},
        "唯一的成本是：专家必须能对任意状态标注。人做专家时，这就是主要开销。",
    ], draw=s4, min_dur=18.0),
    dict(name="impl1", say=[
        "实现第一步，让策略去陌生的地方。",
        "这段代码跑一次闭环，但动作上加了探索噪声。",
        "不加噪声的话，策略每次都走同一条轨迹，数据集永远只有那一条线。",
        "撞墙时按真实导轨做位置钳位、速度清零，保证数据与实物一致。",
    ], draw=s5, min_dur=17.0),
    dict(name="impl2", say=[
        "第二步，请专家在策略到达的状态上标注。",
        "代码只有四行：把记录下来的状态序列喂给专家，逐个取动作。",
        "注意这不是重放专家的轨迹，而是把专家请到策略到达的状态上去。",
        "这一步就是分布 d 的迁移，也是 DAgger 与行为克隆的唯一区别。",
    ], draw=s6, min_dur=17.0),
    dict(name="impl3", say=[
        "第三步是主循环：每轮收集数据、并入、重训。",
        {"say": "本课每轮新增两万六千个窗口，两轮、约十分钟 CPU。",
         "tts": "本课每轮新增两万六千个窗口，两轮、约十分钟 CPU。"},
        "结果：成功率从零到百分之百，摆起时间 2.56 秒和 2.68 秒。",
    ], draw=s7, min_dur=16.0),
    dict(name="paradox", say=[
        "这里有个反直觉的现象，必须讲清楚。",
        {"say": "验证损失从 1.45e-3 涨到 3.38e-3，涨了 2.3 倍。",
         "tts": "验证损失从一点四五乘十的负三次方涨到三点三八乘十的负三次方，涨了二点三倍。"},
        "但闭环成功率从零变成百分之百。",
        "原因是验证集还是旧分布，而新策略的分布更宽、更难拟合。",
        "所以离线指标不能当验收标准，只能看闭环。",
    ], draw=s8, min_dur=19.0),
    dict(name="summary", say=[
        "小结：问题、原理、算法、实现、结果、教训。",
        "下一集，我们不再模仿专家，而是直接用回报去优化策略。",
    ], draw=s9, min_dur=13.0),
]

"""第 10 讲：策略梯度 / PPO / DR（原理版）—— 8 个镜头。"""
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
    cv.text(640, 352, "第 10 讲　策略梯度与 PPO", size=56, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "从 MDP 到域随机化", size=31, color=DIM, anchor="ma", alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 512, r"\nabla_\theta J=\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a|s)\,A(s,a)\big]",
              22, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch10　　代码 · rl.py / export_rl.py", size=19, color=FAINT,
            alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.05 * math.sin(1.3 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


def s2(cv, t, dur):
    title(cv, "策略梯度：从 MDP 到可算的梯度")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 186, r"J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\Big[\sum_t\gamma^t r_t\Big]", 21, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 274, r"\nabla_\theta J=\mathbb{E}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,"
                        r"\big(R_t-\gamma V(s_{t+1})\big)\Big]", 20, CY)
    a3 = appear(t, 3.0, 0.8)
    cv.text(110, 338, "对数似然技巧：只有 π 依赖 θ，所以 p_θ 的梯度变成 log 的梯度", size=19,
            color=DIM, alpha=a3)
    a4 = appear(t, 4.2, 0.8)
    cv.text(110, 382, "本课的参数化", size=19, color=FAINT, alpha=a4)
    math_(cv, 110, 424, r"a\sim\mathcal{N}\big(\mu_\theta(o_t),\ \sigma^2\big),\quad "
                        r"\mu_\theta=\text{Transformer}_\theta(o_t)", 20, INK)
    a5 = appear(t, 5.4, 0.8)
    rows = [("从 BC 权重热启动", "策略梯度是局部搜索，从“已经会控制”出发"),
            ("σ ≈ 0.55 N", "太小无探索，太大打散已学会的策略"),
            ("评估取均值 μ", "确定性部署")]
    for i, (k, v) in enumerate(rows):
        aa = appear(t, 5.6 + 0.5 * i, 0.5)
        cv.text(110, 486 + i * 40, f"{k}：{v}", size=18, color=DIM, alpha=aa)


def s3(cv, t, dur):
    title(cv, "PPO：clip 就是软信任域")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 196, r"r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}",
          21, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 110, 262, r"L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_tA_t,\ "
                        r"\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\big]", 19, CY)
    a3 = appear(t, 3.0, 0.9)
    if a3 > 0:
        X0, Y0, X1, Y1 = 110, 340, 620, 560
        axes(cv, X0, Y0, X1, Y1, a3, "比率 r", "目标")
        sx = lambda v: X0 + (v - 0.4) / 1.2 * (X1 - X0)
        sy = lambda v: (Y0 + Y1) / 2 - v / 2.2 * (Y1 - Y0) / 2
        cv.line((sx(1), Y0), (sx(1), Y1), (90, 108, 130), width=1, alpha=0.6 * a3, dash=5)
        for eps, sign, col in ((0.2, 1, GR), (0.2, -1, MG)):
            lo, hi = 1 - eps, 1 + eps
            pts = []
            for r in np.linspace(0.4, 1.6, 120):
                val = min(r * sign, max(lo, min(hi, r)) * sign)
                pts.append((sx(r), sy(val)))
            curvy(cv, pts, col, 2.2, a3)
        cv.text(sx(1.42), sy(1.55), "A > 0", size=17, color=GR, alpha=a3)
        cv.text(sx(1.42), sy(-1.55), "A < 0", size=17, color=MG, alpha=a3)
        cv.text(sx(1.0), Y1 - 8, "r = 1", size=16, color=FAINT, anchor="ma", alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(680, 380, "A>0：r 涨到 1.2 就截住", size=18, color=GR, alpha=a4)
    cv.text(680, 420, "A<0：r 跌到 0.8 就截住", size=18, color=MG, alpha=a4)
    cv.text(680, 470, "min 只在“贪心过头”时生效", size=18, color=DIM, alpha=a4)
    math_(cv, 680, 520, r"\epsilon=0.2,\ \ 2\ \text{epoch}", 18, DIM)


def s4(cv, t, dur):
    title(cv, "GAE：偏差与方差的折中")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 200, r"\hat A^{\mathrm{GAE}(\lambda)}_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},"
                        r"\quad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)", 20, INK)
    a2 = appear(t, 1.8, 0.8)
    rows = [("λ = 0", "Â = δ", "方差小、偏差大（依赖 V 的准确性）", MG),
            ("λ = 0.95", "本课取值", "偏向低方差一侧", CY),
            ("λ = 1", "Â = R − V", "无偏、方差大", AM)]
    for i, (k, f, note, col) in enumerate(rows):
        aa = appear(t, 2.0 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 290 + i * 60
        cv.text(110, y, k, size=22, color=col, alpha=aa)
        cv.text(260, y, f, size=20, color=INK, alpha=aa)
        cv.text(470, y, note, size=18, color=FAINT, alpha=aa)
    a3 = appear(t, 3.8, 0.8)
    cv.text(110, 500, "为什么本课偏向低方差：V 是同一个小网络估的，不够准", size=19, color=DIM, alpha=a3)
    a4 = appear(t, 4.8, 0.8)
    cv.text(110, 552, "γ = 0.995 → 有效视野 1/(1−γ) = 200 步 = 4 s", size=19, color=DIM, alpha=a4)


def s5(cv, t, dur):
    title(cv, "奖励：写物理量，不写指标")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 200, r"r=\cos\theta-0.15p^2-0.5\Big(\frac{u}{12}\Big)^2"
                        r"+0.5\,\mathbf 1\{|\theta|<0.1\}-4\,\mathbf 1\{\text{撞墙}\}", 19, INK)
    a2 = appear(t, 1.8, 0.8)
    for i, s in enumerate(["cosθ：直立 1、下垂 −1（摆起的动力源）",
                           "−0.15p²：车别跑远", "−0.5(u/12)²：省力、压制饱和",
                           "+0.5：真的站顶才加分", "−4：撞墙重罚"]):
        cv.text(130, 256 + i * 38, s, size=18, color=DIM, alpha=appear(t, 1.9 + 0.35 * i, 0.5))
    a3 = appear(t, 4.2, 0.8)
    chip(cv, 130, 470, "没有“摆起时间”这一项：速度是 cosθ 累积出来的", AM, a3, 18)
    a4 = appear(t, 5.2, 0.8)
    cv.text(700, 300, "势函数整形不改变最优解", size=20, color=CY, alpha=a4)
    math_(cv, 700, 350, r"r'=r+\gamma\Phi(s')-\Phi(s)", 20, INK)
    cv.text(700, 400, "沿轨迹求和后只剩边界项", size=18, color=FAINT, alpha=a4)
    cv.text(700, 448, "所以“加引导奖励”是安全的", size=18, color=DIM, alpha=a4)


def s6(cv, t, dur):
    title(cv, "域随机化 = 分布鲁棒的平均版")
    a = appear(t, 0.3, 0.8)
    math_(cv, 110, 196, r"\max_\pi\ \mathbb{E}_{\xi\sim\mathcal{U}}\ \mathbb{E}_{\tau\sim P_\xi}[R(\tau)]",
          21, INK)
    a2 = appear(t, 1.8, 0.8)
    cv.text(110, 240, "ξ = 风的强度与时段；在“一族环境”上都要不错", size=19, color=DIM, alpha=a2)

    a3 = appear(t, 2.8, 0.9)
    bars = [("无 DR", 0.03, MG), ("DR", 1.00, GR)]
    for i, (k, v, col) in enumerate(bars):
        aa = appear(t, 2.8 + 0.5 * i, 0.55)
        if aa <= 0:
            continue
        x = 240 + i * 220
        h = max(5, v * 200)
        cv.rect(x - 60, 520 - h, x + 60, 520, col, width=1, alpha=0.85 * aa,
                fill_alpha=0.30 * aa, radius=4)
        cv.text(x, 520 - h - 18, f"{v*100:.0f}%", size=26, color=col, anchor="ma", alpha=aa)
        cv.text(x, 550, k, size=19, color=INK, anchor="ma", alpha=aa)
    cv.text(350, 584, "+4 N 持续侧风成功率", size=17, color=FAINT, anchor="ma", alpha=a3)

    a4 = appear(t, 4.6, 0.8)
    cv.text(680, 300, "为什么无 DR 只有 3%？", size=20, color=AM, alpha=a4)
    cv.text(680, 340, "训练里没有恒力，位置误差几乎不被惩罚", size=18, color=DIM, alpha=a4)
    cv.text(680, 372, "→ 学会了“小车漂一点无所谓”", size=18, color=DIM, alpha=a4)
    cv.text(680, 420, "鲁棒性是分布的性质，不是参数的性质", size=19, color=CY, alpha=a4)


def s7(cv, t, dur):
    title(cv, "结果：赢了 3 个指标，输了一项")
    a = appear(t, 0.3, 0.8)
    rows = [("专家（能量+LQR）", "2.84", "2.42", "107", "10.05", "100%", DIM),
            ("BC + DAgger", "2.74", "2.39", "109", "5.36", "80%", MG),
            ("RL", "2.22", "2.06", "65", "3.89", "3%", CY),
            ("RL + DR", "2.31", "2.12", "70", "3.74", "100%", GR)]
    cv.text(110, 190, "策略", size=17, color=FAINT, alpha=a)
    cv.text(480, 190, "摆起", size=17, color=FAINT, alpha=a)
    cv.text(620, 190, "力 RMS", size=17, color=FAINT, alpha=a)
    cv.text(760, 190, "力总变差", size=17, color=FAINT, alpha=a)
    cv.text(920, 190, "最大 Δu", size=17, color=FAINT, alpha=a)
    cv.text(1060, 190, "+4N", size=17, color=FAINT, alpha=a)
    for i, (k, t1, u, tv, du, s, col) in enumerate(rows):
        aa = appear(t, 0.6 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 240 + i * 56
        cv.text(110, y, k, size=19, color=col, alpha=aa)
        cv.text(480, y, t1, size=19, color=col, alpha=aa)
        cv.text(620, y, u, size=19, color=col, alpha=aa)
        cv.text(760, y, tv, size=19, color=col, alpha=aa)
        cv.text(920, y, du, size=19, color=col, alpha=aa)
        cv.text(1060, y, s, size=19, color=col, alpha=aa)
        cv.line((110, y + 14), (1150, y + 14), (60, 76, 94), width=1, alpha=0.4 * aa)
    a2 = appear(t, 4.0, 0.8)
    cv.text(110, 500, "摆起 −22%、力 −15%、力总变差 −40%（107 → 65）", size=19, color=CY, alpha=a2)
    a3 = appear(t, 4.8, 0.8)
    cv.text(110, 540, "最大单步跳变 10.05 → 3.74 N：硬切换被学成连续映射", size=19, color=DIM, alpha=a3)
    a4 = appear(t, 5.6, 0.8)
    cv.text(110, 590, "唯一输的是抗扰：3% → DR 修复到 100%，代价摆起慢 0.09 s", size=19,
            color=AM, alpha=a4)


def s8(cv, t, dur):
    title(cv, "本讲小结")
    items = [("策略梯度", "grad J = E[grad log π · A]，从 BC 热启动"),
             ("PPO", "clip 是软信任域，只在贪心过头时截断"),
             ("GAE", "λ 在偏差与方差间折中，本课 0.95"),
             ("奖励", "写物理量；势函数整形不改最优解"),
             ("域随机化", "分布鲁棒的平均版：3% → 100%"),
             ("结论", "RL 超越的是你写下来的目标")]
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
    cv.text(130, 580, "结课项目：把这些控制器搬到实机 / 半实物上", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "最后一讲。上一讲的策略已经和专家同级，但它只是在模仿。",
        "这一讲换成策略梯度：直接最大化闭环回报。",
        "从 MDP 和梯度定理推导到 PPO、GAE，再讲域随机化为什么能修好分布外。",
    ], draw=s1, min_dur=13.0),
    dict(name="pg", say=[
        "先把目标写出来：期望折扣回报。",
        {"say": "对它求梯度，用对数似然技巧，得到“log 策略的梯度乘上优势”。",
         "tts": "对它求梯度，用对数似然技巧，得到 log 策略的梯度乘上优势。"},
        "优势用回报减去基线；基线不改变期望，只降方差。",
        {"say": "本课用高斯策略，并且从行为克隆的权重热启动——因为策略梯度只是局部搜索。",
         "tts": "本课用高斯策略，并且从行为克隆的权重热启动——因为策略梯度只是局部搜索。"},
    ], draw=s2, min_dur=19.0),
    dict(name="ppo", say=[
        "PPO 的核心是一个裁剪的重要性比率。",
        "比率是“新策略动作概率除以旧策略”，它偏离一太远，估计就失效。",
        {"say": "clip 把它截在 $0.8$ 到 $1.2$：好动作涨到上限就停，坏动作跌到下限就停。",
         "tts": "clip 把它截在零点八到一点二：好动作涨到上限就停，坏动作跌到下限就停。"},
        "这个 min 只在“贪心过头”时生效，所以它等价于一个软信任域。",
    ], draw=s3, min_dur=19.0),
    dict(name="gae", say=[
        "优势怎么估？GAE 用指数加权把多步优势平均起来。",
        "lambda 等于零只用一步，方差小但依赖价值函数；等于一用整条回报，无偏但方差大。",
        "本课取 0.95，因为价值网络是同一个小模型估的，不够准。",
    ], draw=s4, min_dur=17.0),
    dict(name="reward", say=[
        "奖励设计有一条原则：写物理量，不写指标。",
        "这里没有“摆起时间”，速度是 cos 的时间累积自然产生的。",
        {"say": "另外，只要整形项写成势函数的差分，最优策略不变——这是 $1999$ 年的经典结论。",
         "tts": "另外，只要整形项写成势函数的差分，最优策略不变——这是一九九九年的经典结论。"},
    ], draw=s5, min_dur=18.0),
    dict(name="dr", say=[
        "然后是这一讲最关键的一段：域随机化。",
        {"say": "它在“一族环境”上做平均，是分布鲁棒优化的平均版。",
         "tts": "它在“一族环境”上做平均，是分布鲁棒优化的平均版。"},
        "不加随机化，4 牛顿侧风下只剩 3%；加上之后回到 100%。",
        "原因很直白：训练里没有恒力，位置误差不被惩罚，策略就学会了“车漂一点无所谓”。",
    ], draw=s6, min_dur=19.0),
    dict(name="result", say=[
        "最后看这张表。",
        {"say": "摆起时间少 22%，力小 15%，力总变差少 40%。",
         "tts": "摆起时间少百分之二十二，力小百分之十五，力总变差少百分之四十。"},
        "唯一输的是抗扰——这正是分布外问题，靠域随机化修回来。",
    ], draw=s7, min_dur=18.0),
    dict(name="summary", say=[
        "整门课到这里结束：从拉格朗日方程，到能量与滑模，再到从数据里学一个控制器。",
        "每一讲都留下一个数字，也留下一个失败案例。",
        "接下来是结课项目：把这些控制器搬到实机上。",
    ], draw=s8, min_dur=15.0),
]

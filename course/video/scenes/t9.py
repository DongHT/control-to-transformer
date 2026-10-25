"""学习控制 · 第九集：PPO 从原理到实现 —— 10 个镜头。"""
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
RC = json.load(open(os.path.join(BUILD, "out", "rl_compare.json")))
_LINES = open(os.path.join(BUILD, "rl.py"), encoding="utf-8").read().split("\n")


def src(anchor, n, dedent=None):
    for i, ln in enumerate(_LINES):
        if anchor in ln:
            block = _LINES[i:i + n]
            if dedent is None:
                dedent = len(block[0]) - len(block[0].lstrip())
            return [b[dedent:] if len(b) >= dedent else b for b in block]
    return [f"# anchor not found: {anchor}"]


def s1(cv, t, dur):
    cv.text(640, 246, "学习控制 · 从原理到实现　·　第九集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 340, "PPO：直接用回报优化策略", size=54, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 402, "策略梯度 → GAE → clip → 域随机化", size=27, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 440), (810, 440), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 506, r"\max_\theta\ \mathbb{E}_{\tau\sim\pi_\theta}"
                            r"\Big[\sum_t r_t\Big]", 28, DIM, anchor="mc")
    cv.text(640, 600, "Proximal Policy Optimization　近端策略优化", size=18, color=FAINT,
            anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "把控制问题写成 MDP")
    a = appear(t, 0.3, 0.8)
    rows = [("状态 S", "4 维物理状态（用 24 步历史作观测）"),
            ("动作 A", "力 u ∈ [−12, 12] N"),
            ("转移 P", "第 1 讲推导的动力学"),
            ("奖励 r", "每 20 ms 一步（下一页）"),
            ("折扣 γ", "0.995 → 有效视野约 4 s")]
    for i, (k, v) in enumerate(rows):
        aa = appear(t, 0.5 + 0.45 * i, 0.5)
        if aa <= 0:
            continue
        y = 200 + i * 52
        cv.text(110, y, k, size=19, color=CY, alpha=aa)
        cv.text(330, y, v, size=18, color=INK, alpha=aa)
    a2 = appear(t, 3.6, 0.8)
    math_(cv, 110, 490, r"J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\Big[\sum_{t}\gamma^t r_t\Big]",
          20, AM)
    a3 = appear(t, 4.8, 0.8)
    cv.text(110, 546, "与模仿学习的差别：这里没有“专家动作”，只有回报", size=17, color=DIM, alpha=a3)


def s3(cv, t, dur):
    title(cv, "策略梯度定理（三行推导）")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"\nabla_\theta J=\nabla_\theta\int p_\theta(\tau)R(\tau)d\tau"
                        r"=\int p_\theta\nabla_\theta\log p_\theta\,R\,d\tau", 18, INK)
    a2 = appear(t, 1.8, 0.8)
    math_(cv, 100, 238, r"\log p_\theta(\tau)=\sum_t\log\pi_\theta(a_t|s_t)+\text{常数}", 18, CY)
    a3 = appear(t, 3.0, 0.8)
    math_(cv, 100, 290, r"\nabla_\theta J=\mathbb{E}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)\,"
                        r"\big(R_t-\gamma V(s_{t+1})\big)\Big]", 17, AM)
    a4 = appear(t, 4.2, 0.8)
    cv.text(100, 344, "三件事：① 对数似然技巧 ② 因果性去掉未来项 ③ 基线降方差", size=16,
            color=DIM, alpha=a4)
    a5 = appear(t, 5.2, 0.9)
    cv.rect(620, 180, 1180, 400, CY, width=1.2, alpha=0.7 * a5, fill_alpha=0.05 * a5, radius=8)
    cv.text(642, 212, "为什么必须从 BC 热启动", size=18, color=CY, alpha=a5)
    cv.text(642, 252, "策略梯度是局部搜索：", size=16, color=INK, alpha=a5)
    cv.text(642, 282, "从一个“已经会控制”的点出发，", size=16, color=INK, alpha=a5)
    cv.text(642, 312, "只在附近找更优解。", size=16, color=INK, alpha=a5)
    cv.text(642, 356, "随机初始化几乎学不会摆起", size=15, color=MG, alpha=a5)


def s4(cv, t, dur):
    title(cv, "GAE：优势估计的折中")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"\hat A_t^{\mathrm{GAE}(\lambda)}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},"
                        r"\quad \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)", 18, INK)
    a2 = appear(t, 1.8, 0.8)
    rows = [("λ = 0", "只用一步", "方差小、偏差大", MG),
            ("λ = 0.95", "本课取值", "偏向低方差", CY),
            ("λ = 1", "用整条回报", "无偏、方差大", AM)]
    for i, (k, f, note, col) in enumerate(rows):
        aa = appear(t, 2.0 + 0.5 * i, 0.5)
        if aa <= 0:
            continue
        y = 260 + i * 56
        cv.text(110, y, k, size=20, color=col, alpha=aa)
        cv.text(260, y, f, size=18, color=INK, alpha=aa)
        cv.text(470, y, note, size=16, color=FAINT, alpha=aa)
    a3 = appear(t, 3.8, 0.8)
    cv.text(110, 460, "本课实现里价值函数就用同一套历史窗口估计，所以偏向低方差一侧", size=16,
            color=DIM, alpha=a3)
    a4 = appear(t, 4.8, 0.8)
    math_(cv, 110, 512, r"\gamma=0.995\ \Rightarrow\ 1/(1-\gamma)=200\ \text{步}=4\ \mathrm{s}", 17, DIM)


def s5(cv, t, dur):
    title(cv, "PPO：clip 就是软信任域")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 180, r"r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}", 20, INK)
    a2 = appear(t, 1.6, 0.8)
    math_(cv, 100, 240, r"L^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_tA_t,\ "
                        r"\mathrm{clip}(r_t,0.8,1.2)A_t)\big]", 18, CY)
    a3 = appear(t, 2.8, 0.9)
    if a3 > 0:
        X0, Y0, X1, Y1 = 100, 320, 600, 540
        axes(cv, X0, Y0, X1, Y1, a3, "比率 r", "目标")
        sx = lambda v: X0 + (v - 0.4) / 1.2 * (X1 - X0)
        sy = lambda v: (Y0 + Y1) / 2 - v / 2.2 * (Y1 - Y0) / 2
        cv.line((sx(1), Y0), (sx(1), Y1), (90, 108, 130), width=1, alpha=0.5 * a3, dash=5)
        for sign, col in ((1, GR), (-1, MG)):
            pts = [(sx(r), sy(min(r * sign, max(0.8, min(1.2, r)) * sign)))
                   for r in np.linspace(0.4, 1.6, 120)]
            curvy(cv, pts, col, 2.2, a3)
        cv.text(sx(1.45), sy(1.7), "A > 0", size=15, color=GR, alpha=a3)
        cv.text(sx(1.45), sy(-1.7), "A < 0", size=15, color=MG, alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(660, 340, "好动作涨到 1.2 就截住", size=17, color=GR, alpha=a4)
    cv.text(660, 372, "坏动作跌到 0.8 就截住", size=17, color=MG, alpha=a4)
    cv.text(660, 412, "min 只在“贪心过头”时生效", size=17, color=DIM, alpha=a4)
    cv.text(660, 452, "等价于一个软信任域", size=17, color=CY, alpha=a4)


def s6(cv, t, dur):
    title(cv, "实现 ①：收集与奖励")
    code(cv, 90, 178, src("            r = (np.cos(th) - 0.15 * p ** 2", 3), size=13, hl={0}, hl_color=CY)
    a = appear(t, 2.4, 0.8)
    cv.text(90, 290, "奖励的每一项都在说一件事", size=18, color=INK, alpha=a)
    rows = [("cos θ", "直立 1、下垂 −1（摆起的动力）"),
            ("−0.15 p²", "车别跑远"),
            ("−0.5 (u/12)²", "省力，也压制饱和"),
            ("+0.5·1{|θ|<0.1}", "真的站顶才加分"),
            ("−4·1{撞墙}", "撞墙重罚一次")]
    for i, (k, v) in enumerate(rows):
        aa = appear(t, 2.8 + 0.4 * i, 0.5)
        cv.text(110, 330 + i * 40, k, size=16, color=CY, alpha=aa)
        cv.text(330, 330 + i * 40, v, size=16, color=DIM, alpha=aa)
    a2 = appear(t, 5.4, 0.8)
    chip(cv, 620, 300, "没有“摆起时间”这一项：写物理量，不写指标", AM, a2, 17)
    a3 = appear(t, 6.2, 0.8)
    cv.text(620, 380, "48 个并行环境 × 8 s，每轮 2 个 epoch", size=16, color=DIM, alpha=a3)
    cv.text(620, 412, "学习率 1.5e-4，clip 0.2，梯度裁剪 0.5", size=16, color=DIM, alpha=a3)


def s7(cv, t, dur):
    title(cv, "实现 ②：回报与优势")
    code(cv, 90, 176, src("        ret = np.zeros_like(rew_s); G = np.zeros(n_env)", 7),
         size=13, hl={2, 5, 6}, hl_color=CY)
    a = appear(t, 2.6, 0.8)
    cv.text(90, 330, "三个细节", size=18, color=INK, alpha=a)
    a2 = appear(t, 3.0, 0.8)
    cv.text(110, 370, "· 固定视野 8 s，不需要 bootstrap，直接倒序累加回报", size=16, color=DIM, alpha=a2)
    cv.text(110, 402, "· 优势 = 回报 − 批内均值（这一步等价于一个基线）", size=16, color=DIM, alpha=a2)
    cv.text(110, 434, "· 优势再标准化：除以标准差，稳定梯度尺度", size=16, color=DIM, alpha=a2)
    a3 = appear(t, 5.0, 0.9)
    if a3 > 0:
        cv.text(620, 200, "为什么要减均值、除标准差", size=17, color=CY, alpha=a3)
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, 40) + 2
        for i, x in enumerate(v):
            xx = 620 + i * 13
            hh = max(2, abs(x) * 30)
            cv.rect(xx, 300 - hh if x > 0 else 300, xx + 9, 300, CY if x > 0 else MG,
                    width=0, alpha=0.8 * a3, fill_alpha=0.45 * a3, radius=2)
        cv.line((620, 300), (1140, 300), (90, 108, 130), width=1, alpha=0.7 * a3)
        cv.text(620, 330, "原始优势（均值 2、尺度不一）", size=14, color=FAINT, alpha=a3)
        a4 = appear(t, 6.4, 0.8)
        cv.text(620, 380, "不标准化：梯度被少数大值主导，", size=16, color=MG, alpha=a4)
        cv.text(620, 410, "策略更新会被单个样本带偏", size=16, color=MG, alpha=a4)


def s8(cv, t, dur):
    title(cv, "实现 ③：clip 损失与更新")
    code(cv, 90, 176, src("                ratio = torch.exp(logp - oldlogp_t[b])", 7),
         size=13, hl={1, 3, 4}, hl_color=CY)
    a = appear(t, 2.8, 0.8)
    cv.text(90, 380, "这七行就是 PPO 的全部", size=18, color=INK, alpha=a)
    a2 = appear(t, 3.4, 0.8)
    cv.text(110, 420, "· ratio = exp(logp − logp_old)：重要性采样比率", size=16, color=DIM, alpha=a2)
    cv.text(110, 452, "· s1 是原始目标，s2 是裁剪后的目标", size=16, color=DIM, alpha=a2)
    cv.text(110, 484, "· 取负号做梯度下降；梯度裁剪 0.5 防跳飞", size=16, color=DIM, alpha=a2)
    a3 = appear(t, 5.2, 0.9)
    cv.rect(620, 190, 1180, 360, CY, width=1.2, alpha=0.7 * a3,
            fill_alpha=0.05 * a3, radius=8)
    cv.text(642, 222, "策略网络的小改动", size=18, color=CY, alpha=a3)
    cv.text(642, 262, "输出均值 μ（由网络给出）", size=16, color=INK, alpha=a3)
    cv.text(642, 294, "标准差 σ 也是可学习参数（log_std）", size=16, color=INK, alpha=a3)
    cv.text(642, 326, "评估时取均值，即确定性策略", size=16, color=INK, alpha=a3)


def s9(cv, t, dur):
    title(cv, "结果：赢了三个指标，输了一项")
    a = appear(t, 0.3, 0.8)
    rows = [("专家（能量+LQR）", RC["expert"], DIM),
            ("Transformer（BC）", RC["bc"], MG),
            ("Transformer + RL", RC["rl"], CY),
            ("Transformer + RL + DR", RC["rl_dr"], GR)]
    cv.text(110, 190, "策略", size=16, color=FAINT, alpha=a)
    cv.text(470, 190, "摆起", size=16, color=FAINT, alpha=a)
    cv.text(590, 190, "力 RMS", size=16, color=FAINT, alpha=a)
    cv.text(720, 190, "力总变差", size=16, color=FAINT, alpha=a)
    cv.text(860, 190, "最大 Δu", size=16, color=FAINT, alpha=a)
    cv.text(1000, 190, "+4N 侧风", size=16, color=FAINT, alpha=a)
    for i, (k, d, col) in enumerate(rows):
        aa = appear(t, 0.6 + 0.55 * i, 0.5)
        if aa <= 0:
            continue
        y = 240 + i * 56
        n = d["nominal"]; ds = d["disturbed"]
        cv.text(110, y, k, size=18, color=col, alpha=aa)
        cv.text(470, y, f"{n['t_up']:.2f}", size=18, color=col, alpha=aa)
        cv.text(590, y, f"{n['u']:.2f}", size=18, color=col, alpha=aa)
        cv.text(720, y, f"{n['tv']:.0f}", size=18, color=col, alpha=aa)
        cv.text(860, y, f"{n['du']:.2f}", size=18, color=col, alpha=aa)
        cv.text(1000, y, f"{ds['success']*100:.0f}%", size=18,
                color=(MG if ds["success"] < 0.5 else GR), alpha=aa)
        cv.line((110, y + 16), (1150, y + 16), (60, 76, 94), width=1, alpha=0.4 * aa)
    a2 = appear(t, 3.4, 0.8)
    cv.text(110, 500, "摆起 −22%、力 −15%、力总变差 −40%", size=18, color=CY, alpha=a2)
    a3 = appear(t, 4.2, 0.8)
    cv.text(110, 540, "最大单步跳变 10.05 → 3.74 N：硬切换被学成连续映射", size=18, color=DIM, alpha=a3)
    a4 = appear(t, 5.2, 0.8)
    cv.text(110, 588, "但 +4N 侧风只剩 3% —— 训练分布里根本没有风", size=19, color=MG, alpha=a4)


def s10(cv, t, dur):
    title(cv, "域随机化：把鲁棒性写进分布")
    a = appear(t, 0.3, 0.8)
    math_(cv, 100, 186, r"\max_\pi\ \mathbb{E}_{\xi\sim\mathcal{U}}\ \mathbb{E}_{\tau\sim P_\xi}[R(\tau)]",
          19, INK)
    a2 = appear(t, 1.6, 0.8)
    cv.text(100, 230, "ξ = 风的强度与时段；在一族环境上做平均", size=17, color=DIM, alpha=a2)
    a3 = appear(t, 2.4, 0.9)
    bars = [("无 DR", RC["rl"]["disturbed"]["success"], MG), ("DR", RC["rl_dr"]["disturbed"]["success"], GR)]
    for i, (k, v, col) in enumerate(bars):
        aa = appear(t, 2.6 + 0.5 * i, 0.5)
        x = 240 + i * 200
        hh = max(5, v * 180)
        cv.rect(x - 55, 470 - hh, x + 55, 470, col, width=1, alpha=0.85 * aa,
                fill_alpha=0.32 * aa, radius=4)
        cv.text(x, 470 - hh - 16, f"{v*100:.0f}%", size=24, color=col, anchor="ma", alpha=aa)
        cv.text(x, 498, k, size=17, color=INK, anchor="ma", alpha=aa)
    cv.text(340, 534, "+4 N 持续侧风下的成功率", size=15, color=FAINT, anchor="ma", alpha=a3)

    a4 = appear(t, 4.6, 0.8)
    cv.text(620, 220, "为什么无 DR 只有 3%", size=18, color=AM, alpha=a4)
    cv.text(620, 258, "训练里没有恒力，位置误差几乎不被惩罚", size=16, color=DIM, alpha=a4)
    cv.text(620, 288, "于是学会了“车漂一点无所谓”", size=16, color=DIM, alpha=a4)
    a5 = appear(t, 5.6, 0.8)
    chip(cv, 620, 350, "鲁棒性是分布的性质，不是参数的性质", CY, a5, 17)
    a6 = appear(t, 6.4, 0.8)
    cv.text(620, 420, "代价：摆起 2.22 → 2.31 s，几乎没损失", size=16, color=DIM, alpha=a6)


def s11(cv, t, dur):
    title(cv, "本集小结")
    items = [("MDP", "状态、动作、转移、奖励、折扣"),
             ("策略梯度", "对数似然 + 因果性 + 基线"),
             ("GAE", "λ 在偏差与方差之间折中"),
             ("PPO", "clip 是软信任域，只在贪心过头时截断"),
             ("实现", "七行核心更新 + 优势标准化"),
             ("教训", "RL 超越你写下的目标，超越不了没写下的")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.6 * i, 0.55)
        if aa <= 0:
            continue
        y = 186 + i * 58
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=22, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=20, color=INK, alpha=aa)
    a = appear(t, 4.4, 0.8)
    cv.text(130, 580, "下一集：把这些控制器搬到实机（结课项目）", size=24, color=AM, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "第九集讲 PPO，同样从原理到实现。",
        "全称 Proximal Policy Optimization，中文叫近端策略优化。",
        "它只做一件事：不模仿专家，直接用回报去改进策略。",
    ], draw=s1, min_dur=13.0),
    dict(name="mdp", say=[
        "先把控制问题写成马尔可夫决策过程，英文缩写 MDP。",
        {"say": "状态是四维物理量，动作是力，转移就是第一讲的动力学，折扣 $0.995$。",
         "tts": "状态是四维物理量，动作是力，转移就是第一讲的动力学，折扣零点九九五。"},
        "和模仿学习最大的差别：这里没有专家动作，只有回报。",
    ], draw=s2, min_dur=17.0),
    dict(name="pg", say=[
        "目标是期望折扣回报，但怎么对它求导？",
        "三个技巧：对数似然把分布求导变成 log 求导；因果性去掉未来的项；再减一个基线降方差。",
        {"say": "注意这只是一个局部搜索，所以必须从行为克隆的权重热启动。",
         "tts": "注意这只是一个局部搜索，所以必须从行为克隆的权重热启动。"},
        "随机初始化几乎学不会摆起。",
    ], draw=s3, min_dur=19.0),
    dict(name="gae", say=[
        "优势怎么估？用 GAE，广义优势估计。",
        {"say": "λ 等于零只用一步，方差小但依赖价值函数；等于一用整条回报，无偏但方差大。",
         "tts": "lambda 等于零只用一步，方差小但依赖价值函数；等于一用整条回报，无偏但方差大。"},
        "本课取零九五，因为价值函数也是同一个小模型估的，不够准。",
    ], draw=s4, min_dur=17.0),
    dict(name="clip", say=[
        "PPO 的核心是一个裁剪的重要性比率。",
        "比率是“新策略动作概率除以旧策略”，偏离太远估计就失效。",
        {"say": "clip 把它截在 $0.8$ 到 $1.2$：好动作涨到上限就停，坏动作跌到下限就停。",
         "tts": "clip 把它截在零点八到一点二：好动作涨到上限就停，坏动作跌到下限就停。"},
        "这个 min 只在贪心过头时生效，所以等价于一个软信任域。",
    ], draw=s5, min_dur=18.0),
    dict(name="impl1", say=[
        "实现第一步：收集与奖励。",
        "奖励每一项都在说一件事：cos 让你站起来，位置项让车别跑，力的平方项让你省力。",
        "注意这里没有“摆起时间”，写物理量而不是写指标。",
        "四十八个并行环境，每轮两个 epoch，学习率一点五乘十的负四次方。",
    ], draw=s6, min_dur=18.0),
    dict(name="impl2", say=[
        "第二步：算回报与优势。",
        "固定视野八秒，所以不需要 bootstrap，直接倒序累加。",
        {"say": "优势再减均值、除标准差——这一步很关键，否则梯度会被少数大值带偏。",
         "tts": "优势再减均值、除标准差——这一步很关键，否则梯度会被少数大值带偏。"},
    ], draw=s7, min_dur=16.0),
    dict(name="impl3", say=[
        "第三步就是那七行核心更新。",
        "算比率、算原始目标、算裁剪目标、取最小值、加负号、反向传播、梯度裁剪。",
        "策略网络比之前多一个可学习参数：标准差。评估时取均值，就是确定性策略。",
    ], draw=s8, min_dur=16.0),
    dict(name="result", say=[
        "结果是这样：赢了三个指标，输了一项。",
        {"say": "摆起时间少百分之二十二，力小百分之十五，力总变差少百分之四十。",
         "tts": "摆起时间少百分之二十二，力小百分之十五，力总变差少百分之四十。"},
        {"say": "最大单步跳变从 $10$ 牛顿降到 $3.74$ 牛顿：它把硬切换学成了连续映射。",
         "tts": "最大单步跳变从十牛顿降到三点七四牛顿：它把硬切换学成了连续映射。"},
        "但持续侧风下只剩百分之三。",
    ], draw=s9, min_dur=19.0),
    dict(name="dr", say=[
        "最后用域随机化修它。",
        "六成回合随机带风，让策略在“一族环境”上都不错。",
        {"say": "成功率从 3% 回到 100%，代价是摆起慢了 0.09 秒。",
         "tts": "成功率从百分之三回到百分之百，代价是摆起慢了零点零九秒。"},
        "结论：鲁棒性是分布的性质，不是参数的性质。",
    ], draw=s10, min_dur=18.0),
    dict(name="summary", say=[
        "小结：MDP、策略梯度、GAE、PPO、实现、以及那条教训。",
        "RL 能超越你写下来的目标，但超越不了你没写下来的东西。",
    ], draw=s11, min_dur=14.0),
]

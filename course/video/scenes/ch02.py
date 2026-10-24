"""第 2 讲：线性基线（LQR 与能力边界）—— 8 个镜头。"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kit import (math_, math_w, title, chip, axes, cartpole, curvy, basin_image,
                 MC, MP, L, G, UMAX, SCALE, RAIL)
from render import (Canvas, INK, DIM, FAINT, CY, AM, MG, BL, GR, BG,
                    appear, smooth, lerp, W, H)

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "..", "..", "..", "build")
DATA = json.load(open(os.path.join(BUILD, "out", "ch02.json")))
sys.path.insert(0, BUILD)
import plant as pl                      # noqa: E402

K = np.array(DATA["design"]["K"])
BASIN = DATA["basin"]
FAIL = DATA["fail_traj"]


# ---------------------------------------------------------------- data
def rollout(th0, T=2.0):
    x = np.array([0.0, th0, 0.0, 0.0])
    ts, th, us = [], [], []
    for k in range(int(T / 0.02)):
        u = float(np.clip(-K @ x, -UMAX, UMAX))
        ts.append(k * 0.02); th.append(math.degrees(x[1])); us.append(u)
        x = pl.rk4(x, u, 0.02)
        if abs(x[0]) > pl.P["x_max"]:
            break
    return np.array(ts), np.array(th), np.array(us)


def phase_traj(th0, w0, T=2.4):
    """closed-loop phase trajectory + its true outcome (colour must follow the outcome)."""
    x = np.array([0.0, th0, 0.0, w0])
    pts, hit = [], False
    for k in range(int(T / 0.02)):
        u = float(np.clip(-K @ x, -UMAX, UMAX))
        x = pl.rk4(x, u, 0.02)
        if abs(x[0]) > pl.P["x_max"]:
            hit = True
            break
        pts.append((x[1], x[3]))
    good = (not hit) and abs(((x[1] + np.pi) % (2 * np.pi)) - np.pi) < 0.05 and abs(x[3]) < 0.2
    return np.array(pts), good


# ---- pick trajectories that never cross (projecting 4D flow onto (theta, w)
# can create crossings; they look like a bug even when they are correct maths)
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(p1, p2, p3, p4):
    d1 = _orient(p3, p4, p1); d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3); d4 = _orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _self_cross(pts, step=2):
    p = pts[::step]
    n = len(p)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if _seg_cross(p[i], p[i + 1], p[j], p[j + 1]):
                return True
    return False


def _pair_cross(A, B, step=2):
    a = A[::step]; b = B[::step]
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            if _seg_cross(a[i], a[i + 1], b[j], b[j + 1]):
                return True
    return False


def _trim(pts, th_stop=0.25, w_stop=0.8):
    """stop drawing before all trajectories collapse onto the origin."""
    for i, (th, w) in enumerate(pts):
        if abs(th) < th_stop and abs(w) < w_stop:
            return pts[:max(8, i + 1)]
    return pts


def _pick():
    """2 converging + 2 diverging trajectories, pairwise non-crossing.

    The colour always comes from the simulation outcome, never from a hand label.
    """
    from itertools import combinations
    cache = {}

    def traj(ic):
        if ic not in cache:
            pts, good = phase_traj(*ic)
            cache[ic] = (_trim(pts), good)
        return cache[ic]

    infos = {ic: traj(ic) for ic in _CANDS}
    G = [ic for ic in _CANDS if infos[ic][1] and abs(ic[0]) > 0.2]
    B = [ic for ic in _CANDS if not infos[ic][1]]

    best = None
    for k in (2, 1):
        for gs in combinations(G, k):
            for bs in combinations(B, k):
                cands = [(ic, infos[ic][0]) for ic in gs] + [(ic, infos[ic][0]) for ic in bs]
                if any(_self_cross(c[1]) for c in cands):
                    continue
                n = len(cands)
                ok = True
                for i2 in range(n):
                    for m in range(i2 + 1, n):
                        if _pair_cross(cands[i2][1], cands[m][1]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(cands) == 4:
                    # the two converging curves must end visibly apart, not merge
                    d = ((cands[0][1][-1][0] - cands[1][1][-1][0]) ** 2
                         + (cands[0][1][-1][1] - cands[1][1][-1][1]) ** 2) ** 0.5
                    if d < 0.35:
                        continue
                if ok:
                    score = sum(len(c[1]) for c in cands)
                    if best is None or score > best[0]:
                        best = (score, cands)
        if best:
            return [(ic, infos[ic][0], infos[ic][1]) for ic, _ in best[1]]
    return []


_CANDS = [(0.36, -3.6), (0.56, -3.0), (-0.04, 0.0), (-0.44, 3.6), (0.76, 2.6),
          (0.16, -2.4), (-1.04, -3.6), (-1.04, 0.0), (0.96, 3.6), (-0.84, 2.2),
          (1.12, -0.4), (0.6, 3.8), (-0.64, -2.0), (0.24, 3.2)]


def _build_phase():
    sel = _pick()
    if not sel:                          # last resort: 1 + 1, crossings unchecked
        for ic in ((0.56, -3.0), (-1.04, 0.0)):
            pts, good = phase_traj(*ic)
            sel.append((ic, _trim(pts), good))
    return sel


PHASE = _build_phase()


T1, TH1, U1 = rollout(0.10)
T2, TH2, U2 = rollout(0.25)
BASIN_IMG = basin_image(BASIN, 380, 300)


# ============================================================ S1
def s1(cv, t, dur):
    cv.text(640, 252, "倒立摆控制　·　从拉格朗日到 Transformer",
            size=25, color=CY, anchor="ma", alpha=appear(t, 0.2, 0.9))
    cv.text(640, 352, "第 2 讲　线性基线", size=70, color=INK, anchor="ma",
            weight="bold", alpha=appear(t, 0.9, 0.9))
    cv.text(640, 414, "LQR 与它的能力边界", size=31, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((500, 450), (780, 450), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    a = appear(t, 2.6, 0.9)
    if a > 0:
        math_(cv, 640, 512, r"u=-Kx,\qquad K=R^{-1}B^{\top}P", 25, DIM, anchor="mc")
    cv.text(90, 620, "课程讲义 · ch02　　代码 · build/controllers.py", size=19,
            color=FAINT, alpha=appear(t, 3.0, 0.8))
    cartpole(cv, 640, 600, 0.06 * math.sin(0.9 * t), a=0.12 * appear(t, 0.5, 1.2), show_rail=False)


# ============================================================ S2
def s2(cv, t, dur):
    title(cv, "局部表现：快到没感觉")
    a = appear(t, 0.3, 0.8)
    # theta(t) panel
    x0, y0, x1, y1 = 110, 210, 620, 430
    axes(cv, x0, y0, x1, y1, a, "t [s]", "θ [°]")
    sx = lambda tv: x0 + tv / 1.6 * (x1 - x0)
    sy = lambda v: y1 - v / 16.0 * (y1 - y0)
    for v in (0, 4, 8, 12, 16):
        cv.line((x0 - 5, sy(v)), (x0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8)
        cv.text(x0 + 9, sy(v) + 6, str(v), size=15, color=FAINT)
    for tv in (0, 0.4, 0.8, 1.2, 1.6):
        cv.line((sx(tv), y1), (sx(tv), y1 + 5), (92, 110, 132), width=1.2, alpha=0.8)
        cv.text(sx(tv), y1 + 26, f"{tv:g}", size=15, color=FAINT, anchor="ma")
    n1 = int(min(len(T1), 1.6 / 0.02))
    n2 = int(min(len(T2), 1.6 / 0.02))
    prog = min(1.0, max(0.0, (t - 1.2) / 3.0))
    if prog > 0:
        k1 = max(2, int(n1 * prog)); k2 = max(2, int(n2 * prog))
        curvy(cv, [(sx(T2[i]), sy(min(TH2[i], 16))) for i in range(k2)], MG, 2.2, a)
        curvy(cv, [(sx(T1[i]), sy(min(TH1[i], 16))) for i in range(k1)], CY, 2.2, a)
        cv.text(sx(1.0), sy(13.5), "θ₀ = 0.25 rad", size=18, color=MG, alpha=a)
        cv.text(sx(1.0), sy(9.5), "θ₀ = 0.10 rad", size=18, color=CY, alpha=a)

    # numbers on the right
    cards = [("0.24 s", "进入 ±0.02 rad", CY, 1.8),
             ("6.29 N", "峰值力（0.1 rad）", INK, 2.6),
             ("1.36 s", "同样动作（0.25 rad）", MG, 3.4)]
    for txt, sub, col, t0 in cards:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 214 + (t0 - 1.8) / 0.8 * 78
        cv.text(700, y, txt, size=42, color=col, weight="bold", alpha=aa)
        cv.text(700, y + 26, sub, size=19, color=FAINT, alpha=aa)
    a4 = appear(t, 4.4, 0.8)
    cv.text(700, 500, "初值只大 2.5 倍", size=21, color=DIM, alpha=a4)
    cv.text(700, 532, "收敛时间涨 5.7 倍", size=21, color=DIM, alpha=a4)
    chip(cv, 700, 590, "力被限幅截断 → 线性最优失效", AM, appear(t, 5.2, 0.8), 20)


# ============================================================ S3
def s3(cv, t, dur):
    title(cv, "设计：一个方程解出增益")
    a = appear(t, 0.3, 0.8)
    if a > 0:
        math_(cv, 110, 208, r"A^{\top}P+PA-PBR^{-1}B^{\top}P+Q=0", 24, INK)
        cv.text(110, 236, "连续时间代数 Riccati 方程（CARE）", size=19, color=FAINT, alpha=a)
    a2 = appear(t, 1.8, 0.8)
    if a2 > 0:
        math_(cv, 110, 320, r"K=R^{-1}B^{\top}P", 26, CY)
        math_(cv, 110, 392, r"K=", 24, INK)
        math_(cv, 178, 392,
              r"\begin{bmatrix}-4.4721 & -62.8879 & -9.3145 & -16.4020\end{bmatrix}",
              22, INK)
    a3 = appear(t, 3.4, 0.8)
    if a3 > 0:
        math_(cv, 110, 480, r"\text{poles}=\{-11.96,\ -4.43,\ -0.902\pm0.47j\}", 22, AM)
        math_(cv, 110, 540, r"\zeta=\frac{0.902}{\sqrt{0.902^2+0.47^2}}=0.89", 22, AM)
        cv.text(430, 548, "快，但不过冲", size=21, color=DIM, alpha=a3)


# ============================================================ S4
def s4(cv, t, dur):
    title(cv, "增益的直觉：四句话")
    rows = [(r"K_\theta=-62.9", "摆倾 1 rad，就往同向推 63 N", CY),
            (r"K_p=-4.47", "位置几乎不管（只有角度的 1/14）", BL),
            (r"K_{\dot\theta}=-16.4", "摆正在倒，提前加大推力（阻尼）", AM),
            (r"K_{\dot p}=-9.31", "车在跑，给一点反向阻力", MG)]
    for i, (tex, txt, col) in enumerate(rows):
        aa = appear(t, 0.5 + 0.8 * i, 0.6)
        if aa <= 0:
            continue
        y = 212 + i * 92
        math_(cv, 110, y, tex, 24, col)
        cv.text(430, y, txt, size=23, color=INK, alpha=aa)
        cv.line((110, y + 16), (1180, y + 16), (60, 76, 94), width=1, alpha=0.5 * aa)
    a = appear(t, 4.2, 0.8)
    chip(cv, 110, 596, "Kθ / Kp ≈ 14 : 1 —— 只救摆，不管车", AM, a, 21)


# ============================================================ S5
def s5(cv, t, dur):
    title(cv, "能力边界：把“局部”画出来")
    a = appear(t, 0.3, 0.9)
    # basin map
    X0, Y0, BW, BH = 110, 236, 380, 300
    if a > 0:
        img = BASIN_IMG if a >= 1 else None
        cv.img.paste(BASIN_IMG, (X0 * cv.ss, Y0 * cv.ss), BASIN_IMG)
    axes(cv, X0, Y0, X0 + BW, Y0 + BH, a)
    cv.text(X0 + BW / 2, Y0 + BH + 58, "θ₀ [rad]", size=18, color=DIM, anchor="ma", alpha=a)
    cv.text(X0, Y0 - 26, "ω₀ [rad/s]", size=18, color=DIM, alpha=a)
    for th in (-1.2, -0.6, 0, 0.6, 1.2):
        xx = X0 + (th + 1.2) / 2.4 * BW
        cv.line((xx, Y0 + BH), (xx, Y0 + BH + 5), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(xx, Y0 + BH + 26, f"{th:g}", size=15, color=FAINT, anchor="ma", alpha=a)
    for w in (-4, -2, 0, 2, 4):
        yy = Y0 + BH - (w + 4) / 8 * BH
        cv.line((X0 - 5, yy), (X0, yy), (92, 110, 132), width=1.2, alpha=0.8 * a)
        cv.text(X0 + 9, yy + 6, f"{w:g}", size=15, color=FAINT, alpha=a)
    cv.text(X0 + 10, Y0 + 22, "浅色：能救回　深色：失败", size=16, color=FAINT, alpha=a)

    # phase trajectories
    ap = appear(t, 1.6, 0.8)
    if ap > 0:
        prog = min(1.0, max(0.0, (t - 1.6) / 3.2))
        for (th0, w0), pts, good in PHASE:
            n = max(2, int(len(pts) * prog))
            col = (47, 211, 192) if good else (224, 97, 158)
            px = [X0 + (p[0] + 1.2) / 2.4 * BW for p in pts[:n]]
            py = [Y0 + BH - (p[1] + 4) / 8 * BH for p in pts[:n]]
            for i in range(1, n):
                if not (X0 - 2 <= px[i] <= X0 + BW + 2 and Y0 - 2 <= py[i] <= Y0 + BH + 2):
                    continue
                if not (X0 - 2 <= px[i - 1] <= X0 + BW + 2 and Y0 - 2 <= py[i - 1] <= Y0 + BH + 2):
                    continue
                cv.line((px[i - 1], py[i - 1]), (px[i], py[i]), col, width=2.4, alpha=ap)
            # hollow ring at the initial condition
            ix = X0 + (pts[0][0] + 1.2) / 2.4 * BW
            iy = Y0 + BH - (pts[0][1] + 4) / 8 * BH
            if X0 <= ix <= X0 + BW and Y0 <= iy <= Y0 + BH:
                cv.circle(ix, iy, 4.6, col, width=1.6, alpha=0.95 * ap)
            if 0 < n < len(pts):
                hx, hy = px[-1], py[-1]
                if X0 <= hx <= X0 + BW and Y0 <= hy <= Y0 + BH:
                    cv.circle(hx, hy, 5.0, col, width=0, alpha=ap, fill_alpha=ap)
                    cv.circle(hx, hy, 9.0, col, width=1.2, alpha=0.5 * ap)
        cv.text(X0 + 10, Y0 + BH - 12, "细线：相轨迹（青=收敛，粉=发散）", size=15,
                color=FAINT, alpha=ap)

    items = [("51.8%", "扫描区域内的可恢复占比", CY, 2.4),
             ("0.64 rad", "静止释放的极限倾角（≈37°）", AM, 3.2),
             ("点对称", "旋转 180° 与自身重合", DIM, 4.0)]
    for txt, sub, col, t0 in items:
        aa = appear(t, t0, 0.6)
        if aa <= 0:
            continue
        y = 244 + (t0 - 2.4) / 0.8 * 92
        cv.text(556, y, txt, size=32, color=col, weight="bold", alpha=aa)
        cv.text(556, y + 26, sub, size=19, color=FAINT, alpha=aa)
    a5 = appear(t, 5.0, 0.8)
    cv.text(556, 520, "形状是一条斜带：", size=21, color=DIM, alpha=a5)
    cv.text(556, 552, "朝直立方向转的，容错大得多", size=21, color=DIM, alpha=a5)


# ============================================================ S6
def s6(cv, t, dur):
    title(cv, "全局失败：25% 与 82%")
    a = appear(t, 0.3, 0.8)
    cv.text(110, 214, "25%", size=64, color=MG, weight="bold", alpha=a)
    cv.text(110, 246, "60 组随机初值的成功率", size=20, color=FAINT, alpha=a)
    a2 = appear(t, 1.6, 0.8)
    cv.text(110, 320, "82%", size=64, color=AM, weight="bold", alpha=a2)
    cv.text(110, 352, "回合把小车推到导轨尽头", size=20, color=FAINT, alpha=a2)

    # failure animation from hanging
    a3 = appear(t, 2.6, 0.6)
    if a3 > 0:
        idx = min(int((t - 2.6) / 0.05), len(FAIL["t"]) - 1)
        th = FAIL["th"][idx]; pp = FAIL["p"][idx]; uu = FAIL["u"][idx]
        cartpole(cv, 880, 470, th, p=pp, u=uu, a=a3, scale=118, rail=2.4)
        cv.text(880, 214, "从下垂出发（θ₀ ≈ π），整条导轨 ±2.4 m", size=19, color=FAINT,
                anchor="ma", alpha=a3)
        cv.text(880, 528, f"p = {pp:+.2f} m    u = {uu:+.1f} N", size=20, color=DIM,
                anchor="ma", alpha=a3)
        if abs(pp) > 2.35:
            cv.text(880, 566, "撞墙：位置钳位、速度清零", size=20, color=MG, anchor="ma", alpha=a3)
    a4 = appear(t, 6.4, 0.8)
    cv.text(110, 470, "失败不是“慢慢变差”：", size=22, color=INK, alpha=a4)
    cv.text(110, 504, "力一上来就饱和，车被推到墙上，", size=22, color=INK, alpha=a4)
    cv.text(110, 538, "摆还在原地打转。", size=22, color=INK, alpha=a4)


# ============================================================ S7
def s7(cv, t, dur):
    title(cv, "为什么：地图错了，力也不够")
    a = appear(t, 0.3, 0.8)
    cv.rect(110, 190, 600, 320, MG, width=1.2, alpha=0.7 * a, fill_alpha=0.06 * a, radius=8)
    cv.text(136, 222, "① 线性化只在 ±0.2 rad 内可信", size=23, color=MG, alpha=a)
    math_(cv, 136, 268, r"|\theta|>0.2\ \Rightarrow\ 1\,\mathrm{s}\ \text{后误差}\ 35\%", 21, INK)

    a2 = appear(t, 2.2, 0.8)
    cv.rect(110, 356, 600, 486, AM, width=1.2, alpha=0.7 * a2, fill_alpha=0.06 * a2, radius=8)
    cv.text(136, 388, "② 执行器只有 12 N", size=23, color=AM, alpha=a2)
    math_(cv, 136, 434, r"K_\theta\cdot\theta=62.9\times\pi\approx197\ \mathrm{N}", 21, INK)

    a3 = appear(t, 4.2, 0.9)
    # demand vs limit
    X0, Y0, X1, Y1 = 700, 214, 1140, 430
    axes(cv, X0, Y0, X1, Y1, a3, "θ [rad]", "需求 [N]")
    sx = lambda th: X0 + (th + 0.1) / 3.3 * (X1 - X0)
    sy = lambda v: Y1 - v / 210.0 * (Y1 - Y0)
    for v in (0, 50, 100, 150, 200):
        cv.line((X0 - 5, sy(v)), (X0, sy(v)), (92, 110, 132), width=1.2, alpha=0.8 * a3)
        cv.text(X0 + 9, sy(v) + 6, str(v), size=15, color=FAINT, alpha=a3)
    for th in (0, 1, 2, 3):
        cv.line((sx(th), Y1), (sx(th), Y1 + 5), (92, 110, 132), width=1.2, alpha=0.8 * a3)
        cv.text(sx(th), Y1 + 26, f"{th}", size=15, color=FAINT, anchor="ma", alpha=a3)
    cv.line((sx(-0.1), sy(12)), (sx(3.2), sy(12)), MG, width=2, alpha=0.9 * a3, dash=6)
    cv.text(sx(2.1), sy(12) - 10, "限幅 12 N", size=16, color=MG, alpha=a3)
    curvy(cv, [(sx(th), sy(min(62.9 * th, 210))) for th in np.linspace(0, 3.2, 80)],
          AM, 2.2, a3)
    a4 = appear(t, 6.0, 0.8)
    cv.text(700, 500, "角度一到 0.2 rad，需求就超过限幅", size=20, color=DIM, alpha=a4)
    cv.text(700, 532, "之后闭环退化成“恒定 12 N 开环推力”", size=20, color=DIM, alpha=a4)


# ============================================================ S8
def s8(cv, t, dur):
    title(cv, "本讲小结")
    items = [("局部极好", "0.1 rad → 0.24 s 进入 ±0.02 rad"),
             ("吸引域是斜带", "静止释放极限 0.64 rad，面积占 51.8%"),
             ("全局失败", "25% 成功、82% 撞墙"),
             ("两个原因", "线性化失效 + 执行器饱和"),
             ("结论", "局部最优 ≠ 全局可行")]
    for i, (k, v) in enumerate(items):
        aa = appear(t, 0.5 + 0.72 * i, 0.6)
        if aa <= 0:
            continue
        y = 200 + i * 62
        cv.circle(104, y - 8, 7, CY, width=1.6, alpha=aa, fill_alpha=0.15 * aa)
        cv.line((100, y - 8), (103, y - 5), CY, width=1.6, alpha=aa)
        cv.line((103, y - 5), (109, y - 12), CY, width=1.6, alpha=aa)
        cv.text(130, y, k, size=24, color=CY, alpha=aa, weight="bold")
        cv.text(300, y, v, size=22, color=INK, alpha=aa)
    a = appear(t, 4.4, 0.8)
    cv.text(130, 552, "下一讲：部分反馈线性化 —— 把“力”变成“加速度”", size=25, color=AM, alpha=a)
    cv.text(130, 596, "作业：扫一遍吸引域，并解释 $Q/R$ 对它的影响", size=19, color=FAINT, alpha=a)


SHOTS = [
    dict(name="title", say=[
        "上一讲我们把机器写成了方程。这一讲先做最朴素的一件事：把它扶住。",
        "方法是最经典的线性二次型最优控制，简称 LQR。",
        "但这一讲的重点不是它多好，而是它在哪里不行。",
    ], draw=s1, min_dur=12.0),
    dict(name="local", say=[
        "先看它最擅长的地方：初值只在直立附近。",
        {"say": "$0.1$ 弧度的初始倾角，$0.24$ 秒后就进入 $\\pm0.02$ 弧度，峰值力只有 $6.29$ 牛顿。",
         "tts": "零点一弧度的初始倾角，零点二四秒后就进入正负零点零二弧度，峰值力只有六点二九牛顿。"},
        {"say": "把初值放大到 $0.25$ 弧度，力开始顶到 $12$ 牛顿的上限。",
         "tts": "把初值放大到零点二五弧度，力开始顶到十二牛顿的上限。"},
        {"say": "收敛时间从 $0.24$ 秒涨到 $1.36$ 秒。执行器饱和，是第一个真正的非线性。",
         "tts": "收敛时间从零点二四秒涨到一点三六秒。执行器饱和，是第一个真正的非线性。"},
    ], draw=s2, min_dur=18.0),
    dict(name="design", say=[
        "设计只需要解一个方程：连续时间代数 Riccati 方程。",
        {"say": "把 $A$、$B$ 和权重 $Q$、$R$ 代进去，解出 $P$，就得到最优增益 $K$。",
         "tts": "把 A、B 和权重 Q、R 代进去，解出 P，就得到最优增益 K。"},
        {"say": "四个闭环极点全在左半平面，阻尼比 $0.89$：快，但不过冲。",
         "tts": "四个闭环极点全在左半平面，阻尼比零点八九：快，但不过冲。"},
    ], draw=s3, min_dur=16.0),
    dict(name="intuition", say=[
        "这四行讲清了增益的直觉。",
        {"say": "$K_\\theta$ 是 $-62.9$：摆每倾斜 $1$ 弧度，就往同向推 $63$ 牛顿。",
         "tts": "K 西塔是负六十二点九：摆每倾斜一弧度，就往同向推六十三牛顿。"},
        {"say": "$K_p$ 只有 $-4.47$：位置几乎不管，比例大约是 $14$ 比 $1$。",
         "tts": "K p 只有负四点四七：位置几乎不管，比例大约是十四比一。"},
        "后面会看到，这条“不管位置”的设定，正是抗风差的根源。",
    ], draw=s4, min_dur=17.0),
    dict(name="basin", say=[
        "现在画它的能力边界：吸引域。",
        "横轴是初始倾角，纵轴是初始角速度。浅色区域能救回来，深色救不回来。",
        {"say": "静止释放时，极限倾角是 $0.64$ 弧度，约 $37$ 度。",
         "tts": "静止释放时，极限倾角是零点六四弧度，约三十七度。"},
        "形状不是圆，而是一条斜带：如果摆正朝直立方向转，容错就大得多。",
    ], draw=s5, min_dur=19.0),
    dict(name="global", say=[
        {"say": "换到真实任务：$60$ 组随机初值，成功率只有 $25\\%$。",
         "tts": "换到真实任务：六十组随机初值，成功率只有百分之二十五。"},
        {"say": "$82\\%$ 的回合把小车推到导轨尽头。",
         "tts": "百分之八十二的回合把小车推到导轨尽头。"},
        "失败不是慢慢发生的：力一上来就饱和，车被推到墙上，摆还在原地打转。",
    ], draw=s6, min_dur=18.0),
    dict(name="why", say=[
        "为什么会这样？两个原因。",
        {"say": "第一，线性化这张地图只在 $\\pm0.2$ 弧度内可信，$1$ 秒后误差就到 $35\\%$。",
         "tts": "第一，线性化这张地图只在正负零点二弧度内可信，一秒后误差就到百分之三十五。"},
        {"say": "第二，$K_\\theta$ 乘上大角度会给出上百牛顿的需求，但执行器只有 $12$ 牛顿。",
         "tts": "第二，K 西塔乘上大角度会给出上百牛顿的需求，但执行器只有十二牛顿。"},
        "在饱和的世界里，加增益不解决问题。",
    ], draw=s7, min_dur=20.0),
    dict(name="summary", say=[
        {"say": "所以结论是：局部最优，不等于全局可行。",
         "tts": "所以结论是：局部最优，不等于全局可行。"},
        "我们需要一个不依赖小角假设的全局控制律：能量整定。",
        "还需要把两段局部真理缝起来：切换监督器。",
        "下一讲先换一个坐标：把力变成加速度。",
    ], draw=s8, min_dur=19.0),
]

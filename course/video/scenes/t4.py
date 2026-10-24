"""Transformer 教程 · 第四集：前向逐算子（公式 ↔ 代码 ↔ 数值）—— 10 个镜头。

代码块不是手抄的：从 tutorial/code/tinytf_numpy.py 里按锚点切出来，
保证屏幕上显示的就是真正跑的那几行。
"""
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
FW = D["forward"]

_LINES = open(SRC, encoding="utf-8").read().split("\n")


def src(anchor, n, dedent=None):
    """按锚点从源码切片；找不到就返回提示，避免静默错误。"""
    for i, ln in enumerate(_LINES):
        if anchor in ln:
            block = _LINES[i:i + n]
            if dedent is None:
                dedent = len(block[0]) - len(block[0].lstrip())
            return [b[dedent:] if len(b) >= dedent else b for b in block]
    return [f"# anchor not found: {anchor}"]


def pair(cv, t, tex, snip, shape="", num="", t0=0.4, size=19, csize=13):
    """左公式、右代码，底部形状/数值。"""
    a1 = appear(t, t0, 0.6)
    if a1 > 0:
        for i, tx in enumerate(tex):
            math_(cv, 100, 196 + i * 62, tx, size, INK)
    a2 = appear(t, t0 + 0.9, 0.6)
    if a2 > 0:
        code(cv, 690, 190, snip, size=csize, alpha=a2)
    a3 = appear(t, t0 + 1.8, 0.6)
    if a3 > 0 and shape:
        cv.text(100, 470, shape, size=17, color=FAINT, alpha=a3)
    a4 = appear(t, t0 + 2.4, 0.6)
    if a4 > 0 and num:
        cv.text(100, 508, num, size=18, color=CY, alpha=a4)


def s1(cv, t, dur):
    cv.text(640, 240, "Transformer 从零实现　·　第四集", size=24, color=CY, anchor="ma",
            alpha=appear(t, 0.2, 0.9))
    cv.text(640, 334, "前向：逐算子", size=62, color=INK, anchor="ma", weight="bold",
            alpha=appear(t, 0.9, 0.9))
    cv.text(640, 394, "每个公式，配它真正跑的那几行代码", size=29, color=DIM, anchor="ma",
            alpha=appear(t, 1.7, 0.9))
    cv.line((470, 434), (810, 434), CY, width=2, alpha=0.5 * appear(t, 2.3, 0.8))
    if appear(t, 2.6, 0.9) > 0:
        math_(cv, 640, 500, r"x\ \to\ z\ \to\ h\ \to\ q,k,v\ \to\ a\ \to\ y"
                            r"\ \to\ z'\ \to\ u", 24, DIM, anchor="mc")
    cv.text(640, 600, "代码：tutorial/code/tinytf_numpy.py（截图内容由源码实时切片）", size=18,
            color=FAINT, anchor="ma", alpha=appear(t, 3.2, 0.8))


def s2(cv, t, dur):
    title(cv, "9 个算子，一条流水线")
    a = appear(t, 0.3, 0.8)
    ops = [("① 嵌入 + 位置编码", "z = xW^T + b + pos"),
           ("② LayerNorm", "h = (z−μ)/σ·γ + β"),
           ("③ QKV 投影 + 分头", "q,k,v = hW_qkv^T + b"),
           ("④ 缩放点积 + 掩码", "s = qk^T/√dh + M"),
           ("⑤ softmax", "a = softmax(s)"),
           ("⑥ 加权求和 + 合并 + 投影", "y = (a·v)W_o^T"),
           ("⑦ 残差 ⊕", "z ← z + y"),
           ("⑧ LN2 + FFN + GELU", "z ← z + W₂·GELU(W₁h₂)"),
           ("⑨ 头 + tanh", "u = 12·tanh(w^T·LN(z_T))")]
    for i, (k, v) in enumerate(ops):
        aa = appear(t, 0.5 + 0.34 * i, 0.5)
        if aa <= 0:
            continue
        y = 176 + i * 51
        cv.text(110, y, k, size=19, color=INK, alpha=aa)
        cv.text(430, y, v, size=17, color=DIM, alpha=aa)
    a2 = appear(t, 4.0, 0.8)
    cv.rect(760, 200, 1180, 420, CY, width=1.2, alpha=0.7 * a2, fill_alpha=0.05 * a2, radius=8)
    cv.text(782, 232, "为什么按算子讲", size=18, color=CY, alpha=a2)
    cv.text(782, 270, "每个算子都是", size=17, color=INK, alpha=a2)
    cv.text(782, 300, "「一个前向 + 一个反向」", size=17, color=INK, alpha=a2)
    cv.text(782, 340, "独立可测：给它随机输入，", size=17, color=DIM, alpha=a2)
    cv.text(782, 368, "数值梯度一验就知道对不对", size=17, color=DIM, alpha=a2)
    a3 = appear(t, 5.2, 0.8)
    cv.text(110, 640, "下面每个算子：左边公式，右边源码，底部是真实权重下的形状与数值", size=17,
            color=FAINT, alpha=a3)


def s3(cv, t, dur):
    title(cv, "① 嵌入 + 位置编码")
    pair(cv, t,
         [r"z = xW_{\mathrm{in}}^{\top}+b_{\mathrm{in}}+\mathrm{pos}",
          r"x\in\mathbb{R}^{B\times T\times F},\ \ z\in\mathbb{R}^{B\times T\times d}"],
         src('z = x @ p["W_in"].T + p["b_in"] + p["pos"]', 1),
         "形状：x (B,T,F) → z (B,T,d)；pos 是可学习的 (T,d)，只有 24 个位置",
         f"实测：‖z‖₂ = {FW['z_norm']:.2f}，z 的前 6 维 {[round(v,2) for v in FW['z_last']]}",
         size=20)
    a = appear(t, 4.4, 0.8)
    cv.text(100, 570, "为什么不用正弦位置编码：T=24 固定且很短，直接学更简单", size=17,
            color=DIM, alpha=a)
    a2 = appear(t, 5.4, 0.8)
    cv.text(100, 606, "参数量：6×32+32 = 224，加 pos 24×32 = 768", size=17, color=DIM, alpha=a2)


def s4(cv, t, dur):
    title(cv, "② LayerNorm（前向）")
    pair(cv, t,
         [r"\mu=\tfrac1d\textstyle\sum_i z_i,\quad \sigma^2=\tfrac1d\textstyle\sum_i(z_i-\mu)^2",
          r"\hat z=\frac{z-\mu}{\sqrt{\sigma^2+\varepsilon}},\quad h=\hat z\odot\gamma+\beta"],
         src("def ln_forward(x, g, b, eps=1e-5):", 7),
         "对最后一维（d=32）归一化；γ、β 各 32 个参数",
         f"实测：μ = {FW['ln1']['mu']:+.3f}，σ² = {FW['ln1']['var']:.3f}",
         size=18, csize=13)
    a = appear(t, 4.6, 0.8)
    ratio = 1 / math.sqrt(FW["ln1"]["var"] + 1e-5)
    cv.text(100, 560, f"σ² 只有 {FW['ln1']['var']:.3f} → 放大 1/σ = {ratio:.2f} 倍", size=18,
            color=AM, alpha=a)
    a2 = appear(t, 5.6, 0.8)
    cv.text(100, 600, "这一步很关键：否则注意力分数会挤在 0 附近，softmax 几乎均匀", size=17,
            color=DIM, alpha=a2)


def s5(cv, t, dur):
    title(cv, "③ QKV 投影与分头")
    pair(cv, t,
         [r"[Q;K;V]=hW_{\mathrm{qkv}}^{\top}+b_{\mathrm{qkv}}\in\mathbb{R}^{3d}",
          r"(B,T,3d)\ \to\ (B,T,H,d_h)\ \to\ (B,H,T,d_h)"],
         src("qkv = h @ p[k + \"W_qkv\"].T + p[k + \"b_qkv\"]", 6),
         "W_qkv: (3d, d) = (96, 32)，参数量 3,168；H=4，d_h=8",
         f"实测：‖q‖={FW['qkv_last']['q_norm']:.2f}  ‖k‖={FW['qkv_last']['k_norm']:.2f}  "
         f"‖v‖={FW['qkv_last']['v_norm']:.2f}",
         size=18, csize=12)
    a = appear(t, 5.0, 0.8)
    chip(cv, 100, 566, "易错点：先 reshape 再 transpose，否则“头”切在时间维上", MG, a, 18)
    a2 = appear(t, 6.0, 0.8)
    cv.text(100, 612, "3 个投影拼成一个矩阵乘，是性能上的小技巧（一次 GEMM 而不是 3 次）", size=16,
            color=DIM, alpha=a2)


def s6(cv, t, dur):
    title(cv, "④ 缩放点积 + 因果掩码")
    pair(cv, t,
         [r"s_{ts}=\frac{q_t^{\top}k_s}{\sqrt{d_h}}+M_{ts},\qquad "
          r"M_{ts}=\begin{cases}0,& s\le t\\-\infty,& s>t\end{cases}"],
         src("s = np.einsum(\"bhtd,bhsd->bhts\", q, kk) / np.sqrt(nh)", 4),
         "形状：(B,H,T,T) = (B,4,24,24)；上三角置 −1e9",
         "为什么除以 √d_h：Var(q^Tk)=d_h，不缩放 softmax 会饱和",
         size=18, csize=12)
    a = appear(t, 4.6, 0.8)
    if a > 0:
        n = 8
        for i in range(n):
            for j in range(n):
                x, y = 700 + j * 26, 330 + i * 26
                ok = j <= i
                cv.rect(x, y, x + 22, y + 22, CY if ok else (40, 50, 62), width=0,
                        alpha=(0.5 if ok else 0.85) * a, fill_alpha=(0.3 if ok else 0.45) * a)
        cv.text(700, 316, "可看（下三角） / 不可看（上三角）", size=15, color=FAINT, alpha=a)
    a2 = appear(t, 5.6, 0.8)
    cv.text(100, 600, "掩码位置后续 softmax 权重为 0 → 反传时梯度自动为 0，不用额外清零", size=16,
            color=DIM, alpha=a2)


def s7(cv, t, dur):
    title(cv, "⑤ softmax（含数值稳定写法）")
    pair(cv, t,
         [r"a_{ts}=\frac{e^{s_{ts}}}{\sum_{s'\le t}e^{s_{ts'}}}=\frac{e^{s_{ts}-m}}{\sum e^{s_{ts'}-m}}",
          r"m=\max_{s'\le t}s_{ts'}"],
         src("def softmax(x, axis=-1):", 4),
         "减去 m 不改变结果、但避免 exp 上溢（−1e9 的掩码不会变成 inf/nan）",
         f"实测：行和 = {FW['attn']['row_sum']:.3f}；前 6 位 "
         f"{[round(v,3) for v in FW['attn']['head0'][:6]]}",
         size=18, csize=13)
    a = appear(t, 4.8, 0.8)
    cv.text(100, 566, "对比：均匀值 1/24 = 0.0417 —— 这个头几乎在平均历史", size=18, color=AM, alpha=a)
    a2 = appear(t, 5.8, 0.8)
    cv.text(100, 606, "第 2 层不同：峰值可达 0.22，稳摆时锁在最近第 2 步", size=16, color=DIM, alpha=a2)


def s8(cv, t, dur):
    title(cv, "⑥ 加权求和 → 合并 → 输出投影")
    pair(cv, t,
         [r"y^{(h)}_t=\sum_{s\le t}a^{(h)}_{ts}v^{(h)}_s,\qquad y_t=\mathrm{Concat}_h\,y^{(h)}_t",
          r"z'\leftarrow z+\big(yW_o^{\top}+b_o\big)"],
         src("y = np.einsum(\"bhts,bhsd->bhtd\", a, v)", 5),
         "形状：(B,H,T,d_h) → (B,T,d) → (B,T,d)；W_o: (d,d) 参数量 1,056",
         f"实测：‖z‖={FW['after_attn']['resid_norm']:.2f}，‖attn_out‖="
         f"{FW['after_attn']['attn_out_norm']:.2f}，合并后 {FW['after_attn']['z_norm']:.2f}",
         size=17, csize=12)
    a = appear(t, 5.0, 0.8)
    cv.text(100, 570, "注意合并不改变每个 token 自身的位置，只把 4 个头“接起来”", size=17,
            color=DIM, alpha=a)
    a2 = appear(t, 6.0, 0.8)
    cv.text(100, 608, "注意力分支只贡献约 1/3 的幅度 —— 所以训练初期不容易崩", size=17,
            color=CY, alpha=a2)


def s9(cv, t, dur):
    title(cv, "⑦⑧ 残差、LN2、FFN、GELU")
    pair(cv, t,
         [r"z_1=z+\text{Attn}(z)",
          r"z\leftarrow z_1+W_2\,\mathrm{GELU}(W_1\mathrm{LN}_2(z_1)+b_1)+b_2"],
         src("h2, c_ln2 = ln_forward(z1, p[k + \"ln2_g\"], p[k + \"ln2_b\"])", 4),
         "FFN 32→64→32，GELU 逐元素；W₁ 2,112 / W₂ 2,080 参数",
         f"实测：64 个隐单元中 {FW['ffn']['neg_count']} 个输入为负，被 GELU 软压到 (−0.17,0)",
         size=18, csize=12)
    a = appear(t, 4.8, 0.9)
    if a > 0:
        X0, Y0, X1, Y1 = 700, 400, 1120, 560
        axes(cv, X0, Y0, X1, Y1, a, "x", "GELU(x)")
        sx = lambda v: X0 + (v + 3) / 6 * (X1 - X0)
        sy = lambda v: (Y0 + Y1) / 2 - v / 3 * (Y1 - Y0) / 2
        curvy(cv, [(sx(v), sy(0.5 * v * (1 + math.erf(v / math.sqrt(2)))))
                   for v in np.linspace(-3, 3, 160)], GR, 2.2, a)
        cv.line((sx(0), Y0), (sx(0), Y1), (90, 108, 130), width=1, alpha=0.5 * a, dash=5)
    a2 = appear(t, 6.0, 0.8)
    cv.text(100, 620, "与 ReLU 的差别：负半轴不是硬截断，而是软门控（可导、不会死神经元）", size=16,
            color=DIM, alpha=a2)


def s10(cv, t, dur):
    title(cv, "⑨ 输出头：取最后 token")
    pair(cv, t,
         [r"z_T=\mathrm{LN}(z_{:,T})\in\mathbb{R}^{B\times d}",
          r"u=u_{\max}\tanh\big(w^{\top}z_T+b\big),\quad u_{\max}=12"],
         src("zl, c_lnf = ln_forward(z[:, -1], p[\"ln_g\"], p[\"ln_b\"])", 3),
         "只取最后一个 token：因果掩码保证它“看过”整段历史",
         f"实测：u = {FW['out']['u']:.3f} N（归一化 {FW['out']['u_norm']:.3f}）",
         size=18, csize=13)
    a = appear(t, 4.6, 0.8)
    cv.text(100, 566, "为什么是 tanh 而非线性：天然限幅 + 平滑；代价是接近 ±12 时梯度消失",
            size=18, color=AM, alpha=a)
    a2 = appear(t, 5.6, 0.8)
    chip(cv, 100, 614, "这 9 个算子串起来，就是 18,177 个参数的全部前向", CY, a2, 18)


SHOTS = [
    dict(name="title", say=[
        "这一集把前向传播按算子拆开：每个公式，配它真正跑的那几行代码。",
        "屏幕上的代码不是手抄的，是从源码里实时切出来的。",
    ], draw=s1, min_dur=11.0),
    dict(name="overview", say=[
        "整条流水线有 9 个算子。",
        "嵌入、归一化、QKV、缩放点积加掩码、softmax、加权求和加投影、残差、前馈、输出头。",
        "这样拆的好处是：每个算子都能单独测——给它随机输入，用数值梯度一验就知道对不对。",
    ], draw=s2, min_dur=18.0),
    dict(name="embed", say=[
        "第一个：嵌入加位置编码。",
        {"say": "6 维特征线性投影到 $32$ 维，再加一个可学习的位置向量。",
         "tts": "六维特征线性投影到三十二维，再加一个可学习的位置向量。"},
        {"say": "实测这一帧的范数是 $2.67$。",
         "tts": "实测这一帧的范数是二点六七。"},
    ], draw=s3, min_dur=16.0),
    dict(name="ln", say=[
        "第二个：LayerNorm。",
        "对 32 维做归一化：减均值、除标准差，再乘 gamma 加 beta。",
        {"say": "注意实测方差只有 $0.223$，所以这一步把信号放大了 $2.1$ 倍。",
         "tts": "注意实测方差只有零点二二三，所以这一步把信号放大了二点一倍。"},
        "不做这一步，注意力分数会挤在零附近，softmax 几乎均匀。",
    ], draw=s4, min_dur=18.0),
    dict(name="qkv", say=[
        "第 3 个：QKV 投影与分头。",
        "一次矩阵乘得到三份，再切成 4 个头、每头 8 维。",
        "屏幕右边那 3 行代码就是关键：先 reshape 成 B、T、H、dh，再 transpose。",
        "顺序写错的话，头会切在时间维上——后面的注意力就完全错了。",
    ], draw=s5, min_dur=19.0),
    dict(name="score", say=[
        "第 4 个：缩放点积与因果掩码。",
        {"say": "点积除以根号 dh，再把上三角置为负一亿。",
         "tts": "点积除以根号 d 下标 h，再把上三角置为负一亿。"},
        "除以根号 dh 的理由是点积方差等于 dh；掩码的理由是只许看过去。",
        "掩码位置在 softmax 之后权重为 0，所以反传时梯度自动为 0。",
    ], draw=s6, min_dur=18.0),
    dict(name="softmax", say=[
        "第 5 个：softmax。",
        "代码里先减最大值再取指数——这不改变结果，但避免上溢，尤其是掩码那一亿的负值。",
        {"say": "实测这一行加起来是一，前 6 个位置都在 $0.04$ 附近，而均匀值是 0.0417。",
         "tts": "实测这一行加起来是一，前六个位置都在零点零四附近，而均匀值是零点零四一七。"},
    ], draw=s7, min_dur=16.0),
    dict(name="av", say=[
        "第 6 个：加权求和、合并多头、输出投影。",
        "加权求和是 einsum 一行；合并是把 4 个头接回 32 维；投影是又一个矩阵乘。",
        {"say": "实测残差主干范数 $2.67$，注意力分支 $0.88$，合并后 $2.79$。",
         "tts": "实测残差主干范数二点六七，注意力分支零点八八，合并后二点七九。"},
    ], draw=s8, min_dur=17.0),
    dict(name="ffn", say=[
        "第七、第 8 个：残差、第二个归一化、前馈网络。",
        {"say": "64 个隐单元里，有 34 个输入为负。",
         "tts": "六十四个隐单元里，有三十四个输入为负。"},
        "GELU 把它们软压到零附近，而不是像 ReLU 那样直接截断——这保证了处处可导。",
    ], draw=s9, min_dur=17.0),
    dict(name="head", say=[
        "最后第 9 个：只取最后一个 token，做归一化、线性、再乘 12 取 tanh。",
        {"say": "输出是 $4.153$ 牛顿。",
         "tts": "输出是四点一五三牛顿。"},
        "因果掩码保证最后一个 token 已经“看过”整段历史。",
        "这 9 个算子串起来，就是一万八千一百七十七个参数的全部前向。",
    ], draw=s10, min_dur=17.0),
]

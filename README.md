# 小车倒立摆：从拉格朗日到 Transformer

一个完整的控制 × 学习项目：同一台小车倒立摆，先用**经典非线性控制**从零设计控制器，
再用**数据**学一个 18k 参数的 Transformer 控制器，最后用**强化学习**微调它。
全套材料（交互式网页、可复现代码、10 讲讲义与视频）都在这个仓库里。

```
建模 → LQR（划边界）→ 反馈线性化 → 能量整定 → 滑模 → 切换 → 数据 → Transformer → DAgger → PPO
```

## 项目包含什么

| 部分 | 内容 | 入口 |
|---|---|---|
| **① 交互式技术报告** | 单文件 HTML：原理推导 + 四/五个控制器同台动画、相图、注意力热图，浏览器实时仿真 | `index.html` |
| **② 复现管线** | Python：被控对象、控制器、数据生成、训练、int8 导出、统一评测 | `build/*.py` |
| **③ 课程（10 讲 + 附录）** | 讲义（LaTeX/ctex）+ 11 个视频（LaTeX 公式、中文旁白、配乐、烧录字幕） | `course/` |
| **④ Transformer 实现教程** | 讲义 + 3 个视频 + **纯 numpy 手写反向**与 torch 双实现（梯度差 3e-14） | `tutorial/` |

## 结果（60 组随机初值，12 s 闭环，统一协议）

| 控制器 | 成功率 | 摆起时间 | 稳态 \|θ\| | 力 RMS | 最大单步 Δu | +4N 侧风 |
|---|---|---|---|---|---|---|
| LQR（线性基线） | 25% | — | 1.62 | 8.80 | 1.59 | 25% |
| 能量整定 + LQR | 100% | 2.84 s | 0.0004 | 2.42 | 10.05 | 100% |
| 能量整定 + 滑模 | 100% | 2.82 s | 0.0001 | 2.63 | 9.93 | 100% |
| Transformer（模仿） | 100% | 2.74 s | 0.0012 | 2.39 | 5.36 | 80% |
| **Transformer + RL + DR** | 100% | **2.31 s** | 0.0010 | **2.12** | **3.74** | **100%** |

要点：LQR 的能力边界只有 0.64 rad；能量法会甩不会停；滑模用 7% 的力矩换来 2.3 倍的抗扰；
Transformer 把硬切换学成了连续映射（Δu 减半）；RL 在训练目标上超过专家 22%，但在分布外崩到 3%，
域随机化把它修回 100%。

## 目录结构

```
.
├── index.html                 ← 交互式技术报告（自包含，双击即开）
├── build/                     ← 复现管线
│   ├── plant.py                  拉格朗日建模 + RK4
│   ├── controllers.py            LQR / 能量整定 / 滑模 / 切换
│   ├── dataset.py  train.py      数据生成 + BC/DAgger 训练
│   ├── model.py  export.py       因果 Transformer + int8 导出
│   ├── rl.py                      PPO 微调 + 域随机化
│   ├── eval_all.py               统一评测
│   ├── sweep.py  make_figs.py    参数搜索 / 讲义插图
│   └── ch0*_experiments.py       各讲实验（可重跑）
├── tutorial/                  ← Transformer 从零实现（独立教程）
│   ├── latex/tutorial.tex        约 30 页讲义
│   ├── code/tinytf_numpy.py      纯 numpy 前向 + 手写反向
│   ├── code/tinytf_torch.py      torch 等价实现 + 权重互转
│   ├── code/parity_test.py       与 autograd / 有限差分对齐
│   └── out/                      3 个视频 + 4 张图
└── course/
    ├── README.md               ← 课程总纲（10 讲 + 结课项目）
    ├── latex/                  ← 讲义（xelatex + ctex）
    ├── video/                  ← 视频管线（场景 / 渲染 / 配音 / 字幕）
    └── out/                    ← 11 个成片 mp4 + srt
```

## 快速开始

**看报告**：直接打开 `index.html`（零依赖，无需服务器）。

**跑代码**（Python 3.9+）：

```bash
pip install numpy scipy torch pillow av edge-tts
cd build
python3 controllers.py       # 打印 LQR 增益与闭环极点
python3 ch02_experiments.py  # 吸引域扫描（4941 点）
python3 train.py             # BC + DAgger（CPU 约 5 分钟）
python3 rl.py                # PPO 微调（约 9 分钟）
python3 eval_all.py          # 五控制器统一评测
```

**编讲义**：

```bash
cd course/latex && xelatex main.tex && xelatex main.tex   # 需要 ctex 宏包
```

**渲染视频**（任选一讲）：

```bash
cd course/video && python3 make_ch01.py    # 渲染 + 配音 + 配乐 + 烧字幕
```

## 工具链说明

- **公式**：讲义用真 LaTeX；视频用 **KaTeX 本地化**（`course/video/assets/katex/`，离线）渲染成透明 PNG 再合成。
- **画面**：Pillow 逐帧绘制 + 2× 超采样 + PyAV 编码（H.264/AAC）；本机没有 ffmpeg/manim，
  其中 manim 在 macOS 需要 `pycairo`，而它没有 macOS wheel，故使用等价的自研渲染器。
- **配音**：`edge-tts`（`zh-CN-XiaoxiaoNeural`）逐句合成，字幕时间轴由语音时长反推。
- **质检**：逐帧文本重叠检测（0 处）、字形覆盖检查（`check_math.py`，0 处缺字）、
  音轨唯一性校验（11 条互不相同）。

## 视频索引

| # | 标题 | 时长 |
|---|---|---|
| 1 | 建模：把一台机器写成方程 | 3:54 |
| 2 | 线性基线：LQR 与它的能力边界 | 3:11 |
| 3 | 部分反馈线性化：把力变成加速度 | 2:05 |
| 4 | 能量整定：把摆起来变成标量目标 | 2:11 |
| 5 | 滑模控制：把鲁棒性做进结构 | 2:00 |
| 6 | 切换与监督：把两种局部真理缝起来 | 1:32 |
| 7 | 数据与模仿：把控制器变成函数逼近问题 | 2:33 |
| 8 | 因果 Transformer：从软检索到 int8 部署 | 2:28 |
| 9 | DAgger 与评测：把 T² 降成 T | 2:23 |
| 10 | 策略梯度、PPO 与域随机化 | 3:04 |
| A | 附录：Transformer 前向、反向与注意力解剖 | 3:17 |

合计约 28.7 分钟。

### Transformer 实现教程（`tutorial/`）

| # | 标题 | 时长 |
|---|---|---|
| ① | 为什么长这样：RNN 硬伤、检索视角、√d、掩码 | 2:07 |
| ② | 前向与反向：六行公式、两个雅可比、注意力反传 | 2:25 |
| ③ | 两种实现：numpy 手写反向 vs torch autograd | 2:37 |

关键数字：前向差 3×10⁻¹⁵、31 个参数张量梯度差 3×10⁻¹⁴、中心差分相对误差 5e-11～8e-9；
numpy 版可直接加载生产权重（`max|Δu| = 5.4e-6` N，闭环 30/30）。

## 许可

代码与文档采用 MIT 许可；视频与讲义中的推导可自由引用，注明出处即可。

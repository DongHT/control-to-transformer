# Transformer 从零实现

一份**可以逐行验证**的 Transformer 教程：从动机、推导，到前向/反向的每一个雅可比，
再到**纯 numpy 手写反向**与 **torch autograd** 两份实现，以及三张结果可视化。

> 核心承诺：手写反向与 autograd 的前向输出差 **3×10⁻¹⁵**、全部 31 个参数张量的梯度差 **3×10⁻¹⁴**。
> 这不是"跑通了"，而是"证明它是对的"。

## 内容

| 部分 | 文件 | 说明 |
|---|---|---|
| **讲义** | `latex/tutorial.tex` | 约 30 页：背景 → 结构 → 逐层前向/反向 → 两份实现 → 可视化 → 习题 |
| **视频①** | `out/t1_为什么长这样.mp4` | 2:07　RNN 的两个硬伤、检索视角、√d 的方差推导、因果掩码 |
| **视频②** | `out/t2_前向与反向.mp4` | 2:25　结构框图、六行前向公式、softmax/LN 雅可比、注意力反传 |
| **视频③** | `out/t3_两种实现.mp4` | 2:37　代码骨架、五个易错点、训练曲线重合、加载生产权重、注意力×倒立摆 |
| **视频④** | `out/t4_前向逐算子.mp4` | 3:43　**9 个算子逐个讲**：公式 ↔ 源码 ↔ 真实数值（代码由源码实时切片）|
| **视频⑤** | `out/t5_反向逐算子.mp4` | 3:54　**每个雅可比的推导 ↔ 对应代码**：tanh、LN、softmax、注意力、FFN、残差、验证 |
| **视频⑥** | `out/t6_串起来.mp4` | 3:22　**全链路**：一次迭代 → 训练循环 → 参数更新 → int8 部署 → 闭环应用 |
| **视频⑦** | `out/t7_训练方法.mp4` | 3:37　**优化器**：SGD→Momentum→RMSProp→Adam→AdamW，二维病态曲面轨迹 + 真实模型对比 |
| **视频⑧** | `out/t8_DAgger原理到实现.mp4` | 3:41　**DAgger**：分布漂移 → no-regret 归约 → 三个函数 → 0/30→30/30 |
| **视频⑨** | `out/t9_PPO原理到实现.mp4` | 4:01　**PPO**：MDP → 策略梯度 → GAE → clip → 七行核心更新 → 域随机化 |
| **numpy 实现** | `code/tinytf_numpy.py` | 前向 40 行 + 手写反向 80 行 + AdamW，无任何自动微分 |
| **torch 实现** | `code/tinytf_torch.py` | 等价结构，含权重互转（`state_from_numpy` / `numpy_from_state`）|
| **对齐检验** | `code/parity_test.py` | 与 autograd、中心差分双重验证 |
| **训练对比** | `code/train_demo.py` | 同初始化同数据，双实现同步训练 |
| **加载预训练** | `code/load_pretrained.py` | 生产权重装进 numpy 版：一致性、闭环、注意力 |
| **插图** | `code/make_figs.py` | 损失曲线、梯度量级 |

## 第四、五集：公式与代码逐行对应

这两集把 9 个算子拆开讲，每一屏都是「左边公式、右边源码、底部真实数值」，
其中源码是用锚点**从 `tinytf_numpy.py` 实时切出来**的（不是手抄），所以屏幕上看到的就是真正在跑的代码。

| 集 | 算子 | 公式 | 对应代码 |
|---|---|---|---|
| ④ | ① 嵌入+位置编码 | $z=xW^T+b+\mathrm{pos}$ | `z = x @ W_in.T + b_in + pos` |
| ④ | ② LayerNorm | $\hat z=(z-\mu)/\sigma$ | `ln_forward()` 7 行 |
| ④ | ③ QKV 投影+分头 | $[Q;K;V]=hW^T+b$ | reshape + transpose |
| ④ | ④ 缩放点积+掩码 | $s=qk^T/\sqrt{d_h}+M$ | `einsum` + `triu` |
| ④ | ⑤ softmax | $a=e^{s-m}/\sum e^{s-m}$ | `softmax()` 4 行 |
| ④ | ⑥ 加权求和+投影 | $y=(av)W_o^T$ | `einsum` + reshape |
| ④ | ⑦⑧ 残差+FFN+GELU | $z\leftarrow z+W_2\mathrm{GELU}(W_1h)$ | `fc1/gelu/fc2` |
| ④ | ⑨ 头+tanh | $u=12\tanh(w^Tz_T)$ | `head` + `tanh` |
| ⑤ | 输出层 | $\bar r=\bar u\cdot12(1-\tanh^2 r)$ | `dr = du * u_max * (1-tanh²)` |
| ⑤ | LayerNorm 反向 | $\partial z=\frac{\gamma}{\sigma}(g-\bar g-\hat z\overline{g\hat z})$ | `ln_backward()` 10 行 |
| ⑤ | softmax 反向 | $\partial s_j=a_j(g_j-\sum a_ig_i)$ | 一行 `dsm` |
| ⑤ | 注意力反向 | $\bar V=A^T\bar Y,\ \bar A=\bar YV^T$ 等 5 式 | 7 行 `einsum` |
| ⑤ | FFN 反向 | $\partial W_2=z^Ta$（**不是** $\bar a^Ta$）| `dz.T @ act` |
| ⑤ | 残差/嵌入 | $\bar z=\bar z_{out}(I+\partial f/\partial z)$ | `dz` 累加 + `pos.sum(0)` |

## 第六集：把算子串起来

前面两集拆到算子，这一集把它们接成一条完整的流水线：

| 段落 | 内容 | 屏幕上的可视化 |
|---|---|---|
| 一次迭代 | 前向 9 算子 → 损失 → 反向 9 算子 → AdamW | 两条管道（正向/逆向），一个小方块随时间在管道上移动 |
| 前向 | $x\to z\to h\to qkv\to a\to y\to u$ | 6 段流水线 + 7 个张量形状 |
| 反向 | $\partial u\to\partial r\to\partial z\to\partial S\to\partial Q,K\to\partial W$ | 同一管道逆流，7 个梯度表达式 |
| 训练 | epoch → mini-batch → step | 高亮逐行跳动的训练循环代码（3,264 步） |
| 更新 | $m_t,v_t,\theta_t$ | AdamW 五⼀行代码 + 为什么不衰减 LN 参数 |
| 部署 | 逐张量 int8 | 量化公式 + 代码 + 71 KB→16.6 KB / 0.14 N / 0.32 ms |
| 应用 | 传感器→窗口→模型→力→小车→传感器 | 闭环框图 + 实时倒立摆（50 Hz，30/30） |

代码块采用 **VS Code Dark+ 配色**：关键字 `#569CD6`、流程控制 `#C586C0`、函数名 `#DCDCAA`、
参数 `#9CDCFE`、字符串 `#CE9178`、数字 `#B5CEA8`、注释 `#6A9955`，
并带"当前行"底纹，跟着旁白逐行高亮。

## 第七集：训练方法（优化器）

前面讲的是"梯度怎么算"，这一集讲"拿到梯度后怎么走"。所有曲线与轨迹都是**真实迭代**算出来的
（`code/optim_demo.py`），不是示意图。

| 优化器 | 公式 | 本模型实测（3 epoch） |
|---|---|---|
| SGD | $\theta\leftarrow\theta-\eta g$ | $\eta{=}10^{-3}$：44.2 → 4.47；$\eta{=}2\times10^{-2}$：**发散**（70.7 → 72.0）|
| Momentum | $v\leftarrow\beta v+g$ | $\eta{=}10^{-3}$：63.5 → 59.9（有效 η 被放大 10 倍，震荡）|
| RMSProp | $\theta\leftarrow\theta-\eta g/(\sqrt v+\varepsilon)$ | 57.6 → 0.043 |
| Adam | $+\hat m/\sqrt{\hat v}$ 偏差修正 | **7.04 → 0.047** |
| AdamW | 解耦 $\lambda\theta$ | 同 Adam（本模型量级下差异小）|

两个关键可视化：

- **二维病态曲面** $f=\tfrac12(x^2+\kappa y^2)$（$\kappa=50$）上的真实轨迹：
  60 步后 SGD $5.7\times10^{-2}$、Momentum $4.4\times10^{-2}$、RMSProp $2\times10^{-77}$、Adam $2.5\times10^{-2}$；
- **逐参数步长**：第一步时 SGD 下 `w_head` 的步长是 `l1.W_qkv` 的 **31 倍**，Adam 下两者完全相同。

还讲了两个容易含混的点：`1−β₂ᵗ=0.001` 时不做偏差修正会把首步放大 **31.6 倍**；
Adam 加 L2 与 AdamW 的区别（衰减是否被二阶矩归一化），以及为什么 LayerNorm 参数/位置编码/bias 不该衰减。

## 快速验证（三分钟）

```bash
cd code
python3 parity_test.py     # 手写反向 vs autograd：看到 3e-14 就说明对了
```

输出节选：

```
① 前向输出        最大偏差 = 3.109e-15
② 反向梯度        全部参数的最大梯度偏差 = 3.020e-14
③ 中心差分        l0.W1(3,7): 相对误差 = 8.04e-09
结论: ✅ 手写反向与 autograd 一致（<1e-6）
```

## 训练与部署

```bash
python3 train_demo.py       # 双实现各训 8 个 epoch（numpy 142s / torch 35s）
python3 load_pretrained.py  # 加载生产权重：512 窗口 max|Δu|=5.4e-6 N，闭环 30/30
python3 make_figs.py        # 生成 out/fig_*.png
```

训练对比（同初始化、同数据、同批次顺序）：

| | 训练损失 | 验证损失 | 耗时 |
|---|---|---|---|
| numpy（手写反向） | 3.60×10⁻³ | 3.694×10⁻³ | 142 s |
| torch（autograd） | 3.60×10⁻³ | 3.695×10⁻³ | 35 s |
| 差异 | 1.7×10⁻⁷ | 1.3×10⁻⁷ | 4.1× 加速 |

## 结构（18,177 参数）

```
x (B,24,6) → 嵌入 (B,24,32) + 位置编码
  → ×2 层: [LN → 4 头因果注意力 → ⊕] → [LN → FFN 32→64→32 + GELU → ⊕]
  → 取最后 token → LN → 线性 → u = 12·tanh(·)
```

每个算子的张量形状与参数量见讲义第 2 节。

## 五个易错点（讲义第 6.2 节）

1. **多头 reshape 顺序**：`(B,T,d) → (B,T,H,dh) → transpose → (B,H,T,dh)`
2. **softmax 减基线项**：`dS = a * (da - sum(da*a))`，漏了会拖慢收敛
3. **W2 / b2 的梯度**来自残差主干 `dz`，不是 `dact`（形状会从 (d,ff) 变 (ff,ff)）
4. **LayerNorm 用紧凑式**：`dx = γ/σ (g − ḡ − ẑ·mean(gẑ))`
5. **掩码位置梯度自动为零**：`a=0 ⇒ dS=0`，无需再清零

## 与其他材料的关系

本教程的模型就是主项目里部署的控制器（`build/model.py`）。
把它变成控制器只差三处：输入特征（6 维归一化）、窗口（24 步 @ 50 Hz）、输出限幅（12 N）。
量化到 int8 后 16.6 KB，浏览器内单步 0.32 ms。

## 编讲义

```bash
cd latex && xelatex tutorial.tex && xelatex tutorial.tex   # 需要 ctex 与 listings
```

## 许可

MIT（与主项目一致）。

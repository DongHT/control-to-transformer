"""把 tutorial.tex（ctexart）拆成书稿的 7 章（ctexbook 章节）。

输出：book/chapters/p3ch01.tex … p3ch07.tex
规则：
  · 按 \\section 切分，\\section→\\chapter，\\subsection→\\section，\\subsubsection→\\subsection
  · 图片路径 ../out/xxx.png → xxx.png（由 main.tex 的 \\graphicspath 解析）
  · 去掉 abstract / tableofcontents（内容并入前言）
"""
import os
import re

ROOT = "/Users/liwenhao/my_doc/transformer"
SRC = os.path.join(ROOT, "tutorial", "latex", "tutorial.tex")
OUT = os.path.join(ROOT, "book", "chapters")
os.makedirs(OUT, exist_ok=True)

body = open(SRC, encoding="utf-8").read()
body = body[body.index("\\begin{document}") + len("\\begin{document}"):]
body = body[:body.index("\\end{document}")]
body = re.sub(r"\\maketitle|\\tableofcontents|\\newpage", "", body)
body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)

# 按 \section 切
parts = re.split(r"\\section\{(.+?)\}\s*(?:\\label\{[^}]*\})?", body)
pairs = []
for i in range(1, len(parts), 2):
    pairs.append((parts[i].strip(), parts[i + 1]))
print("找到 section:", [p[0] for p in pairs])

# 分组 → 章
GROUPS = [
    ("为什么长这样：从 RNN 到注意力", ["背景：为什么需要 Transformer", "结构总览"], "第 11 章"),
    ("前向与反向：公式与雅可比", ["逐层前向与反向"], "第 12 章"),
    ("纯 numpy 实现", ["纯 numpy 实现"], "第 13 章"),
    ("torch 实现与训练对比", ["torch 实现与训练对比"], "第 14 章"),
    ("训练方法：从 SGD 到 AdamW", ["训练方法：从 SGD 到 AdamW"], "第 15 章"),
    ("结果可视化与落地", ["结果可视化", "从教学模型到控制器"], "第 16 章"),
    ("习题与代码索引", ["习题", "代码索引"], "第 17 章"),
]

by_title = {t: c for t, c in pairs}
for idx, (chap_title, secs, label) in enumerate(GROUPS, 1):
    chunks = []
    for s in secs:
        if s not in by_title:
            print("  ! 缺 section:", s)
            continue
        chunks.append(by_title[s])
    text = "\n".join(chunks)
    # 层级提升
    text = text.replace("\\subsection{", "\\section{")
    text = text.replace("\\subsubsection{", "\\subsection{")
    # 图片路径
    text = text.replace("{../out/", "{")
    header = (f"% 由 tutorial.tex 自动拆分为{label}\n"
              f"\\chapter{{{chap_title}}}\n")
    open(os.path.join(OUT, f"p3ch{idx:02d}.tex"), "w", encoding="utf-8").write(header + text)
    print(f"  写入 p3ch{idx:02d}.tex  ({chap_title})  {len(text)} 字符")

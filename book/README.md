# 合订本讲义

三篇 + 附录，约 150 页。

## 结构

```
book/
├── main.tex              合订本主文件（前言 + 分篇 + 附录 + 资源索引）
└── chapters/
    └── p3ch01..07.tex    第三篇（由 tutorial/latex/tutorial.tex 自动拆分）
```

第一、二篇直接 `\input` 课程讲义原文件（`../course/latex/chapters/`），
所以**改课程讲义就等于改书，不存在两份内容不同步的问题**。

## 编译

```bash
cd book
xelatex main.tex     # 第一遍生成目录与交叉引用
xelatex main.tex     # 第二遍填好页码
```

需要 TeX Live（含 `ctex`、`empheq`、`listings`）。
图片通过 `\graphicspath` 从 `../course/latex/` 与 `../tutorial/out/` 取，不需要拷贝。

## 拆分第三篇

若 `tutorial/latex/tutorial.tex` 有改动，重新生成第三篇：

```bash
python3 build/split_tutorial.py
```

它会按 `\section` 切成 7 章，并把图片路径改写成 `\graphicspath` 能解析的形式。

## 章节与视频的对应

| 书 | 视频 | 时长 |
|---|---|---|
| 第 1–6 章 | 第 1–6 讲 | 15:33 |
| 第 7–10 章 | 第 7–10 讲 | 10:28 |
| 第 11–17 章 | 第 11–17 集 | 21:45 |
| 附录 A | 附录 A | 3:17 |

章标题里的“第 N 讲”就是视频编号，与章号一致（1–10）。

"""把「显示文本」里的中文数词改成阿拉伯数字（保守规则）。

只改数值 ≥3 的数词（含十以上的复合数），因为「一个/两个/一步」在中文里是冠词或固定搭配：
  · 四牛顿 → 4 牛顿        三十一倍 → 31 倍
  · 四个   → 4 个          二十四步 → 24 步
  · 第二个 → 第 2 个        三个 → 3 个
不改：一个 / 两个 / 一样 / 第一步 / 一次 / 一条（冠词或固定搭配）
朗读文本（tts 行）整行跳过。
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cn_num import cn2int  # noqa: E402

SCENES = "/Users/liwenhao/my_doc/transformer/course/video/scenes"
MEASURE = "个条行维步层次倍组项节秒度米牛顿焦耳"

# 固定搭配：出现即跳过（先做保护，再匹配）
IDIOM = ["一样", "一些", "一定", "一直", "一般", "一起", "唯一", "一下", "一点",
         "一步", "一部分", "一边", "一方面", "一系列", "一律", "一切", "一致",
         "一半", "十分", "几乎", "统一", "之一", "不一", "千万", "万一", "一再",
         "一同", "一并", "一位", "一并"]

DIGIT = {"零":"0","一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9"}
DEC = re.compile(r"([负])?([零一二三四五六七八九])(点)([零一二三四五六七八九]+)")
PCT = re.compile(r"百分之([零一二三四五六七八九十百两]+)")
PAT = re.compile(r"(第)?([零一二三四五六七八九十百两]+)"
                 "(牛顿|焦耳|个|条|行|维|步|层|次|倍|组|项|节|秒|度|米)")


def _cjk(c):
    return "\u4e00" <= c <= "\u9fff"


def _space_before(line, i):
    """在阿拉伯数字前补一个空格（若前一个字符是汉字）"""
    if i > 0 and _cjk(line[i - 1]):
        return line[:i] + " " + line[i:]
    return line


def convert(line):
    changes = []

    # ① 小数：一点四 → 1.4，负二点九三 → −2.93
    def rep_dec(m):
        neg, a, _, b = m.groups()
        v = DIGIT[a] + "." + "".join(DIGIT[c] for c in b)
        new = ("−" if neg else "") + v
        changes.append((m.group(0), new))
        return new

    line = DEC.sub(rep_dec, line)

    # ② 百分之X → X%
    def rep_pct(m):
        v = cn2int(m.group(1))
        if v is None:
            return m.group(0)
        changes.append((m.group(0), f"{v}%"))
        return f"{v}%"

    line = PCT.sub(rep_pct, line)

    # ③ 数词 + 量词（只改 ≥3）
    def rep(m):
        ordinal, num, meas = m.group(1), m.group(2), m.group(3)
        whole = m.group(0)
        i = m.start()
        for w in IDIOM:
            if line[max(0, i - 1):i + len(w)] == w or whole.startswith(w):
                return whole
        v = cn2int(num)
        if v is None or v < 3:
            return whole
        new = (f"第 {v} {meas}" if ordinal else f"{v} {meas}")
        changes.append((whole, new))
        return new

    line = PAT.sub(rep, line)

    # ④ 百分之1.2 → 1.2%
    line = re.sub(r"百分之\s*([\d.]+)", r"\1%", line)
    # ⑤ 数字与单位之间补空格：-6.2牛顿 → -6.2 牛顿
    line = re.sub(r"([\d.])\s*(牛顿|焦耳|个|条|行|维|步|层|次|倍|组|项|节|秒|度|米)",
                  r"\1 \2", line)
    # ⑥ 孤立的“零” → 0
    line = re.sub(r"(?<=[，。：、\s])零(?=[，。、\s]|$)", "0", line)
    line = re.sub(r"(?<=[\u4e00-\u9fff])零(?=[，。、；])", " 0", line)

    # ⑦ 数字前补空格：这4 行 → 这 4 行
    out = []
    for k, ch in enumerate(line):
        if ch.isdigit() and k > 0 and _cjk(line[k - 1]) and line[k - 1] not in "−-":
            out.append(" ")
        out.append(ch)
    return "".join(out), changes


def main():
    total = 0
    for p in sorted(glob.glob(os.path.join(SCENES, "*.py"))):
        src = open(p, encoding="utf-8").read().split("\n")
        out, logs = [], []
        for ln in src:
            if '"tts"' in ln:
                out.append(ln)
                continue
            new, ch = convert(ln)
            if ch:
                logs.append((ln.strip()[:78], new.strip()[:78]))
            out.append(new)
        if logs:
            open(p, "w", encoding="utf-8").write("\n".join(out))
            print(f"\n=== {os.path.basename(p)}  {len(logs)} 行 ===")
            for a, b in logs[:6]:
                print(f"  - {a}\n  + {b}")
            if len(logs) > 6:
                print(f"  … 另有 {len(logs)-6} 行")
            total += len(logs)
    print(f"\n共修改 {total} 行")


if __name__ == "__main__":
    main()

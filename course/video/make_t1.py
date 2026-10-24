"""Build the chapter 1 video:  python3 make_ch01.py [--bench N]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))

import render
import t1

if __name__ == "__main__":
    bench = None
    if "--bench" in sys.argv:
        bench = int(sys.argv[sys.argv.index("--bench") + 1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                       "../../tutorial/out/t1_为什么长这样.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render.build(dict(shots=t1.SHOTS), out,
                 work=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "_work"),
                 voice="zh-CN-XiaoxiaoNeural", rate=-4, bench=bench, engine="edge")

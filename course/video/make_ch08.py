"""Build the chapter 1 video:  python3 make_ch01.py [--bench N]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))

import render
import ch08

if __name__ == "__main__":
    bench = None
    if "--bench" in sys.argv:
        bench = int(sys.argv[sys.argv.index("--bench") + 1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                       "ch08_Transformer.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render.build(dict(shots=ch08.SHOTS), out,
                 work=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "_work"),
                 voice="zh-CN-XiaoxiaoNeural", rate=-4, bench=bench, engine="edge")

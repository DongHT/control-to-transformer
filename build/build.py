"""Assemble the single-file, self-contained HTML deliverable."""
import base64, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "parts")
OUT = os.path.join(HERE, "out")
DEST = os.path.join(os.path.dirname(HERE), "index.html")

PARTS = ["00_head.html", "10_model.html", "20_classical.html",
         "30_transformer.html", "40_lab.html", "50_results.html", "60_rl.html",
         "70_js.html", "80_js.html", "90_foot.html"]


def load_json(name):
    return json.load(open(os.path.join(OUT, name)))


def probe_pack(n=5):
    xin = np.load(os.path.join(OUT, "probe_in.npy"))
    xout = np.load(os.path.join(OUT, "probe_out_q.npy"))
    idx = np.linspace(0, len(xin) - 1, n).astype(int)
    return dict(inputs=[np.round(xin[i], 6).tolist() for i in idx],
                expect=[round(float(xout[i]), 6) for i in idx])


def downsample_hist(hist, k=2):
    out = []
    for i, h in enumerate(hist):
        out.append(dict(epoch=i + 1, val=h["val"], train=h["train"]))
    return out


def ref_trace():
    """A reference closed-loop trace produced by the Python stack, used to check
    that the JS re-implementation of plant + Energy+LQR matches it bit-for-bit."""
    import sys
    sys.path.insert(0, HERE)
    import numpy as np
    import controllers as C
    ctl = C.build("hybrid_lqr")
    X, U = C.rollout(ctl, [0.0, np.pi - 0.14, 0.0, 0.0], T=3.0)
    st = 10
    return dict(x0=[0.0, np.pi - 0.14, 0.0, 0.0], stride=st,
                th=[round(float(v), 8) for v in X[::st, 1]],
                p=[round(float(v), 8) for v in X[::st, 0]],
                u=[round(float(v), 8) for v in U[::st]])


def main():
    blob = load_json("policy_weights.json")
    blob_rl = load_json("policy_dr_weights.json")
    if "meta" in blob:
        meta = blob["meta"]
    else:                                  # (re)derive it from the blob itself
        import numpy as _np
        q = 0.0
        if os.path.exists(os.path.join(OUT, "policy_meta.json")):
            q = load_json("policy_meta.json").get("quant_max_err", 0.0)
        ps = blob["params"]
        meta = dict(h=24, feat=6, umax=12.0,
                    n_params=int(sum(_np.prod(e["shape"]) for e in ps.values())),
                    size_kb=round(sum(len(e["data"]) for e in ps.values()) / 1024, 1),
                    quant_max_err=q)
    log = load_json("train_log.json")
    met = load_json("metrics.json")
    plant = dict(m_c=1.0, m_p=0.25, l=0.6, g=9.81, b_c=0.12, b_p=0.004,
                 u_max=12.0, dt=0.02, x_max=2.4)
    # take the *exact* controller parameters from the Python implementations so that
    # the browser build can never drift away from the evaluated design
    import sys
    sys.path.insert(0, HERE)
    import controllers as C
    K, _ = C.lqr_gain()
    e, sm, hy = C.EnergyFBL(), C.SMC(), C.Hybrid(C.LQR())
    cfg = dict(
        plant=plant,
        lqr=dict(K=[float(v) for v in K[0]],
                 poles=[[-11.909, 0.0], [-4.473, 0.0], [-1.09, 0.429], [-1.09, -0.429]],
                 Q=[1, 40, 1, 2], R=0.05),
        energy=dict(kE=e.kE, phi=e.phi, kx=e.kx, kd=e.kd, Ed=e.Ed),
        smc=dict(lam=sm.lam, eta=sm.eta, ks=sm.ks, phi=sm.phi, kp=sm.kp, kd=sm.kd),
        sw=dict(th_in=hy.th_in, w_in=hy.w_in, th_out=hy.th_out, w_out=hy.w_out),
        policy=dict(h=meta["h"], feat=meta["feat"], umax=meta["umax"],
                    n_params=meta["n_params"], size_kb=meta["size_kb"],
                    quant_err=meta["quant_max_err"]),
    )
    html = []
    for name in PARTS:
        html.append(open(os.path.join(P, name), encoding="utf-8").read())
    html = "\n".join(html)
    html = (html
            .replace("__CFG_JSON__", json.dumps(cfg, ensure_ascii=False))
            .replace("__BLOB_JSON__", json.dumps(blob["params"], separators=(",", ":")))
            .replace("__PROBES_JSON__", json.dumps(probe_pack(), separators=(",", ":")))
            .replace("__TRAINLOG_JSON__", json.dumps(downsample_hist(log["history"])))
            .replace("__PYMETRICS_JSON__", json.dumps(met))
            .replace("__REFTRACE_JSON__", json.dumps(ref_trace()))
            .replace("__BLOB_RL_JSON__", json.dumps(blob_rl["params"], separators=(",", ":")))
            .replace("__RL_LOGS_JSON__", json.dumps(load_json("rl_logs.json"), separators=(",", ":")))
            .replace("__PROBES2_JSON__", json.dumps(load_json("probes2.json"), separators=(",", ":")))
            .replace("__RLCOMPARE_JSON__", json.dumps(load_json("rl_compare.json"))))
    open(DEST, "w", encoding="utf-8").write(html)
    print("wrote %s  (%.1f KB)" % (DEST, len(html.encode()) / 1024))


if __name__ == "__main__":
    main()

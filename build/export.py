"""Export the trained policy to a compact int8 blob + verification vectors.

Layout: every parameter is quantised per-tensor to int8 (scale stored separately),
except biases / LayerNorm params which stay float32 (they are tiny and sensitive).
The browser decodes the base64 blob into Float32Array weights and runs an identical
forward pass.
"""
import base64, json, os
import numpy as np
import torch

import plant as pl
import dataset as D
from dataset import H, FEAT, DT
from model import Policy
from plant import P

OUT = os.path.join(os.path.dirname(__file__), "out")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    model = Policy()
    model.load_state_dict(torch.load(os.path.join(OUT, "policy.pt")))
    model.eval()

    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    blob = {}
    for k, v in sd.items():
        if k.endswith("weight") and v.ndim == 2:
            m = np.abs(v).max() or 1.0
            q = np.round(v / m * 127).astype(np.int8)
            blob[k] = dict(dtype="i8", shape=list(v.shape), scale=float(m / 127), data=b64(q))
        else:
            blob[k] = dict(dtype="f32", shape=list(v.shape), data=b64(v.astype(np.float32)))

    # ---- rebuild a float model from the quantised blob and check the gap
    W = {}
    for k, e in blob.items():
        raw = np.frombuffer(base64.b64decode(e["data"]), dtype=np.int8 if e["dtype"] == "i8" else np.float32)
        raw = raw.astype(np.float64).reshape(e["shape"])
        W[k] = raw * e["scale"] if e["dtype"] == "i8" else raw
    m2 = Policy()
    m2.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in W.items()})
    m2.eval()

    Xv = torch.from_numpy(np.random.RandomState(0).randn(64, H, FEAT).astype(np.float32))
    with torch.no_grad():
        a = model(Xv)
        b = m2(Xv)
    print("quantisation max |du| = %.5f N (u_max=%.1f)"
          % ((a - b).abs().max().item(), P["u_max"]))

    total_i8 = sum(len(base64.b64decode(e["data"])) for e in blob.values() if e["dtype"] == "i8")
    total_f32 = sum(4 * int(np.prod(e["shape"])) for e in blob.values() if e["dtype"] == "f32")
    print("int8 bytes %d + f32 bytes %d  -> base64 %.1f KB"
          % (total_i8, total_f32, sum(len(e["data"]) for e in blob.values()) / 1024))

    # ---- verification vectors for the JS port (real windows from the expert data)
    X, Y = D.gen(n_ep=6, T=6.0, noise=1.5, seed=42)
    idx = np.random.RandomState(3).choice(len(X), 48, replace=False)
    probes = X[idx]
    with torch.no_grad():
        out = model(torch.from_numpy(probes)).numpy()
        out2 = m2(torch.from_numpy(probes)).numpy()
    np.save(os.path.join(OUT, "probe_in.npy"), probes)
    np.save(os.path.join(OUT, "probe_out.npy"), out)
    np.save(os.path.join(OUT, "probe_out_q.npy"), out2)

    meta = dict(
        h=H, feat=FEAT, umax=P["u_max"], dt=DT,
        probes=dict(inputs=[np.round(p, 6).tolist() for p in probes[:3]],
                    outputs=[round(float(o), 6) for o in out[:3]],
                    outputs_q=[round(float(o), 6) for o in out2[:3]]),
        n_params=int(sum(p.numel() for p in model.parameters())),
        size_kb=round(sum(len(e["data"]) for e in blob.values()) / 1024, 1),
        quant_max_err=float((a - b).abs().max().item()),
    )
    json.dump(dict(params=blob, meta=meta), open(os.path.join(OUT, "policy_weights.json"), "w"))
    json.dump(meta, open(os.path.join(OUT, "policy_meta.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "policy_weights.json"))


if __name__ == "__main__":
    main()

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.deploy import OnnxPredictor
from edgekit.hwprobe import probe_onnx
from edgekit.zoo import ALL_KINDS, VARIANTS, dataset_of, restore, tags_for

FIELDS = ["model", "variant", "dataset", "params_M", "weights_MB", "onnx_MB", "onnx_top1", "onnx_top5",
          "torch_ms", "torch_p95_ms", "ort_ms", "ort_p95_ms", "images_per_s", "peak_mem_MB"]


def torch_timing(model, runs, threads):
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.inference_mode():
            for _ in range(20):
                m(x)
            t = []
            for _ in range(runs):
                s = time.perf_counter()
                m(x)
                t.append((time.perf_counter() - s) * 1000)
    finally:
        torch.set_num_threads(prev)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser(description="CPU benchmark of every exported model")
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no_accuracy", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    rows = []
    for kind in ALL_KINDS:
        for variant in (["-"] if kind == "nas" else VARIANTS):
            tag = tags_for(kind, variant)
            onnx_path = paths.onnx_file(tag)
            if not (os.path.exists(onnx_path) and os.path.exists(paths.weight_file(tag))):
                print("skipping", tag)
                continue
            model = restore(kind, variant)
            t_med, t_p95 = torch_timing(model, a.runs, a.threads)
            ort_stats = probe_onnx(onnx_path, a.threads, a.runs)

            top1 = top5 = ""
            if not a.no_accuracy:
                loaders = make_loaders(dataset_of(kind), 256, a.smoke, workers=0 if a.smoke else 2)
                top1, top5 = [round(v, 2) for v in OnnxPredictor(onnx_path).accuracy(loaders.test)]

            n_params = sum(p.numel() for p in model.parameters())
            mem = ort_stats["peak_mem_mb"]
            row = {"model": kind, "variant": "nas" if kind == "nas" else variant, "dataset": dataset_of(kind),
                   "params_M": round(n_params / 1e6, 3),
                   "weights_MB": round(os.path.getsize(paths.weight_file(tag)) / 2**20, 3),
                   "onnx_MB": round(os.path.getsize(onnx_path) / 2**20, 3),
                   "onnx_top1": top1, "onnx_top5": top5,
                   "torch_ms": round(t_med, 3), "torch_p95_ms": round(t_p95, 3),
                   "ort_ms": round(ort_stats["latency_ms"], 3), "ort_p95_ms": round(ort_stats["p95_ms"], 3),
                   "images_per_s": round(1000 / ort_stats["latency_ms"], 1),
                   "peak_mem_MB": "" if mem is None else round(mem, 1)}
            rows.append(row)
            print(row)

    out = paths.log_file("benchmark_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()

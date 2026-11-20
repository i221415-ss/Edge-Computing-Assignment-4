"""One entry point for inference with every trained model.

examples
  python 07_predict.py --kind squeezenet --variant distilled --image cat.png
  python 07_predict.py --kind nas --image truck.png --runtime onnx
  python 07_predict.py --kind all --variant both --evaluate --runtime onnx
"""
import argparse
import os

import numpy as np
import torch

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.deploy import OnnxPredictor
from edgekit.engine import score
from edgekit.zoo import ALL_KINDS, VARIANTS, dataset_of, image_to_array, label_names, restore, tags_for


def make_runner(kind, variant, runtime):
    """returns (logits function on numpy batches, torch model or None)"""
    tag = tags_for(kind, variant)
    if runtime == "onnx":
        pred = OnnxPredictor(paths.onnx_file(tag))
        return pred.logits, None
    model = restore(kind, variant)

    def fn(batch):
        with torch.no_grad():
            return model(torch.from_numpy(batch)).numpy()

    return fn, model


def show(tag, logits, names, k=5):
    z = logits[0] - logits[0].max()
    p = np.exp(z) / np.exp(z).sum()
    print(f"\n{tag}")
    for i in np.argsort(-p)[:k]:
        print(f"   {names[i] if names else i:<18} {p[i] * 100:6.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Unified inference for students and the NAS model")
    ap.add_argument("--kind", default="all", choices=ALL_KINDS + ["all"])
    ap.add_argument("--variant", default="distilled", choices=VARIANTS + ["both"])
    ap.add_argument("--runtime", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--image")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not (a.image or a.evaluate):
        ap.error("pass --image and/or --evaluate")

    for kind in (ALL_KINDS if a.kind == "all" else [a.kind]):
        which = dataset_of(kind)
        for variant in (["-"] if kind == "nas" else (VARIANTS if a.variant == "both" else [a.variant])):
            tag = tags_for(kind, variant)
            if not os.path.exists(paths.weight_file(tag)) or (a.runtime == "onnx" and not os.path.exists(paths.onnx_file(tag))):
                print(f"{tag}: files missing, skipped")
                continue
            run, model = make_runner(kind, variant, a.runtime)
            if a.image:
                show(f"{tag} [{a.runtime}]", run(image_to_array(a.image, which)), label_names(which))
            if a.evaluate:
                data = make_loaders(which, 256, a.smoke, workers=0 if a.smoke else 2)
                hit1 = hit5 = n = 0
                for xb, yb in data.test:
                    z = run(xb.numpy())
                    best = np.argsort(-z, axis=1)[:, :5]
                    y = yb.numpy()
                    hit1 += int((best[:, 0] == y).sum())
                    hit5 += int((best == y[:, None]).any(1).sum())
                    n += len(y)
                print(f"{tag:<28s} [{a.runtime}] top1 {hit1 * 100 / n:.2f}  top5 {hit5 * 100 / n:.2f}")


if __name__ == "__main__":
    main()

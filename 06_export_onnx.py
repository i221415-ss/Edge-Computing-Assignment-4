import argparse
import os

import numpy as np
import torch

from edgekit import paths
from edgekit.deploy import OnnxPredictor, export_onnx
from edgekit.zoo import ALL_KINDS, VARIANTS, restore, tags_for


def main():
    ap = argparse.ArgumentParser(description="Export trained models to ONNX and check them against PyTorch")
    ap.add_argument("--kind", default="all", choices=ALL_KINDS + ["all"])
    a = ap.parse_args()

    for kind in (ALL_KINDS if a.kind == "all" else [a.kind]):
        for variant in (["-"] if kind == "nas" else VARIANTS):
            tag = tags_for(kind, variant)
            if not os.path.exists(paths.weight_file(tag)):
                print(f"no weights for {tag}, skipped")
                continue
            model = restore(kind, variant)
            target = paths.onnx_file(tag)
            export_onnx(model, target)

            probe = torch.randn(4, 3, 32, 32)
            with torch.no_grad():
                expected = model(probe).numpy()
            got = OnnxPredictor(target).logits(probe.numpy())
            print(f"{tag}: {os.path.getsize(target) / 2**20:.2f} MB, "
                  f"max abs diff vs torch {np.abs(expected - got).max():.2e}")


if __name__ == "__main__":
    main()

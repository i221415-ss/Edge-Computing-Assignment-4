import argparse
import json

import torch
import torch.nn.functional as F

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.engine import pick_device, run_training, score, store_weights, write_json
from edgekit.searchspace import build_from_genome


def main():
    ap = argparse.ArgumentParser(description="Train the best searched architecture from scratch")
    ap.add_argument("--genome", default=paths.log_file("best_genome.json"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    with open(a.genome) as fh:
        genome = json.load(fh)["genome"]

    torch.manual_seed(1)
    data = make_loaders("cifar10", a.bs, a.smoke, workers=a.workers)
    net = build_from_genome(genome, 10)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"training searched model from scratch ({n_params / 1e6:.3f}M parameters)")

    log = run_training(net, data.train, data.test, a.epochs, a.lr, a.wd,
                       lambda xb, yb: F.cross_entropy(net(xb).float(), yb), tag="nas_best")
    final = score(net, data.test, pick_device())
    print("test result", final)

    store_weights(paths.weight_file("nas_best"), net, genome=genome, epochs=a.epochs, top1=final["top1"])
    write_json(paths.log_file("nas_final.json"),
               {"final": final, "genome": genome, "epochs": a.epochs, "params": n_params, "history": log})


if __name__ == "__main__":
    main()

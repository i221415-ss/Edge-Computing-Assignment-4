import argparse

import torch
import torch.nn.functional as F

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.engine import pick_device, run_training, score, store_weights, write_json
from edgekit.students import KINDS, create_student


def train_kind(kind, a):
    torch.manual_seed(1)
    data = make_loaders("cifar100", a.bs, a.smoke, workers=a.workers)
    net = create_student(kind, 100)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"--- {kind} without distillation, {n_params / 1e6:.2f}M parameters")

    log = run_training(net, data.train, data.test, a.epochs, a.lr, a.wd,
                       lambda xb, yb: F.cross_entropy(net(xb).float(), yb), tag=f"{kind}/plain")
    final = score(net, data.test, pick_device())
    print(kind, "plain", final)

    store_weights(paths.weight_file(f"{kind}_plain"), net, epochs=a.epochs, top1=final["top1"])
    write_json(paths.log_file(f"{kind}_plain.json"),
               {"final": final, "epochs": a.epochs, "params": n_params, "history": log})


def main():
    ap = argparse.ArgumentParser(description="Student baselines trained with hard labels only")
    ap.add_argument("--kind", default="all", choices=KINDS + ["all"])
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    for k in (KINDS if a.kind == "all" else [a.kind]):
        train_kind(k, a)


if __name__ == "__main__":
    main()

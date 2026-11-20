import argparse

import torch
import torch.nn.functional as F

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.engine import pick_device, run_training, score, store_weights, write_json
from edgekit.teacher import ResNet50Cifar


def main():
    ap = argparse.ArgumentParser(description="ResNet-50 teacher on CIFAR-100")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="random data, only checks the code path")
    a = ap.parse_args()

    torch.manual_seed(1)
    data = make_loaders("cifar100", a.bs, a.smoke, workers=a.workers)
    net = ResNet50Cifar(100)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-50 has {n_params / 1e6:.1f}M parameters")

    log = run_training(net, data.train, data.test, a.epochs, a.lr, a.wd,
                       lambda xb, yb: F.cross_entropy(net(xb).float(), yb), tag="teacher")
    final = score(net, data.test, pick_device())
    print("test result", final)

    store_weights(paths.weight_file("teacher_resnet50"), net, epochs=a.epochs, top1=final["top1"])
    write_json(paths.log_file("teacher.json"),
               {"final": final, "epochs": a.epochs, "params": n_params, "history": log})


if __name__ == "__main__":
    main()

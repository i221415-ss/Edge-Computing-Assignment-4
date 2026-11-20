import argparse

import torch

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.distill import Distiller
from edgekit.engine import pick_device, run_training, score, store_weights, write_json
from edgekit.students import KINDS, create_student
from edgekit.teacher import ResNet50Cifar


def load_teacher(device):
    saved = torch.load(paths.weight_file("teacher_resnet50"), map_location="cpu")
    t = ResNet50Cifar(100)
    t.load_state_dict(saved["weights"])
    t.to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def distill_kind(kind, teacher, a):
    torch.manual_seed(1)
    data = make_loaders("cifar100", a.bs, a.smoke, workers=a.workers)
    student = create_student(kind, 100)
    n_params = sum(p.numel() for p in student.parameters())
    loss_fn = Distiller(student, teacher, a.temperature, a.alpha, a.beta)
    print(f"--- distilling {kind}: T={a.temperature} alpha={a.alpha} beta={a.beta}")

    log = run_training(student, data.train, data.test, a.epochs, a.lr, a.wd, loss_fn,
                       tag=f"{kind}/distilled")
    final = score(student, data.test, pick_device())
    print(kind, "distilled", final)

    store_weights(paths.weight_file(f"{kind}_distilled"), student, epochs=a.epochs, top1=final["top1"],
                  temperature=a.temperature, alpha=a.alpha, beta=a.beta)
    write_json(paths.log_file(f"{kind}_distilled.json"),
               {"final": final, "epochs": a.epochs, "params": n_params, "temperature": a.temperature,
                "alpha": a.alpha, "beta": a.beta, "history": log})


def main():
    ap = argparse.ArgumentParser(description="Distil the ResNet-50 teacher into the small students")
    ap.add_argument("--kind", default="all", choices=KINDS + ["all"])
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.6, help="share of the soft target loss")
    ap.add_argument("--beta", type=float, default=300.0, help="attention transfer weight, 0 turns it off")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    teacher = load_teacher(pick_device())
    for k in (KINDS if a.kind == "all" else [a.kind]):
        distill_kind(k, teacher, a)


if __name__ == "__main__":
    main()

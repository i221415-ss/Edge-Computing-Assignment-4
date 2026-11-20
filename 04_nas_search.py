import argparse
import json

import torch
import torch.nn.functional as F

from edgekit import paths
from edgekit.cifar import make_loaders
from edgekit.engine import pick_device, run_training, score, write_json
from edgekit.evolution import Evolution, Limits
from edgekit.searchspace import build_from_genome


def main():
    ap = argparse.ArgumentParser(description="Hardware aware evolutionary architecture search on CIFAR-10")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3, help="epochs used to rank one candidate")
    ap.add_argument("--subset", type=int, default=12000, help="training images used during search")
    ap.add_argument("--holdout", type=int, default=5000, help="validation images taken from the train split")
    ap.add_argument("--max_latency", type=float, default=12.0, help="ms per image, CPU, batch 1")
    ap.add_argument("--max_size", type=float, default=1.2, help="MB of stored weights")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    data = make_loaders("cifar10", 128, a.smoke, holdout=48 if a.smoke else a.holdout,
                        limit=a.subset, workers=a.workers, seed=a.seed)
    device = pick_device()

    def train_eval(genome):
        net = build_from_genome(genome, 10)
        run_training(net, data.train, None, a.epochs, 0.1, 5e-4,
                     lambda xb, yb: F.cross_entropy(net(xb).float(), yb), tag="cand")
        return score(net, data.val, device)["top1"]

    evo = Evolution(train_eval, Limits(a.max_latency, a.max_size),
                    paths.log_file("nas_history.csv"), paths.log_file("nas_history.json"),
                    pop_size=a.pop, generations=a.generations, threads=a.threads, seed=a.seed)
    best = evo.run()

    trained = [r for r in evo.rows if r["status"] == "trained"]
    result = {"genome": best["genome"],
              "metrics": {"val_acc": round(best["acc"], 2), "fitness": round(best["fitness"], 3),
                          "latency_ms": round(best["probe"]["latency_ms"], 3),
                          "size_mb": round(best["probe"]["size_mb"], 3), "params": best["probe"]["params"],
                          "peak_mem_mb": best["probe"]["peak_mem_mb"]},
              "limits": {"max_latency_ms": a.max_latency, "max_size_mb": a.max_size},
              "evaluated": len(evo.rows), "rejected": len(evo.rows) - len(trained)}
    write_json(paths.log_file("best_genome.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

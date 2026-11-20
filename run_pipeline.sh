#!/usr/bin/env bash
# Full run, in the order the deliverables depend on each other.
# Append --smoke to a python command to try it on random data first.
set -e
cd "$(dirname "$0")"

python 01_train_teacher.py --epochs 35
python 02_train_plain.py   --kind all --epochs 45
python 03_distill.py       --kind all --epochs 45

python 04_nas_search.py    --pop 12 --generations 8 --epochs 3 --max_latency 12 --max_size 1.2
python 05_train_nas_best.py --epochs 50

python 06_export_onnx.py
python 07_predict.py       --kind all --variant both --evaluate --runtime torch
python 07_predict.py       --kind all --variant both --evaluate --runtime onnx
python 08_benchmark.py

python tools/build_report.py

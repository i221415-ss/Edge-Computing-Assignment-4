# Assignment 4: Edge Computing and Intelligent Systems (i221415)

Q1 distils a ResNet-50 teacher (CIFAR-100) into MobileNetV2, SqueezeNet and EfficientNet-Lite0.
Q2 runs an evolutionary, hardware aware architecture search on CIFAR-10, trains the winner, and deploys it through ONNX Runtime.

## Install

```
pip install -r requirements.txt
```

Python 3.9+. Training needs a GPU to finish in reasonable time (a free Colab T4 works). CIFAR is downloaded automatically into `data/`. All latency and memory numbers are taken on CPU.

## Layout

```
edgekit/        library code (data, models, losses, search, export helpers)
01 ... 08       one script per stage, run in numeric order
tools/          build_report.py
weights/        checkpoints (.pt)
onnx/           exported models
logs/           json and csv results, logs/screenshots for report images
run_pipeline.sh everything in one go
```

## Running it

| Step | Command | Output |
|------|---------|--------|
| Teacher | `python 01_train_teacher.py` | `weights/teacher_resnet50.pt`, `logs/teacher.json` |
| Plain students | `python 02_train_plain.py` | `weights/<kind>_plain.pt` |
| Distillation | `python 03_distill.py` | `weights/<kind>_distilled.pt` |
| NAS | `python 04_nas_search.py` | `logs/nas_history.csv/json`, `logs/best_genome.json` |
| Train NAS winner | `python 05_train_nas_best.py` | `weights/nas_best.pt` |
| ONNX export | `python 06_export_onnx.py` | `onnx/*.onnx` |
| Inference | `python 07_predict.py ...` | printed predictions / accuracy |
| Benchmark | `python 08_benchmark.py` | `logs/benchmark_results.csv` |
| Report | `python tools/build_report.py` | `Report_i221415.pdf` |

`<kind>` is `mobilenetv2`, `squeezenet` or `efficientnet_lite0`.

Useful options

- `03_distill.py --temperature 3 --alpha 0.6 --beta 300` (`--beta 0` removes the feature term)
- `04_nas_search.py --max_latency 12 --max_size 1.2` (ms per image on CPU, MB of weights). Pick limits that make sense for your machine.
- `07_predict.py --kind squeezenet --variant distilled --image cat.png`
- `07_predict.py --kind nas --image truck.png --runtime onnx`
- `07_predict.py --kind all --variant both --evaluate --runtime onnx`

Every training script and the NAS search also take `--smoke` which swaps CIFAR for a few random images, handy for checking that the chain runs end to end before a long job.

For the report, save screenshots of your terminal runs as PNG files in `logs/screenshots/` and run the report script again.

## Notes

- Models are fed 32x32 images; the searched resolution is applied inside the NAS model so all models share one input shape.
- Candidates that break the latency or size limit are written to the history as `rejected` and are not trained.
- Hardware probes run in a separate process so memory readings are not polluted by training.

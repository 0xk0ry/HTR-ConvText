# HTR-ConvText

## Introduction
HTR-ConvText is a research codebase for mixed-masking handwritten text recognition (HTR). The model couples a convolutional backbone with a Textual Context Module (TCM) that learns masked-span language priors and is trained end-to-end with CTC loss. The training recipe mirrors the paper setup: strong grayscale augmentations, mixed span/block/random masking, SAM optimization, EMA tracking, TensorBoard/W&B logging, and exportable checkpoints (`checkpoints/` and `weights/`). This repository provides reference training, validation, evaluation, and visualization scripts for IAM, READ2016, LAM, and VNOnDB line-level datasets.

## Installation & Dependencies
1. **Clone the repository**
   ```cmd
   git clone https://github.com/0xk0ry/HTR-ConvText.git
   cd HTR-ConvText
   ```
2. **Create and activate a Python 3.9+ environment** (venv, Conda, or your preferred tool)
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install PyTorch** using the wheel that matches your CUDA driver (swap the index for CPU-only builds):
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
4. **Install the remaining project requirements** (everything except PyTorch, which you already picked in step 3).
   ```cmd
   pip install -r requirements.txt
   ```
   This list intentionally omits PyTorch so you can choose the correct CUDA/CPU wheel, but it does cover Pillow, OpenCV, scikit-image, NumPy, TensorBoard, and W&B so every script (`train.py`, `test.py`, `visualize_convtext.py`, etc.) can run without hunting for missing modules.

## Datasets
**Provided split manifests.** Ready-to-use `train.ln`, `val.ln`, and `test.ln` files already live under `data/{iam,read2016,lam,vnondb}/`. Point `--*-data-list` to these paths or duplicate them into your own dataset directory structure.

**Expected file layout.** Every line sample pairs a grayscale crop and a UTF-8 transcription with matching stems:

```
<root>/lines/a01/a01-000u_00.png   # line crop
<root>/lines/a01/a01-000u_00.txt   # same stem, single-line text
```

Each `.ln` file contains forward-slash relative image paths. `data/dataset.py` prepends `--data-path` at runtime, so a list entry should look like `/lines/a01/a01-000u_00.png`.

**Download links.**
<details>
   <summary>IAM</summary>
   Register at the FKI website (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and download the archive from https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database. Extract the `lines/` folder and reuse the manifests in `data/iam/`.
</details>
<details>
   <summary>READ2016</summary>
   ```
   wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
   ```
   Unpack both tarballs, merge their line images into a common root, and map them using the `data/read2016/*.ln` files.
</details>
<details>
   <summary>LAM</summary>
   Download the Lam dataset from https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=46, extract the line images, and reuse the split manifests under `data/LAM/`.
</details>

**Supported alphabets (`--dataset`).** Character sets and class counts are pre-defined:

| Flag | Description | `nb_cls` |
| --- | --- | --- |
| `iam` | IAM Handwriting Database | 80 |
| `read2016` | ICDAR 2016 READ lines | 90 |
| `lam` | Lam Vietnamese corpus | 91 |
| `vnondb` | VNOnDB historical Vietnamese | 162 |

**Custom data checklist.**
1. Arrange line crops and `.txt` files together (one transcript per image).
2. Generate `*.ln` lists pointing to the images; follow the provided templates if you need a reference.
3. Set `--data-path`, `--train-data-list`, `--val-data-list`, and `--test-data-list` accordingly.
4. Extend `data/dataset.py` with your alphabet if it differs from the presets.

## Quick Start
1. **Train (example: IAM recipe, adapted from `run/iam.sh`)**
   ```cmd
   python train.py ^
       --dataset iam ^
       --data-path D:/datasets/iam/lines/ ^
       --train-data-list D:/datasets/iam/train.ln ^
       --val-data-list D:/datasets/iam/val.ln ^
       --tcm-enable --use-masking --use-wandb ^
       --exp-name htr-convtext-iam ^
       --train-bs 32 --val-bs 8 --img-size 512 64 ^
       --max-lr 1e-3 --warm-up-iter 1000 --total-iter 100000
   ```
   Logs, TensorBoard summaries, and checkpoints will be stored under `output/htr-convtext-iam/` (or the directory given by `--out-dir`).

2. **Evaluate a checkpoint**
   ```cmd
   python test.py ^
       --dataset iam ^
       --data-path D:/datasets/iam/lines/ ^
       --train-data-list D:/datasets/iam/train.ln ^
       --test-data-list D:/datasets/iam/test.ln ^
       --val-bs 8 --resume checkpoints/htr-context-8-12-nahn/best_CER.pth
   ```
   The script reports CER/WER and also produces `predictions.json` with per-sample metrics.

3. **Validate during training**
   `valid.py` is automatically invoked every `--eval-iter` iterations and logs CER/WER. You can also run it manually in isolation if you load a checkpoint through `utils.utils.load_checkpoint`.

4. **Inspect predictions**
   Use `visualize_convtext.py` (or `visualize/`) to overlay predicted spans or compare hypotheses:
   ```cmd
   python visualize_convtext.py --image D:/datasets/iam/lines/a04-089_326.png --checkpoint weights/iam.pth
   ```

For additional hyper-parameter presets see the helper scripts under `run/` (IAM, LAM, READ2016, VNOnDB). Adjust `--nb-cls`, masking ratios, and augmentation parameters as needed for your dataset.

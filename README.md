# SF-UniDABench Supplementary Code

### 📌 ***Additional Tables with standard deviation are in folder*** ``tables``


This repository contains the supplementary code for the SF-UniDABench experiments. It includes source-model pretraining, SF-UniDA adaptation runs, threshold analyses, and scripts to generate summary tables.

Run commands from the repository root unless stated otherwise.

## ⚠️ Important Notes

- The datasets are **not** included in this repository. Download the archive from [Google Drive](https://drive.google.com/file/d/1WV6ENJjGHpDNXY7eM_pPMMQqEfwBwJh_/view?usp=drive_link), extract it, and place the dataset folders so the tree looks like:
  ```
  data/
  ├── HAR/
  │   ├── train_0.pt
  │   ├── test_0.pt
  │   └── ...
  ├── HHAR/
  │   └── ...
  └── EEG/
      └── ...
  ```
  The scripts assume this layout and will look for data under `data/` by default.
- The architecture referred to as **TFE** in the paper corresponds to `FNO` in this codebase.
- The dataset referred to as **EDF** in the paper corresponds to `EEG` in this codebase.
- Edit the arrays (BACKBONES, METHODS and DATASETS) inside the scripts to restrict datasets, backbones, or methods.
- Run scripts from the repository root.
- If using Docker, build the image first with `docker build -t sf_unidabench:latest .`.
- We provide the hyperparameters used for source pretraining and adaptation, but Docker does not guarantee bitwise-identical results across machines. Results can still vary with hardware, drivers, Linux distribution, and low-level numerical libraries. The Docker setup is provided to make the code easy to run on any computer with Docker and NVIDIA GPU support installed.

## Repository Organization

```text
algorithms/              Domain adaptation methods
configs/                 Dataset/model configs and cached hyperparameters
dataloader/              Dataset loading utilities
models/                  Backbone and classifier definitions
trainers/                Training loops
data/                    Dataset root (HAR/, HHAR/, EEG/ — download separately)
scripts/conda/           Scripts for an activated local Python environment
scripts/docker_script_exec.sh
                         Helper to run scripts/conda scripts inside Docker
scripts/tables/          Table generation scripts
pretrain_source_models.py
pretrain_source_models_umad.py
main.py                  Main adaptation entry point
requirements.txt         Main Python dependencies
requirements_no_deps.txt Packages installed without dependency resolution
Dockerfile               Reproducible Docker environment
```

## Python Environment

We recommend Python 3.11. With Conda:

```bash
conda create -n sf-unidabench python=3.11 -y
conda activate sf-unidabench
pip install -r requirements.txt
pip install --no-deps -r requirements_no_deps.txt
```

The second requirements file is needed because `momentfm==0.1.4` pins an older `huggingface-hub`, while this environment uses the newer version from `requirements.txt`.

You can also use the helper script:

```bash
bash scripts/setup_conda_env.sh sf-unidabench
conda activate sf-unidabench
```

## Docker Environment

To build the Docker image locally:

```bash
docker build -t sf_unidabench:latest .
```

The helper below runs any script from `scripts/conda/` inside that Docker image, mounting the repository at `/workspace`:

```bash
bash scripts/docker_script_exec.sh scripts/conda/pretrain_all_source_models.sh
```

It uses GPU access, your host UID/GID, a large shared-memory allocation, and `HF_HOME=/workspace/.hf_cache`.

## 1. Source Pretraining

Always pretrain source models before running adaptation. Source checkpoints are reused by the SF-UniDA methods.

One backbone/dataset at a time:

```bash
python pretrain_source_models.py \
  --backbone CNN \
  --dataset HAR \
  --data_path data/ \
  --device cuda \
  --force
```

For UMAD source models:

```bash
python pretrain_source_models_umad.py \
  --backbone CNN \
  --dataset HAR \
  --data_path data/ \
  --device cuda
```

To pretrain all source models declared in the scripts:

```bash
bash scripts/conda/pretrain_all_source_models.sh
bash scripts/conda/pretrain_all_source_models_umad.sh
```

To run the same scripts inside Docker:

```bash
bash scripts/docker_script_exec.sh scripts/conda/pretrain_all_source_models.sh
bash scripts/docker_script_exec.sh scripts/conda/pretrain_all_source_models_umad.sh
```

## 2. SF-UniDA Adaptation

Run one dataset/backbone/method combination directly with `main.py`:

```bash
python main.py \
  --dataset HAR \
  --backbone CNN \
  --da_method GLC \
  --data_path data/ \
  --save_dir logs_test_mlsp26/run_sf_unida_auto \
  --exp_name EXP1 \
  --num_runs 10 \
  --auto_threshold \
  --threshold_method yen \
  --hparams_json configs/best_hparams/sfunida/best_hparams_CNN.json
```

To run the combinations declared in the script:

```bash
bash scripts/conda/run_sf_unida.sh
```

To run it inside Docker:

```bash
bash scripts/docker_script_exec.sh scripts/conda/run_sf_unida.sh
```

## 3. Threshold Experiments

Threshold comparison:

```bash
bash scripts/conda/run_threshold_comparison.sh
```

Threshold sensitivity:

```bash
bash scripts/conda/run_threshold_sensitivity.sh
```

Docker versions:

```bash
bash scripts/docker_script_exec.sh scripts/conda/run_threshold_comparison.sh
bash scripts/docker_script_exec.sh scripts/conda/run_threshold_sensitivity.sh
```

## 4. Tables and Result Overview

After the runs finish, generate summary tables with:

```bash
bash scripts/tables/example_tables.sh
```

This recomputes `results2.csv`, generates LaTeX summary tables, and compiles the tables to images under `tables/`.

The PNG compilation step requires a local LaTeX installation with `pdflatex`.
This dependency is not included in `requirements.txt` or in the Docker image.
If `pdflatex` is not available, the scripts still print readable tables in the
command line and write the `.tex` table files.

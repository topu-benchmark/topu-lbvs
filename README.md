# TopU-LBVS

Multi-target benchmark for ligand-based virtual screening (LBVS) under hard-negative screening conditions. NeurIPS 2026 Datasets and Benchmarks track.

**Paper**: *TopU-LBVS: A Realistic Multi-Target Benchmark for Ligand-Based Virtual Screening*
**Data**: https://huggingface.co/datasets/topu-benchmark/topu-lbvs
**License**: MIT (code) · CC-BY-SA-4.0 (data)

---

## What's in this benchmark

- **93 protein targets** across 7 protein classes (cytochromes, GPCRs, ion channels, kinases, nuclear receptors, proteases, miscellaneous enzymes)
- **Hard-negative TopU libraries** — decoys are property-matched *and* structurally similar to actives, selected via a constrained genetic algorithm so that simple physicochemical filters and nearest-neighbour retrieval cannot separate them from actives
- **Fixed 1:40 active-to-decoy ratio** in test sets
- **Three evaluation protocols**:
  - **TopU-LBVS-full**: train on historical ChEMBL\* SAR, evaluate on hard TopU library (93 targets)
  - **TopU-LBVS-few**: few-shot, train and test both drawn from TopU (93 targets, tier 1 / tier 2)
  - **TopU-LBVS-mini**: 7-target compact protocol with paired random-decoy control
- **10 reference baselines**: Morgan-RF, Tanimoto-NN, GIN(+FP), GAT(+FP), GPS(+FP), D-MPNN (Chemprop), MolFormer

---

## Repository layout

```
.
├── run_setting1.py          # CLI: TopU-LBVS-full
├── run_setting2.py          # CLI: TopU-LBVS-few
├── run_setting3.py          # CLI: TopU-LBVS-mini (random-decoy control)
├── tune_gnn.py              # GNN hyperparameter grid search
├── create_setting2_data.py  # Builds the few-shot s2 splits
├── create_setting3_data.py  # Builds the random-decoy s3 splits
├── aggregate_results*.py    # Aggregate per-target results into tables
├── data/loader.py           # NPZ / split loaders
├── metrics/screening.py     # EF@k%, PR-AUC, ROC-AUC, BEDROC, LogAUC
├── models/                  # Baseline models
│   ├── morgan_rf.py
│   ├── tanimoto_nn.py
│   ├── gin.py / gat.py / gps.py
│   ├── gnn_common.py        # Shared backbone + late-fusion FP head
│   ├── dmpnn.py             # Chemprop v2 wrapper
│   └── molformer.py         # IBM MoLFormer fine-tuning
├── training/runner.py       # Per-target train+evaluate loop
├── utils/mol_utils.py       # SMILES → graph utilities
├── requirements.txt
├── CITATION.cff
└── LICENSE                  # MIT
```

---

## Installation

```bash
git clone https://github.com/topu-benchmark/topu-lbvs.git
cd topu-lbvs
pip install -r requirements.txt
```

Tested on Python 3.10–3.11 with CUDA 12.x. RDKit and PyTorch Geometric have CUDA-specific wheels — see their docs if you hit install errors.

---

## Getting the data

Download the benchmark data from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download topu-benchmark/topu-lbvs --repo-type dataset --local-dir ./data
```

The expected layout after download:

```
data/
├── topu-lbvs-full/         # ChEMBL* training pool + TopU library per target
├── topu-lbvs-few-tier1/    # Few-shot splits, < 50 actives (46 targets)
├── topu-lbvs-few-tier2/    # Few-shot splits, ≥ 50 actives (47 targets)
└── topu-lbvs-mini/         # 7 targets with s1/ and s3/ subfolders
```

Point the run scripts at this directory via `--dataset_dir ./data/topu-lbvs-full` (and so on per protocol).

---

## Quickstart

### TopU-LBVS-full (main benchmark, 93 targets)

```bash
# Single target, all 3 seeds, no wandb
python run_setting1.py --model gin --target egfr --no_wandb

# All targets, single seed
python run_setting1.py --model morgan_rf --seeds 2026

# With per-target tuned hyperparameters from tune_gnn.py output
python run_setting1.py --model gin \
    --tuned_params_dir ./results/tuning/setting1
```

### TopU-LBVS-few (few-shot, 93 targets)

```bash
python run_setting2.py --model gat --tier 1 --no_wandb
python run_setting2.py --model dmpnn --tier 2 --no_wandb
```

### TopU-LBVS-mini + random-decoy control

```bash
# TopU hard-decoy test
python run_setting1.py --model gin --target aa2ar --no_wandb
# Paired random-decoy test (same training, random ChEMBL* decoys at test time)
python run_setting3.py --model gin --target aa2ar --no_wandb
```

### Available models

`morgan_rf`, `tanimoto_nn`, `gin`, `ginfp`, `gat`, `gatfp`, `gps`, `gpsfp`, `dmpnn`, `molformer`

---

## Evaluation metrics

Computed per target by `metrics/screening.py`, then aggregated:

- **EF@1%, EF@5%, EF@10%** — enrichment factor at top-k% (primary metric varies by protocol)
- **PR-AUC** — used for model selection on the validation set
- **ROC-AUC, BEDROC (α=20), LogAUC** — secondary metrics reported in the appendix

Primary test metrics:
- TopU-LBVS-full: **EF@1%**
- TopU-LBVS-few: **EF@10%** (EF@1% is unstable on the smallest tier-1 libraries)
- TopU-LBVS-mini: **EF@1%**

---

## Reproducing paper numbers

```bash
# Setting 1 (full benchmark, all 93 targets, all 3 seeds)
for model in morgan_rf tanimoto_nn gin ginfp gat gatfp gps gpsfp dmpnn molformer; do
    python run_setting1.py --model $model
done
python aggregate_results.py
```

Expected wall-clock per model on a single A100: ~30 min for fingerprint baselines, 6–12 hr for D-MPNN/MolFormer across all 93 targets × 3 seeds.

---

## Citation

```bibtex
@inproceedings{kumar2026topu,
  title     = {TopU-LBVS: A Realistic Multi-Target Benchmark for Ligand-Based Virtual Screening},
  author    = {Kumar, Surbhi and Zhou, Yuhe and Shiralkar, Varun and Huang, Niu and Coskunuzer, Baris},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2026}
}
```

---

## License

- **Code** (this repository): MIT — see `LICENSE`
- **Data** (Hugging Face dataset): CC-BY-SA-4.0

---

## Contact

Issues and questions: please use the [GitHub issue tracker](https://github.com/topu-benchmark/topu-lbvs/issues).

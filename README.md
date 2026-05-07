# TopU-LBVS

Multi-target benchmark for ligand-based virtual screening (LBVS) under hard-negative conditions. 93 protein targets across 7 protein classes, with property-matched and structurally similar decoys at a fixed 1:40 active-to-decoy ratio.

*Submitted to the NeurIPS 2026 Datasets and Benchmarks track (under review).*

- **Data**: https://huggingface.co/datasets/topu-benchmark/topu-lbvs
- **License**: MIT (code) · CC-BY-SA-4.0 (data)

---

## Three evaluation protocols

| Protocol             | Description                                                |
|----------------------|------------------------------------------------------------|
| `topu-lbvs-full`     | Train on ChEMBL\* SAR, test on TopU library (93 targets)   |
| `topu-lbvs-few`      | Few-shot, train and test from TopU (93 targets, 2 tiers)   |
| `topu-lbvs-mini`     | 7-target compact protocol with paired random-decoy control |

Primary metric: **EF@1%** for `full` and `mini`, **EF@10%** for `few`. Validation: PR-AUC.

---

## Installation

```bash
git clone https://github.com/topu-benchmark/topu-lbvs.git
cd topu-lbvs
pip install -r requirements.txt
```

Tested on Python 3.10–3.11 with CUDA 12.x.

## Get the data

```bash
huggingface-cli download topu-benchmark/topu-lbvs --repo-type dataset --local-dir ./data
```

## Run

```bash
# Setting 1 — main benchmark
python run_setting1.py --model gin --target egfr --no_wandb

# Setting 2 — few-shot
python run_setting2.py --model gat --tier 1 --no_wandb

# Setting 3 — random-decoy control (paired with Setting 1)
python run_setting3.py --model gin --target aa2ar --no_wandb
```

Available models: `morgan_rf`, `tanimoto_nn`, `gin`, `ginfp`, `gat`, `gatfp`, `gps`, `gpsfp`, `dmpnn`, `molformer`.

WandB is optional — disable with `--no_wandb`, or set `--wandb_entity your-entity --wandb_project your-project`.

---

## Citation

```bibtex
@unpublished{topu_lbvs_2026,
  title  = {TopU-LBVS: A Realistic Multi-Target Benchmark for Ligand-Based Virtual Screening},
  author = {Kumar, Surbhi and Zhou, Yuhe and Shiralkar, Varun and Huang, Niu and Coskunuzer, Baris},
  year   = {2026},
  note   = {Under review at NeurIPS 2026 Datasets and Benchmarks Track}
}
```

Issues and questions: please use the [GitHub issue tracker](https://github.com/topu-benchmark/topu-lbvs/issues).

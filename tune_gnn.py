import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
from sklearn.metrics import average_precision_score

from data.loader import discover_targets, get_train_val, get_train_val_setting2, get_train_val_setting3
from models.base import BaseModel
from models.gat import GAT, GATFP
from models.gin import GIN, GINFP
from models.gps import GPS, GPSFP


DEFAULT_DATASET_DIR = "./topu_dataset"
DEFAULT_SPLITS_DIR = "./splits"
DEFAULT_TUNING_DIR = "./results/tuning"

MODEL_REGISTRY: Dict[str, Tuple[Type[BaseModel], Dict]] = {
    "gat": (GAT, {}),
    "gatfp": (GATFP, {}),
    "gin": (GIN, {}),
    "ginfp": (GINFP, {}),
    "gps": (GPS, {}),
    "gpsfp": (GPSFP, {}),
}

FIXED_N_LAYERS = 5
FIXED_HIDDEN_DIM = 256
FIXED_HEADS = 4
FIXED_LR = 1e-3
FIXED_WEIGHT_DECAY = 1e-4
FIXED_BATCH_SIZE = 64
FIXED_MAX_EPOCHS = 100
FIXED_PATIENCE = 20
FIXED_LR_PATIENCE = 5
FIXED_LR_FACTOR = 0.5
GRID_DROPOUT_VALUES = [0.3, 0.5]
GRID_FP_HIDDEN_DIM = [128, 256, 512]
GRID_FUSION_HIDDEN_DIM = [64, 128, 256]





def parse_args():
    parser = argparse.ArgumentParser(
        description="Deterministic grid sweep for GNN and late-fusion FP variants."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--target",
        default=None,
        help="Single target name. Omit to tune all discovered targets.",
    )
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--tuning_dir", default=DEFAULT_TUNING_DIR)
    parser.add_argument("--dropout", type=float, default=None,
                        help="Pin dropout to one value. When set, overrides grid.")
    parser.add_argument("--fp_hidden_dim", type=int, default=None,
                        help="Pin fp_hidden_dim to one value (FP variants only).")
    parser.add_argument("--fusion_hidden_dim", type=int, default=None,
                        help="Pin fusion_hidden_dim to one value (FP variants only).")
    parser.add_argument("--out_suffix", type=str, default=None,
                        help="When set, write trial_{out_suffix}.json instead of best_params.json. "
                             "Used for per-config parallel tuning.")
    parser.add_argument("--consolidate", action="store_true",
                        help="Instead of running, scan {tuning_dir}/setting{N}/{model}/{target}/ "
                             "for trial_*.json files and emit best_params.json + trials.csv.")
    parser.add_argument("--seed", type=int, default=2026, help="Split/model seed.")
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2, 3],
                        help="Which evaluation setting to load data for.")
    return parser.parse_args()


def _base_fixed_params(model_key: str) -> Dict:
    params = {
        "hidden_dim": FIXED_HIDDEN_DIM,
        "n_layers": FIXED_N_LAYERS,
        "lr": FIXED_LR,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "batch_size": FIXED_BATCH_SIZE,
        "max_epochs": FIXED_MAX_EPOCHS,
        "patience": FIXED_PATIENCE,
        "lr_patience": FIXED_LR_PATIENCE,
        "lr_factor": FIXED_LR_FACTOR,
    }
    if model_key.startswith("gat") or model_key.startswith("gps"):
        params["heads"] = FIXED_HEADS
    return params


def grid_configs(
    model_key: str,
    dropout_override: float = None,
    fp_hidden_override: int = None,
    fusion_hidden_override: int = None,
) -> List[Dict]:
    base = _base_fixed_params(model_key)
    configs: List[Dict] = []

    dropouts = [dropout_override] if dropout_override is not None else GRID_DROPOUT_VALUES
    fp_hiddens = [fp_hidden_override] if fp_hidden_override is not None else GRID_FP_HIDDEN_DIM
    fusion_hiddens = [fusion_hidden_override] if fusion_hidden_override is not None else GRID_FUSION_HIDDEN_DIM

    for dropout in dropouts:
        if model_key.endswith("fp"):
            for fp_hidden_dim, fusion_hidden_dim in product(fp_hiddens, fusion_hiddens):
                cfg = dict(base)
                cfg["dropout"] = float(dropout)
                cfg["fp_hidden_dim"] = int(fp_hidden_dim)
                cfg["fusion_hidden_dim"] = int(fusion_hidden_dim)
                configs.append(cfg)
        else:
            cfg = dict(base)
            cfg["dropout"] = float(dropout)
            configs.append(cfg)

    return configs


def _to_trial_row(number: int, value: float, params: Dict) -> Dict:
    row = {
        "number": number,
        "value": value,
        "state": "COMPLETE",
    }
    for key, val in params.items():
        row[f"params_{key}"] = val
    return row


def _write_trials_csv(path: Path, trial_rows: List[Dict]) -> None:
    if not trial_rows:
        path.write_text("number,value,state\n", encoding="utf-8")
        return

    fieldnames = ["number", "value", "state"]
    param_fields = sorted(
        {key for row in trial_rows for key in row.keys() if key.startswith("params_")}
    )
    fieldnames.extend(param_fields)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)


def _run_grid_target(
    model_cls: Type[BaseModel],
    model_key: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    smiles_train: np.ndarray,
    smiles_val: np.ndarray,
    dropout_override: float = None,
    fp_hidden_override: int = None,
    fusion_hidden_override: int = None,
) -> Tuple[float, Dict, List[Dict]]:
    configs = grid_configs(
        model_key=model_key,
        dropout_override=dropout_override,
        fp_hidden_override=fp_hidden_override,
        fusion_hidden_override=fusion_hidden_override,
    )
    best_value = -np.inf
    best_params: Dict = {}
    trial_rows: List[Dict] = []

    for idx, cfg in enumerate(configs):
        kwargs = dict(cfg)
        kwargs["use_wandb"] = False
        model = model_cls(seed=seed, **kwargs)

        fit_start = time.time()
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            smiles_train=smiles_train,
            smiles_val=smiles_val,
        )
        fit_seconds = time.time() - fit_start

        val_scores = model.predict_proba(X_test=X_val, smiles_test=smiles_val)
        valid_mask = np.isfinite(val_scores)
        if valid_mask.sum() == 0:
            value = 0.0
        else:
            y_true = y_val[valid_mask]
            y_score = val_scores[valid_mask]
            if np.unique(y_true).size < 2:
                value = 0.0
            else:
                value = float(average_precision_score(y_true, y_score))

        trial_rows.append(_to_trial_row(number=idx, value=value, params=cfg))
        if value > best_value:
            best_value = value
            best_params = dict(cfg)

    if best_value == -np.inf:
        best_value = 0.0
    return best_value, best_params, trial_rows

def _best_payload(model_key: str, target: str, seed: int, best_value: float, best_params: Dict, n_evals: int) -> Dict:
    return {
        "model": model_key,
        "target": target,
        "seed": seed,
        "n_trials_total": n_evals,
        "best_value": float(best_value),
        "best_params": best_params,
    }


def tune_target(
    model_cls: Type[BaseModel],
    model_key: str,
    target: str,
    seed: int,
    setting: int,
    dataset_dir: str,
    splits_dir: str,
    tuning_dir: str,
    dropout_override: float = None,
    fp_hidden_override: int = None,
    fusion_hidden_override: int = None,
    out_suffix: str = None,
):
    model_cls, default_kwargs = MODEL_REGISTRY[model_key]
    if default_kwargs:
        raise ValueError(f"Unexpected default kwargs for {model_key}: {default_kwargs}")

    load_start = time.time()
    if setting == 1:
        X_train, y_train, _, smiles_train, X_val, y_val, _, smiles_val, _ = get_train_val(
            dataset_dir=dataset_dir, splits_dir=splits_dir,
            target=target, seed=seed, include_smiles=True,
        )
    elif setting == 2:
        X_train, y_train, _, smiles_train, X_val, y_val, _, smiles_val = get_train_val_setting2(
            dataset_dir=dataset_dir, target=target, seed=seed, include_smiles=True,
        )
    elif setting == 3:
        X_train, y_train, _, smiles_train, X_val, y_val, _, smiles_val = get_train_val_setting3(
            dataset_dir=dataset_dir, target=target, seed=seed, include_smiles=True,
        )
    else:
        raise ValueError(f"Unknown setting: {setting}")
   

    out_dir = Path(tuning_dir) / f"setting{setting}" / model_key / target
    out_dir.mkdir(parents=True, exist_ok=True)

    best_value, best_params, trial_rows = _run_grid_target(
        model_cls=model_cls,
        model_key=model_key,
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        smiles_train=smiles_train,
        smiles_val=smiles_val,
        dropout_override=dropout_override,
        fp_hidden_override=fp_hidden_override,
        fusion_hidden_override=fusion_hidden_override,
    )

    payload = _best_payload(
        model_key=model_key,
        target=target,
        seed=seed,
        best_value=best_value,
        best_params=best_params,
        n_evals=len(trial_rows),
    )

    if out_suffix is not None:
        # Per-config mode: write a single-trial file. Consolidation happens later.
        trial_path = out_dir / f"trial_{out_suffix}.json"
        with open(trial_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    else:
        # Full-grid mode (original behavior): write canonical best + trials.csv.
        with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        _write_trials_csv(out_dir / "trials.csv", trial_rows=trial_rows)
    print(
        f"[{model_key}/{target}] best PR-AUC={best_value:.5f} "
        f"with params={best_params}"
    )

def _consolidate(
    tuning_dir: str,
    model_key: str,
    target: str,
    setting: int,
) -> None:
    """Scan trial_*.json files for (model, target), pick best, write canonical outputs."""
    out_dir = Path(tuning_dir) / f"setting{setting}" / model_key / target
    if not out_dir.is_dir():
        print(f"[consolidate] no directory for {model_key}/{target}: {out_dir}")
        return

    trial_files = sorted(out_dir.glob("trial_*.json"))
    if not trial_files:
        print(f"[consolidate] no trial_*.json files in {out_dir}")
        return

    payloads = []
    trial_rows = []
    for idx, tf in enumerate(trial_files):
        try:
            with open(tf, encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as e:
            print(f"[consolidate] failed to read {tf}: {e}")
            continue

        value = float(blob.get("best_value", 0.0))
        params = blob.get("best_params", {})
        payloads.append((value, params, blob))
        trial_rows.append(_to_trial_row(number=idx, value=value, params=params))

    if not payloads:
        print(f"[consolidate] no valid trials for {model_key}/{target}")
        return

    payloads.sort(key=lambda x: x[0], reverse=True)
    best_value, best_params, best_blob = payloads[0]

    canonical = {
        "model": model_key,
        "target": target,
        "seed": best_blob.get("seed"),
        "n_trials_total": len(payloads),
        "best_value": best_value,
        "best_params": best_params,
    }

    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2, sort_keys=True)
    _write_trials_csv(out_dir / "trials.csv", trial_rows=trial_rows)
    print(
        f"[consolidate] {model_key}/{target}: best PR-AUC={best_value:.5f} "
        f"from {len(payloads)} trials, params={best_params}"
    )
    

def main():
    args = parse_args()
    targets = [args.target] if args.target else discover_targets(args.dataset_dir)

    if args.consolidate:
        for target in targets:
            _consolidate(
                tuning_dir=args.tuning_dir,
                model_key=args.model,
                target=target,
                setting=args.setting,
            )
        return

    model_cls, _ = MODEL_REGISTRY[args.model]

    for target in targets:
        tune_target(
            model_cls=model_cls,
            model_key=args.model,
            target=target,
            seed=args.seed,
            setting=args.setting,
            dataset_dir=args.dataset_dir,
            splits_dir=args.splits_dir,
            tuning_dir=args.tuning_dir,
            dropout_override=args.dropout,
            fp_hidden_override=args.fp_hidden_dim,
            fusion_hidden_override=args.fusion_hidden_dim,
            out_suffix=args.out_suffix,
        )


if __name__ == "__main__":
    main()

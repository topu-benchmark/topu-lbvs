"""
training/runner.py
TopU-LBVS Settings 1, 2, & 3 - per-target training and evaluation runner.

Runs one model on one target across all seeds. Saves raw scores and
per-seed metrics to disk, logs to wandb.

Call signature
--------------
result = run_target(
    model_cls   = GIN,
    model_kwargs= {"n_layers": 3},
    target      = "egfr",
    dataset_dir = "/path/to/topu_dataset",
    splits_dir  = "/path/to/splits",
    results_dir = "/path/to/results/setting1",
    seeds       = [2026, 2027, 2028],
    use_wandb   = True,
    wandb_entity= "my-entity",
    wandb_project="LBVS",
    setting     = 1,  # 1 = Setting 1, 2 = Setting 2 (TopU few-shot), 3 = Setting 3
)

Output layout
-------------
results_dir/{model_name}/
    raw_scores/{target}_seed{seed}.npz    <- ids, y_true, scores
    per_seed/{target}_seed{seed}.csv      <- all metrics for one seed
    errors.log                            <- appended on target skip

Returns
-------
Dict mapping seed -> metrics dict, or None if target was skipped.

Design decisions
----------------
- One model * one target * all seeds (caller loops over targets/models)
- Target-level failure: skip + log to errors.log, return None
- Seed-level failure: crash hard (raise)
- TanimotoNN (is_deterministic=True): only runs seed[0], skips rest
- wandb: one run per (model, target, seed), finished after each seed
- Raw scores saved as npz: ids (str), y_true (int8), scores (float64)
"""

import csv
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import wandb
import inspect

# CHANGED: added Setting 2 loader imports
from data.loader import (
    get_test,
    get_train_val,
    get_train_val_setting2,
    get_test_setting2,
    get_train_val_setting3,
    get_test_setting3,
)
from metrics.screening import compute_all
from models.base import BaseModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dirs(results_dir: Path, model_name: str) -> tuple:
    """
    Create output subdirectories, return (scores_dir, per_seed_dir).
    """
    scores_dir   = results_dir / model_name / "raw_scores"
    per_seed_dir = results_dir / model_name / "per_seed"
    scores_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    return scores_dir, per_seed_dir


def _save_scores(
    scores_dir: Path,
    target:     str,
    seed:       int,
    ids:        np.ndarray,
    y_true:     np.ndarray,
    scores:     np.ndarray,
) -> None:
    """
    Save raw scores to npz.

    File: {scores_dir}/{target}_seed{seed}.npz
    Keys: ids (object/str), y_true (int8), scores (float64)
    """
    path = scores_dir / f"{target}_seed{seed}.npz"
    np.savez_compressed(
        path,
        ids    = ids.astype(object),
        y_true = y_true.astype(np.int8),
        scores = scores.astype(np.float64),
    )


def _save_per_seed_metrics(
    per_seed_dir: Path,
    target:       str,
    seed:         int,
    metrics:      Dict[str, Any],
) -> None:
    """
    Save per-seed metrics as a single-row CSV.

    File: {per_seed_dir}/{target}_seed{seed}.csv
    Columns: target, seed, ef_1pct, ef_5pct, ef_10pct,
             prauc, rocauc, bedroc, logauc,
             n_actives, n_decoys, n_total
    """
    path = per_seed_dir / f"{target}_seed{seed}.csv"
    row  = {"target": target, "seed": seed, **metrics}

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _log_error(results_dir: Path, model_name: str, message: str) -> None:
    """
    Append an error message to errors.log.
    """
    log_path = results_dir / model_name / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def _needs_smiles(model_cls: Type[BaseModel]) -> bool:
    """
    Return True if model requires SMILES (GNN models).
    Checks class name against known GNN model names.
    """
    gnn_names = {"gin", "ginfp", "gat", "gatfp", "gps", "gpsfp",
                 "dmpnn", "molformer", "rdkitrf", "unimol2", "dmpnnmorgan"}
    return model_cls.__name__.lower().replace("+", "").replace("_", "") in gnn_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_target(
    model_cls:     Type[BaseModel],
    model_kwargs:  Dict[str, Any],
    target:        str,
    dataset_dir:   str,
    splits_dir:    str,
    results_dir:   str,
    seeds:         List[int]         = None,
    use_wandb:     bool              = True,
    wandb_entity:  Optional[str]     = None,
    wandb_project: Optional[str]     = "LBVS",
    setting:       int               = 1,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Train and evaluate one model on one target across all seeds.

    Parameters
    ----------
    model_cls     : BaseModel subclass (not instance)
    model_kwargs  : kwargs passed to model_cls constructor (excluding seed)
    target        : target name string, e.g. "egfr"
    dataset_dir   : path to topu_dataset directory
    splits_dir    : path to splits directory (used by Setting 1 only)
    results_dir   : root results directory
    seeds         : list of random seeds (default [2026, 2027, 2028])
    use_wandb     : whether to log to wandb
    wandb_entity  : wandb entity (team/user name)
    wandb_project : wandb project name
    setting       : 1 = Setting 1 (TopU hard decoys, ChEMBL* train),
                    2 = Setting 2 (TopU few-shot, TopU train+test),
                    3 = Setting 3 (random ChEMBL* decoys)

    Returns
    -------
    Dict mapping seed -> metrics dict if successful.
    None if the target was skipped due to a data loading error.

    Raises
    ------
    Any exception raised during model.fit() or model.predict_proba()
    propagates immediately (seed-level failures crash hard).
    """
    if seeds is None:
        seeds = [2026, 2027, 2028]

    dataset_dir = Path(dataset_dir)
    splits_dir  = Path(splits_dir)
    results_dir = Path(results_dir)

    # Instantiate a temporary model just to get its name
    _tmp        = model_cls(seed=seeds[0], **model_kwargs)
    model_name  = _tmp.name
    del _tmp

    scores_dir, per_seed_dir = _make_dirs(results_dir, model_name)

    include_smiles = _needs_smiles(model_cls)

    # -- Load test set ---------------------------------------------------------
    # Setting 1: test is shared across seeds (loaded once)
    # Settings 2 & 3: test is seed-specific (loaded per seed in loop)
    if setting == 1:
        try:
            test_data = get_test(
                dataset_dir  = str(dataset_dir),
                target       = target,
                include_smiles = include_smiles,
            )
            if include_smiles:
                X_test, y_test, ids_test, smiles_test = test_data
            else:
                X_test, y_test, ids_test = test_data
                smiles_test = None

        except Exception as e:
            msg = f"TARGET SKIP [{target}] - failed to load test set: {e}\n{traceback.format_exc()}"
            logger.warning(msg)
            _log_error(results_dir, model_name, msg)
            return None
    elif setting in (2, 3):
        # Settings 2 & 3: test files are seed-specific, loaded in the seed loop
        X_test = y_test = ids_test = smiles_test = None
    else:
        raise ValueError(f"Unknown setting: {setting}. Expected 1, 2, or 3.")

    # -- Determine seeds to run -----------------------------------------------
    # Deterministic models (TanimotoNN) only need one seed
    # We check is_deterministic on a temporary instance
    _tmp2 = model_cls(seed=seeds[0], **model_kwargs)
    if _tmp2.is_deterministic:
        seeds_to_run = seeds[:1]
    else:
        seeds_to_run = seeds
    del _tmp2

    # -- Per-seed loop --------------------------------------------------------
    seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds_to_run:

        # -- Load train/val/test for this seed ------------------------------------
        try:
            if setting == 1:
                # Setting 1: load train/val from indices
                train_val_data = get_train_val(
                    dataset_dir    = str(dataset_dir),
                    splits_dir     = str(splits_dir),
                    target         = target,
                    seed           = seed,
                    include_smiles = include_smiles,
                )
                # Test already loaded above (shared across seeds)
                X_test_seed, y_test_seed, ids_test_seed, smiles_test_seed = X_test, y_test, ids_test, smiles_test

            elif setting == 2:
                # Setting 2: TopU few-shot. Train/val/test all from TopU only.
                train_val_data = get_train_val_setting2(
                    dataset_dir    = str(dataset_dir),
                    target         = target,
                    seed           = seed,
                    include_smiles = include_smiles,
                )

                # Load test (seed-specific)
                test_data_s2 = get_test_setting2(
                    dataset_dir    = str(dataset_dir),
                    target         = target,
                    seed           = seed,
                    include_smiles = include_smiles,
                )
                if include_smiles:
                    X_test_seed, y_test_seed, ids_test_seed, smiles_test_seed = test_data_s2
                else:
                    X_test_seed, y_test_seed, ids_test_seed = test_data_s2
                    smiles_test_seed = None

            elif setting == 3:
                # Setting 3: random ChEMBL* decoys
                train_val_data = get_train_val_setting3(
                    dataset_dir    = str(dataset_dir),
                    target         = target,
                    seed           = seed,
                    include_smiles = include_smiles,
                )

                # Load test (seed-specific)
                test_data_s3 = get_test_setting3(
                    dataset_dir    = str(dataset_dir),
                    target         = target,
                    seed           = seed,
                    include_smiles = include_smiles,
                )
                if include_smiles:
                    X_test_seed, y_test_seed, ids_test_seed, smiles_test_seed = test_data_s3
                else:
                    X_test_seed, y_test_seed, ids_test_seed = test_data_s3
                    smiles_test_seed = None

            else:
                raise ValueError(f"Unknown setting: {setting}. Expected 1, 2, or 3.")

        except Exception as e:
           raise RuntimeError(
              f"Failed to load data for {target} seed={seed} setting={setting}"
            ) from e

        # Unpack train/val. Settings 2 & 3 don't carry split_info.
        if include_smiles:
            if setting == 1:
                (X_train, y_train, ids_train, smiles_train,
                 X_val,   y_val,   ids_val,   smiles_val,
                 split_info) = train_val_data
            else:  # Settings 2 & 3
                (X_train, y_train, ids_train, smiles_train,
                 X_val,   y_val,   ids_val,   smiles_val) = train_val_data
                split_info = None
        else:
            if setting == 1:
                (X_train, y_train, ids_train,
                 X_val,   y_val,   ids_val,
                 split_info) = train_val_data
            else:  # Settings 2 & 3
                (X_train, y_train, ids_train,
                 X_val,   y_val,   ids_val) = train_val_data
                split_info = None
            smiles_train = smiles_val = None

        # -- Init wandb run ---------------------------------------------------
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                entity  = wandb_entity,
                project = wandb_project,
                group   = model_name,
                name    = f"{target}_seed{seed}",
                job_type= "train_eval",
                config  = {
                    "model":      model_name,
                    "target":     target,
                    "seed":       seed,
                    "setting":    setting,
                    "n_train":    int(y_train.sum()),
                    "n_val":      int(y_val.sum()),
                    "n_test":     int(y_test_seed.sum()),
                    **model_kwargs,
                },
                reinit  = "create_new",
            )

        # -- Train (crashes hard on failure) ----------------------------------
        # Only pass use_wandb if the model's __init__ accepts it

        _init_params = inspect.signature(model_cls.__init__).parameters
        _extra = {"use_wandb": use_wandb} if "use_wandb" in _init_params else {}
        model = model_cls(seed=seed, **_extra, **model_kwargs)
        # Check if model.fit accepts 'target' parameter
        _fit_params = inspect.signature(model.fit).parameters
        fit_kwargs = {
            'X_train':      X_train,
            'y_train':      y_train,
            'X_val':        X_val,
            'y_val':        y_val,
            'smiles_train': smiles_train,
            'smiles_val':   smiles_val,
        }

        # Add target if the model accepts it (for UniMol2 cache key)
        if 'target' in _fit_params:
            fit_kwargs['target'] = target

        model.fit(**fit_kwargs)

        # -- Predict (crashes hard on failure) --------------------------------
        # Use conformer ensemble for UniMol2 models
        if hasattr(model, 'predict_proba_ensemble'):
            logger.info("Using conformer ensemble for prediction...")
            scores = model.predict_proba_ensemble(
              X_test=X_test_seed,
              smiles_test=smiles_test_seed,
              n_conformers=5,  # Can change to 3, 7, 10, etc.
            )
        else:
            # Regular prediction for other models (GIN, RF, etc.)
            scores = model.predict_proba(
              X_test=X_test_seed,
              smiles_test=smiles_test_seed,
            )

        # -- Compute metrics --------------------------------------------------
        metrics = compute_all(y_test_seed, scores)

        # -- Save to disk -----------------------------------------------------
        _save_scores(scores_dir, target, seed, ids_test_seed, y_test_seed, scores)
        _save_per_seed_metrics(per_seed_dir, target, seed, metrics)

        # -- Log to wandb -----------------------------------------------------
        if use_wandb and wandb_run is not None:
            wandb_run.log({
                "test/ef_1pct":   metrics["ef_1pct"],
                "test/ef_5pct":   metrics["ef_5pct"],
                "test/ef_10pct":  metrics["ef_10pct"],
                "test/prauc":     metrics["prauc"],
                "test/rocauc":    metrics["rocauc"],
                "test/bedroc":    metrics["bedroc"],
                "test/bedroc_rdkit":  metrics["bedroc_rdkit"],
                "test/logauc":    metrics["logauc"],
                "test/n_actives": metrics["n_actives"],
                "test/n_decoys":  metrics["n_decoys"],
            })
            wandb_run.finish()

        seed_results[seed] = metrics

        logger.info(
            f"[{model_name}] {target} seed={seed} | "
            f"EF@1%={metrics['ef_1pct']:.3f} "
            f"EF@5%={metrics['ef_5pct']:.3f} "
            f"PR-AUC={metrics['prauc']:.3f}"
        )

    return seed_results
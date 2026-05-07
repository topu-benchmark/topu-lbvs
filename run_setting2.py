"""
run_setting2.py
TopU-LBVS Setting 2 - top-level CLI entry point with seed control.

Setting 2: TopU -> TopU (few-shot)
- Both training and test compounds drawn from the same TopU library.
- Tiered scaffold split on actives:
    * Tier 1 (TopU actives <  50): 6:2:2 train/val/test split of actives
    * Tier 2 (TopU actives >= 50): 7:1:2 train/val/test split of actives
- 1:40 active:inactive ratio in train, val, AND test
- Primary metric: EF@5%, Secondary: PR-AUC

Usage
-----
# Single target, all default seeds
python run_setting2.py --model gin --target egfr

# Single target, single seed
python run_setting2.py --model morgan_rf --target egfr --seeds 2026

# All 95 targets
python run_setting2.py --model morgan_rf

# Restrict to a tier
python run_setting2.py --model gin --tier 1
python run_setting2.py --model gin --tier 2

# Multiple specific targets
python run_setting2.py --model dmpnn --targets egfr akt2 cp2d6

# Disable wandb
python run_setting2.py --model gin --target egfr --no_wandb

Available models
----------------
    morgan_rf     MorganRF (fingerprint + random forest)
    tanimoto_nn   TanimotoNN (nearest-neighbour similarity search)
    gin           GIN (graph isomorphism network)
    dmpnn         D-MPNN (directed message passing)
    molformer     MolFormer (transformer)

"""

import argparse
import os
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_DEBUG'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import glob
import json
import logging
import sys
import time
from pathlib import Path

from data.loader import discover_targets
from training.runner import run_target


# -- Tier classification ------------------------------------------------------

TIER_CUTOFF = 50  # mirrors create_setting2_data.py


def _classify_tier(dataset_dir: str, target: str) -> int:
    """
    Read the TopU NPZ for a target and classify it into Tier 1 or Tier 2 based on
    the number of TopU actives. Returns 1, 2, or 0 (unknown / not found).
    """
    import numpy as np
    matches = list(Path(dataset_dir).glob(f"{target}/CHEMBL*_topUnbiased_ecfp4.npz"))
    if not matches:
        return 0
    try:
        topu = np.load(matches[0], allow_pickle=True)
        n_act = int((topu["y"] == 1).sum())
        return 1 if n_act < TIER_CUTOFF else 2
    except Exception:
        return 0


# -- Model registry -----------------------------------------------------------

def _build_registry():
    """Build model registry with all available models. Mirrors run_setting1/3."""
    from models.morgan_rf import MorganRF
    from models.tanimoto_nn import TanimotoNN

    registry = {
        "morgan_rf":   (MorganRF,   {}),
        "tanimoto_nn": (TanimotoNN, {}),
    }

    try:
        from models.gin import GIN, GINFP
        registry["gin"] = (GIN, {
          "n_layers": 5, "hidden_dim": 256, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
        })
        registry["ginfp"] = (GINFP, {
          "n_layers": 5, "hidden_dim": 256, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
          "fp_hidden_dim": 256, "fusion_hidden_dim": 128,
        })
    except ImportError:
        pass
  
    try:
        from models.gat import GAT, GATFP
        registry["gat"] = (GAT, {
          "n_layers": 5, "hidden_dim": 256, "heads": 4, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
        })
        registry["gatfp"] = (GATFP, {
          "n_layers": 5, "hidden_dim": 256, "heads": 4, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
          "fp_hidden_dim": 256, "fusion_hidden_dim": 128,
        })
    except ImportError:
        pass

    try:
       from models.gps import GPS, GPSFP
       registry["gps"] = (GPS, {
          "n_layers": 5, "hidden_dim": 256, "heads": 4, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
        })
       registry["gpsfp"] = (GPSFP, {
          "n_layers": 5, "hidden_dim": 256, "heads": 4, "dropout": 0.2,
          "max_epochs": 100, "patience": 20, "batch_size": 64,
          "fp_hidden_dim": 256, "fusion_hidden_dim": 128,
        })
    except ImportError:
        pass
        
    try:
        from models.dmpnn import DMPNN
        registry["dmpnn"] = (DMPNN, {
            "hidden_dim":     300,
            "depth":          3,
            "ffn_num_layers": 2,
            "dropout":        0.0,
            "max_epochs":     50,
            "patience":       25,
            "batch_size":     64,
        })
    except ImportError:
        pass

    try:
        from models.molformer import MolFormer
        registry["molformer"] = (MolFormer, {
            "lr": 3e-5,
            "max_epochs":     30,
            "patience":       5,
            "batch_size":     32,
            "use_amp":        False,
        })
    except ImportError:
        pass
        
    return registry


# -- Defaults ------------------------------------------------------------------

DEFAULT_DATASET_DIR = "./topu_dataset"
DEFAULT_SPLITS_DIR  = "./splits"
DEFAULT_RESULTS_DIR = "./results/setting2" 
DEFAULT_SEEDS       = [2026, 2027, 2028]

DEFAULT_WANDB_PROJECT = "LBVS"


# -- CLI argument parser -------------------------------------------------------

_ALL_MODEL_NAMES = [
    "morgan_rf", "tanimoto_nn",
    "gin", "ginfp", "gat", "gatfp", "gps", "gpsfp",
    "dmpnn", "molformer",]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="TopU-LBVS Setting 2 (TopU few-shot) - "
                    "run one model on one or all targets."
    )
    parser.add_argument("--model", required=True, choices=_ALL_MODEL_NAMES,
                        help="Model to run.")
    parser.add_argument("--target", default=None,
                        help="Single target name (e.g. egfr). Omit to run all.")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Multiple target names. Overrides --target.")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=None,
                        help="Restrict to a tier (1 or 2). Combine with no target "
                             "args to run that tier across all 95 targets.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                        help=f"Default: {DEFAULT_DATASET_DIR}")
    parser.add_argument("--splits_dir", default=DEFAULT_SPLITS_DIR,
                        help=f"Default: {DEFAULT_SPLITS_DIR} (unused by Setting 2 "
                             "but kept for runner signature compatibility)")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Random seeds. Default: {DEFAULT_SEEDS}")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging.")
    parser.add_argument("--wandb_entity", default=None,
                        help="WandB entity (team/user). If None, uses the user's default entity "
                             "from `wandb login`. Ignored when --no_wandb is set.")
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT,
                        help=f"WandB project name. Default: {DEFAULT_WANDB_PROJECT}")
                        
    parser.add_argument("--tuned_params_dir", default=None,
                        help="If set, load per-target best_params.json from "
                             "{tuned_params_dir}/{model}/{target}/best_params.json "
                             "and merge into model_kwargs.")
    return parser.parse_args()
    
def _load_tuned_params(tuned_dir, model_name, target):
    """Load best_params.json for (model, target). Returns dict or {}."""
    import json, os
    if tuned_dir is None:
        return {}
    path = os.path.join(tuned_dir, model_name, target, "best_params.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            blob = json.load(f)
        # best_params.json schema: {"best_params": {...}, "best_value": ..., "seed": ..., ...}
        # Fall back to alternate keys, then to the blob itself.
        if isinstance(blob, dict):
            params = blob.get("best_params") or blob.get("params") or blob
        else:
            params = {}
        # Drop any non-hparam metadata that may have leaked in
        for k in ("score", "val_score", "best_value", "n_trials_total",
                  "trial_id", "model", "target", "setting", "seed"):
            params.pop(k, None)
        return params
        
    except Exception as e:
        print(f"[warn] failed to load tuned params from {path}: {e}", flush=True)
        return {}

# -- Main ----------------------------------------------------------------------

def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    MODEL_REGISTRY = _build_registry()
    if args.model not in MODEL_REGISTRY:
        logger.error(f"Unknown model: '{args.model}'")
        logger.error(f"Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_cls, model_kwargs = MODEL_REGISTRY[args.model]

    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    if not seeds:
        logger.error("No seeds provided!")
        sys.exit(1)

    # Get model name for logging
    _tmp = model_cls(seed=seeds[0], **model_kwargs)
    model_name = _tmp.name
    del _tmp

    # Get targets
    if args.targets is not None:
        targets = args.targets
    elif args.target is not None:
        targets = [args.target]
    else:
        targets = discover_targets(args.dataset_dir)

    # Optional tier filter
    if args.tier is not None:
        before = len(targets)
        targets = [t for t in targets if _classify_tier(args.dataset_dir, t) == args.tier]
        logger.info(f"Tier filter: {args.tier} -> kept {len(targets)}/{before} targets")

    if not targets:
        logger.error("No targets to run after filtering!")
        sys.exit(1)

    # Log run configuration
    logger.info("=" * 80)
    logger.info("TopU-LBVS Setting 2 (TopU Few-Shot)")
    logger.info("=" * 80)
    logger.info(f"Model      : {model_name} ({args.model})")
    logger.info(f"Targets    : {len(targets)} targets")
    if len(targets) <= 12:
        logger.info(f"             {targets}")
    logger.info(f"Seeds      : {seeds}")
    logger.info(f"Dataset    : {args.dataset_dir}")
    logger.info(f"Results    : {args.results_dir}")
    logger.info(f"Wandb      : {'disabled' if args.no_wandb else 'enabled'}")
    logger.info("=" * 80)

    # Verify Setting 2 data files exist
    logger.info("\nVerifying Setting 2 data files...")
    missing = []
    for target in targets:
        for seed in seeds:
            for split in ["train", "val", "test"]:
                pat = f"{args.dataset_dir}/{target}/CHEMBL*_{split}_s2_seed{seed}_ecfp4.npz"
                if not glob.glob(pat):
                    if target not in missing:
                        missing.append(target)
                    logger.warning(f"  Missing: {target} seed {seed} {split}")
    if missing:
        logger.error(f"\nERROR: Missing Setting 2 data files for: {missing}")
        logger.error("Run create_setting2_data.py first:")
        logger.error(f"  python create_setting2_data.py --targets {' '.join(missing)}")
        sys.exit(1)
    logger.info("All Setting 2 data files found.\n")

    # Run model on each target
    start_time = time.time()
    n_success = 0
    n_failed  = 0

    for i, target in enumerate(targets, 1):
        tier = _classify_tier(args.dataset_dir, target)
        logger.info(f"\n[{i}/{len(targets)}] Running {model_name} on {target} (Tier {tier})...")
        
        target_kwargs = dict(model_kwargs)
        tuned = _load_tuned_params(args.tuned_params_dir, args.model, target)
        if tuned:
            logger.info(f"  Loaded tuned params for {args.model}/{target}: {tuned}")
            target_kwargs.update(tuned)

        try:
            result = run_target(
                model_cls     = model_cls,
                model_kwargs  = target_kwargs,
                target        = target,
                dataset_dir   = args.dataset_dir,
                splits_dir    = args.splits_dir,
                results_dir   = args.results_dir,
                seeds         = seeds,
                use_wandb     = not args.no_wandb,
                wandb_entity  = args.wandb_entity,
                wandb_project = args.wandb_project,
                setting       = 2,  # KEY: Setting 2
            )

            if result is not None:
                n_success += 1
                # Setting 2 primary metric is EF@5%
                mean_ef5 = sum(r["ef_5pct"] for r in result.values()) / len(result)
                logger.info(f"{target} complete (Tier {tier}). Mean EF@5%={mean_ef5:.2f}")
            else:
                n_failed += 1
                logger.warning(f"{target} skipped (see logs).")

        except Exception as e:
            n_failed += 1
            logger.error(f"{target} FAILED: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Setting    : 2 (TopU Few-Shot)")
    logger.info(f"Model      : {model_name}")
    logger.info(f"Targets    : {n_success} success, {n_failed} failed")
    logger.info(f"Time       : {elapsed/60:.1f} min")
    logger.info(f"Seeds used : {seeds}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

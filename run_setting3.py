"""
run_setting3.py
TopU-LBVS Setting 3 - top-level CLI entry point with seed control.

Setting 3: Random ChEMBL* decoys instead of TopU hard decoys.
Same training/validation scaffold splits as Setting 1.

Usage
-----
# Single target, all seeds (default: 2026, 2027, 2028)
python run_setting3.py --model gin --target egfr

# Single target, SINGLE SEED for testing
python run_setting3.py --model morgan_rf --target egfr --seeds 2026

# Single target, CUSTOM SEEDS
python run_setting3.py --model gin --target egfr --seeds 2026 2027

# All targets (7 representative targets)
python run_setting3.py --model morgan_rf

# Specific targets
python run_setting3.py --model dmpnn --targets cp2d6 egfr

# Disable wandb
python run_setting3.py --model gin --target egfr --no_wandb

Available models
----------------
    morgan_rf     MorganRF (fingerprint + random forest)
    tanimoto_nn   TanimotoNN (nearest-neighbour similarity search)
    gin           GIN (graph isomorphism network)
    dmpnn         D-MPNN (directed message passing)
    molformer     MolFormer (transformer)

Setting 3 Targets (7 representative targets from paper)
--------------------------------------------------------
    cp2d6    Enzyme
    aa2ar    GPCR
    kcnh2    Ion channel
    egfr     Kinase
    esr1     Nuclear receptor
    aces     Enzyme
    thrb     Nuclear receptor
"""

import argparse
import os
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_DEBUG'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import logging
import sys
import time
from pathlib import Path
import glob

from data.loader import discover_targets
from training.runner import run_target

# -- Model registry -----------------------------------------------------------

def _build_registry():
    """Build model registry with all available models."""
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
           "hidden_dim":      300,
           "depth":           3,
           "ffn_num_layers":  2,
           "dropout":         0.0,
           "max_epochs":      50,
           "patience":        25,
           "batch_size":      64,
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
DEFAULT_RESULTS_DIR = "./results/setting3"  
DEFAULT_SEEDS       = [2026, 2027, 2028]

# Setting 3 targets (7 representative targets from paper)
SETTING3_TARGETS = ["cp2d6", "aa2ar", "kcnh2", "egfr", "esr1", "aces", "thrb"]

DEFAULT_WANDB_PROJECT = "LBVS"


# -- CLI argument parser -------------------------------------------------------

# Known model names for argparse choices
_ALL_MODEL_NAMES = [
    "morgan_rf", "tanimoto_nn",
    "gin", "ginfp", "gat", "gatfp", "gps", "gpsfp",
    "dmpnn", "molformer",
    "unimol2_84m", "unimol2_164m", "unimol2_310m", "unimol2_1_1b",
]

def _parse_args():
    parser = argparse.ArgumentParser(
        description="TopU-LBVS Setting 3 - run one model on one or all targets with random ChEMBL* decoys."
    )
    parser.add_argument(
        "--model", required=True,
        choices=_ALL_MODEL_NAMES,
        help="Model to run.",
    )
    parser.add_argument(
        "--target", default=None,
        help="Single target name (e.g. egfr). Omit to run all Setting 3 targets.",
    )
    parser.add_argument(
        "--targets", nargs='+', default=None,
        help="Multiple target names (e.g. --targets cp2d6 egfr). Overrides --target.",
    )
    parser.add_argument(
        "--dataset_dir", default=DEFAULT_DATASET_DIR,
        help=f"Path to topu_dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--splits_dir", default=DEFAULT_SPLITS_DIR,
        help=f"Path to splits directory. Default: {DEFAULT_SPLITS_DIR}",
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR,
        help=f"Path to results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--seeds", nargs='+', type=int, default=None,
        help="Random seeds to run (e.g., --seeds 2026 or --seeds 2026 2027 2028). "
             f"Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--wandb_entity", default=None,
        help="WandB entity (team/user). If None, uses the user's default entity "
             "from `wandb login`. Ignored when --no_wandb is set.",
    )
    parser.add_argument(
        "--wandb_project", default=DEFAULT_WANDB_PROJECT,
        help=f"WandB project name. Default: {DEFAULT_WANDB_PROJECT}",
    )
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
        if isinstance(blob, dict):
            params = blob.get("best_params") or blob.get("params") or blob
        else:
            params = {}
        # Drop non-hparam metadata that may have leaked in
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

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Get model
    MODEL_REGISTRY = _build_registry()
    if args.model not in MODEL_REGISTRY:
        logger.error(f"Unknown model: '{args.model}'")
        logger.error(f"Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_cls, model_kwargs = MODEL_REGISTRY[args.model]
    
    # Use custom seeds if provided, otherwise default
    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    
    # Validate seeds
    if not seeds:
        logger.error("No seeds provided!")
        sys.exit(1)
    
    # Create temporary model to get name (for logging)
    _tmp       = model_cls(seed=seeds[0], **model_kwargs)
    model_name = _tmp.name
    del _tmp

    # Get targets
    if args.targets is not None:
        # Multiple targets specified via --targets
        targets = args.targets
    elif args.target is not None:
        # Single target specified via --target
        targets = [args.target]
    else:
        # No target specified - use Setting 3 default targets
        targets = SETTING3_TARGETS

    # Log run configuration
    logger.info("=" * 80)
    logger.info("TopU-LBVS Setting 3 (Random ChEMBL* Decoys)")
    logger.info("=" * 80)
    logger.info(f"Model      : {model_name} ({args.model})")
    logger.info(f"Targets    : {len(targets)} targets - {targets}")
    logger.info(f"Seeds      : {seeds}")
    logger.info(f"Dataset    : {args.dataset_dir}")
    logger.info(f"Splits     : {args.splits_dir}")
    logger.info(f"Results    : {args.results_dir}")
    logger.info(f"Wandb      : {'disabled' if args.no_wandb else 'enabled'}")
    logger.info("=" * 80)

    # Verify Setting 3 data files exist for each target
    logger.info("\nVerifying Setting 3 data files...")
    missing_targets = []
    for target in targets:
        for seed in seeds:
            for split in ['train', 'val', 'test']:
                pattern = f"{args.dataset_dir}/{target}/CHEMBL*_{split}_s3_seed{seed}_ecfp4.npz"
                if not glob.glob(pattern):
                    logger.warning(f"  Missing: {target} seed {seed} {split}")
                    if target not in missing_targets:
                        missing_targets.append(target)
    
    if missing_targets:
        logger.error(f"\nERROR: Missing Setting 3 data files for targets: {missing_targets}")
        logger.error("Run create_setting3_data.py first to generate these files:")
        logger.error(f"  python create_setting3_data.py --targets {' '.join(missing_targets)}")
        sys.exit(1)
    
    logger.info("✓ All Setting 3 data files found!\n")

    # Run model on each target
    start_time = time.time()
    n_success  = 0
    n_failed   = 0

    for i, target in enumerate(targets, 1):
        logger.info(f"\n[{i}/{len(targets)}] Running {model_name} on {target}...")
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
                setting       = 3,  # ← KEY CHANGE: Use Setting 3
            )

            if result is not None:
                n_success += 1
                # Log mean EF@1% across seeds for quick progress check
                mean_ef1 = sum(r["ef_1pct"] for r in result.values()) / len(result)
                logger.info(f"✓ {target} complete. Mean EF@1%={mean_ef1:.1f}")
            else:
                n_failed += 1
                logger.warning(f"✗ {target} skipped (see logs).")

        except Exception as e:
            n_failed += 1
            logger.error(f"✗ {target} FAILED with exception: {e}", exc_info=True)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Setting    : 3 (Random ChEMBL* Decoys)")
    logger.info(f"Model      : {model_name}")
    logger.info(f"Targets    : {n_success} success, {n_failed} failed")
    logger.info(f"Time       : {elapsed/60:.1f} min")
    logger.info(f"Seeds used : {seeds}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

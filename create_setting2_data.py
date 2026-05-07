"""
create_setting2_data.py

Create Setting 2 data files for TopU-LBVS benchmark.

Setting 2 Design (paper §4.2 + Appendix E.3, Table 9):
- TopU -> TopU (few-shot): all splits drawn from the TopU library only.
- Tiered scaffold split on actives:
    * Tier 1 (TopU actives <  50): 6:2:2 train/val/test split of actives
    * Tier 2 (TopU actives >= 50): 7:1:2 train/val/test split of actives
- 1:40 active:inactive ratio in train, val, AND test
- Bemis-Murcko scaffold-based split for actives, seed-deterministic
- Inactives are shuffled per seed and sliced to preserve the 1:40 ratio

Files created per target per seed:
    CHEMBL*_train_s2_seed{seed}_ecfp4.npz
    CHEMBL*_val_s2_seed{seed}_ecfp4.npz
    CHEMBL*_test_s2_seed{seed}_ecfp4.npz

Plus a manifest:
    {dataset_dir}/setting2_manifest.json   (tier, sizes, paths per target/seed)

Usage:
    # Create for all targets, all default seeds
    python create_setting2_data.py

    # Specific targets (e.g. one per tier)
    python create_setting2_data.py --targets egfr akt2

    # Specific seeds
    python create_setting2_data.py --seeds 2026

    # Dry run (no files written)
    python create_setting2_data.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATASET_DIR = "./topu_dataset"
DEFAULT_SEEDS = [2026, 2027, 2028]

# Tier boundaries (paper Table 9)
TIER_CUTOFF = 50
TIER1_FRACS = (0.6, 0.2, 0.2)   # 6:2:2 train:val:test
TIER2_FRACS = (0.7, 0.1, 0.2)   # 7:1:2 train:val:test

# Active:inactive ratio in every split
RATIO = 40


# ============================================================================
# Helpers
# ============================================================================

def discover_targets(dataset_dir: str) -> List[str]:
    """Find all targets that have a CHEMBL*_topUnbiased_ecfp4.npz file."""
    targets = []
    for name in sorted(os.listdir(dataset_dir)):
        target_dir = Path(dataset_dir) / name
        if not target_dir.is_dir():
            continue
        if list(target_dir.glob("CHEMBL*_topUnbiased_ecfp4.npz")):
            targets.append(name)
    return targets


def find_npz(dataset_dir: str, target: str, suffix: str) -> Path:
    """Find single NPZ file matching CHEMBL*{suffix} in {dataset_dir}/{target}/."""
    matches = list(Path(dataset_dir).glob(f"{target}/CHEMBL*{suffix}"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matching CHEMBL*{suffix} in {dataset_dir}/{target}/"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files matching CHEMBL*{suffix} in {dataset_dir}/{target}/: {matches}"
        )
    return matches[0]


def murcko_scaffold(smi: str) -> str:
    """Bemis-Murcko scaffold SMILES; falls back to '' on parse error."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split_indices(
    smiles: np.ndarray,
    fracs: Tuple[float, float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bemis-Murcko scaffold split. Greedy largest-bin-first assignment:
      1. Group indices by scaffold SMILES.
      2. Sort scaffold groups by size (desc); break ties deterministically using
         a seeded permutation among same-size groups.
      3. Assign each group greedily to whichever split is most below its target,
         in priority order train -> val -> test.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int
    """
    n = len(smiles)
    frac_train, frac_val, frac_test = fracs
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, fracs

    n_train_t = int(round(frac_train * n))
    n_val_t   = int(round(frac_val   * n))
    n_test_t  = n - n_train_t - n_val_t  # remainder

    # Group indices by scaffold
    scaffold_to_idx: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        s = murcko_scaffold(str(smi))
        scaffold_to_idx.setdefault(s, []).append(i)

    # Deterministic ordering: shuffle then stable-sort by size desc
    rng = np.random.default_rng(seed)
    groups = list(scaffold_to_idx.values())
    perm = rng.permutation(len(groups))
    groups = [groups[i] for i in perm]
    groups.sort(key=lambda g: len(g), reverse=True)

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for g in groups:
        # Pick split most below target (in absolute deficit)
        deficits = [
            (n_train_t - len(train_idx), 0, train_idx),
            (n_val_t   - len(val_idx),   1, val_idx),
            (n_test_t  - len(test_idx),  2, test_idx),
        ]
        # Prefer non-overflow splits; among those, the largest deficit
        deficits.sort(key=lambda x: (x[0] >= len(g), x[0]), reverse=True)
        deficits[0][2].extend(g)

    # Edge case: an empty split (small targets, big single scaffold)
    # In that case borrow one molecule from the largest split that still has
    # at least 2 molecules, so model code doesn't choke on zero-size splits.
    splits = [train_idx, val_idx, test_idx]
    names  = ["train", "val", "test"]
    for j, s in enumerate(splits):
        if len(s) == 0:
            # find largest split with >=2 elements
            donor = max(range(3), key=lambda k: len(splits[k]) if len(splits[k]) >= 2 else -1)
            if len(splits[donor]) >= 2:
                splits[j].append(splits[donor].pop())
                logger.warning(f"  scaffold split: empty {names[j]} - borrowed 1 from {names[donor]}")

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx,   dtype=np.int64),
        np.array(test_idx,  dtype=np.int64),
    )


# ============================================================================
# Main per-target/seed routine
# ============================================================================

def create_setting2_files_for_target(
    dataset_dir: str,
    target: str,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Create Setting 2 train/val/test NPZ files for one target and one seed.

    Returns
    -------
    Dict with statistics and output paths.
    """
    logger.info(f"[{target}] Creating Setting 2 files (seed={seed})...")

    # ------------------------------------------------------------------------
    # Step 1: Load TopU NPZ (only source for Setting 2)
    # ------------------------------------------------------------------------
    topu_path = find_npz(dataset_dir, target, "_topUnbiased_ecfp4.npz")
    topu = np.load(topu_path, allow_pickle=True)

    X      = topu["X"]
    y      = topu["y"]
    smiles = topu["smiles"]
    ids    = topu["ids"]

    active_mask   = (y == 1)
    inactive_mask = (y == 0)
    n_actives     = int(active_mask.sum())
    n_inactives   = int(inactive_mask.sum())
    logger.info(f"[{target}] TopU library: {n_actives} actives, {n_inactives} inactives")

    # ------------------------------------------------------------------------
    # Step 2: Determine tier and split fractions
    # ------------------------------------------------------------------------
    if n_actives < TIER_CUTOFF:
        tier = 1
        fracs = TIER1_FRACS
    else:
        tier = 2
        fracs = TIER2_FRACS
    logger.info(
        f"[{target}] Tier {tier} (actives {'<' if tier == 1 else '>='} {TIER_CUTOFF}), "
        f"split {':'.join(str(int(f * 10)) for f in fracs)}"
    )

    # ------------------------------------------------------------------------
    # Step 3: Scaffold split actives
    # ------------------------------------------------------------------------
    active_indices = np.where(active_mask)[0]
    active_smiles  = smiles[active_indices]

    tr_local, va_local, te_local = scaffold_split_indices(
        smiles=active_smiles,
        fracs=fracs,
        seed=seed,
    )

    train_active_idx = active_indices[tr_local]
    val_active_idx   = active_indices[va_local]
    test_active_idx  = active_indices[te_local]

    n_train_actives = len(train_active_idx)
    n_val_actives   = len(val_active_idx)
    n_test_actives  = len(test_active_idx)
    logger.info(
        f"[{target}] Active scaffold split -> "
        f"train={n_train_actives}, val={n_val_actives}, test={n_test_actives}"
    )

    # ------------------------------------------------------------------------
    # Step 4: Compute inactive counts at 1:40 ratio
    # ------------------------------------------------------------------------
    n_train_inact = n_train_actives * RATIO
    n_val_inact   = n_val_actives   * RATIO
    n_test_inact  = n_test_actives  * RATIO
    n_total_inact_needed = n_train_inact + n_val_inact + n_test_inact

    if n_total_inact_needed > n_inactives:
        raise ValueError(
            f"[{target}] Not enough TopU inactives. "
            f"Need {n_total_inact_needed}, have {n_inactives}. "
            f"(train={n_train_inact}, val={n_val_inact}, test={n_test_inact})"
        )
    logger.info(
        f"[{target}] Inactive counts (1:{RATIO}) -> "
        f"train={n_train_inact}, val={n_val_inact}, test={n_test_inact} "
        f"({n_total_inact_needed}/{n_inactives} used)"
    )

    # ------------------------------------------------------------------------
    # Step 5: Shuffle inactives with seed and slice
    # ------------------------------------------------------------------------
    inactive_indices = np.where(inactive_mask)[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(inactive_indices))
    inactive_indices = inactive_indices[perm]

    train_inact_idx = inactive_indices[:n_train_inact]
    val_inact_idx   = inactive_indices[n_train_inact: n_train_inact + n_val_inact]
    test_inact_idx  = inactive_indices[
        n_train_inact + n_val_inact: n_train_inact + n_val_inact + n_test_inact
    ]

    # ------------------------------------------------------------------------
    # Step 6: Assemble and save files
    # ------------------------------------------------------------------------
    def _build(idx_act: np.ndarray, idx_inact: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.concatenate([idx_act, idx_inact])
        return {
            "X":      X[idx],
            "y":      y[idx].astype(np.int32),
            "smiles": smiles[idx],
            "ids":    ids[idx],
        }

    train_data = _build(train_active_idx, train_inact_idx)
    val_data   = _build(val_active_idx,   val_inact_idx)
    test_data  = _build(test_active_idx,  test_inact_idx)

    train_out = topu_path.parent / topu_path.name.replace(
        "_topUnbiased_ecfp4.npz", f"_train_s2_seed{seed}_ecfp4.npz"
    )
    val_out = topu_path.parent / topu_path.name.replace(
        "_topUnbiased_ecfp4.npz", f"_val_s2_seed{seed}_ecfp4.npz"
    )
    test_out = topu_path.parent / topu_path.name.replace(
        "_topUnbiased_ecfp4.npz", f"_test_s2_seed{seed}_ecfp4.npz"
    )

    if not dry_run:
        for path, data in [(train_out, train_data), (val_out, val_data), (test_out, test_data)]:
            np.savez_compressed(
                path,
                X=data["X"],
                y=data["y"],
                smiles=data["smiles"],
                ids=data["ids"],
            )
            logger.info(f"[{target}]   saved {path.name} ({len(data['y'])} compounds)")
    else:
        for path, data in [(train_out, train_data), (val_out, val_data), (test_out, test_data)]:
            logger.info(f"[{target}]   [DRY RUN] would save {path.name} ({len(data['y'])} compounds)")

    return {
        "target": target,
        "seed": seed,
        "tier": tier,
        "split_fracs": list(fracs),
        "n_topu_actives_total": n_actives,
        "n_topu_inactives_total": n_inactives,
        "n_train": int(len(train_data["y"])),
        "n_train_actives": n_train_actives,
        "n_train_inactives": n_train_inact,
        "n_val": int(len(val_data["y"])),
        "n_val_actives": n_val_actives,
        "n_val_inactives": n_val_inact,
        "n_test": int(len(test_data["y"])),
        "n_test_actives": n_test_actives,
        "n_test_inactives": n_test_inact,
        "train_path": str(train_out),
        "val_path": str(val_out),
        "test_path": str(test_out),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Create Setting 2 (TopU few-shot) data files."
    )
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                   help=f"Path to topu_dataset (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Specific targets (default: all discovered)")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help=f"Seeds (default: {DEFAULT_SEEDS})")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write any files; print what would happen.")
    p.add_argument("--manifest", default=None,
                   help="Manifest JSON path (default: {dataset_dir}/setting2_manifest.json)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Creating Setting 2 Data Files (TopU few-shot)")
    logger.info("=" * 80)
    logger.info(f"Dataset dir : {args.dataset_dir}")
    logger.info(f"Seeds       : {args.seeds}")
    logger.info(f"Tier cutoff : {TIER_CUTOFF} actives")
    logger.info(f"Tier 1 split: {TIER1_FRACS} (6:2:2)")
    logger.info(f"Tier 2 split: {TIER2_FRACS} (7:1:2)")
    logger.info(f"Ratio       : 1:{RATIO} (active:inactive in all splits)")
    logger.info(f"Dry run     : {args.dry_run}")
    logger.info("=" * 80)

    if args.targets:
        targets = args.targets
        logger.info(f"Processing {len(targets)} specified targets")
    else:
        targets = discover_targets(args.dataset_dir)
        logger.info(f"Discovered {len(targets)} targets")

    all_stats: List[dict] = []
    n_failed = 0

    for i, target in enumerate(targets, 1):
        logger.info(f"\n[{i}/{len(targets)}] Processing {target}...")
        try:
            for seed in args.seeds:
                stats = create_setting2_files_for_target(
                    dataset_dir=args.dataset_dir,
                    target=target,
                    seed=seed,
                    dry_run=args.dry_run,
                )
                all_stats.append(stats)
            logger.info(f"[{target}] complete")
        except Exception as e:
            n_failed += 1
            logger.error(f"[{target}] FAILED: {e}", exc_info=True)

    # Manifest
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(args.dataset_dir) / "setting2_manifest.json"
    )
    if not args.dry_run and all_stats:
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "tier_cutoff": TIER_CUTOFF,
                    "tier1_fracs": list(TIER1_FRACS),
                    "tier2_fracs": list(TIER2_FRACS),
                    "ratio": RATIO,
                    "seeds": args.seeds,
                    "entries": all_stats,
                },
                f,
                indent=2,
            )
        logger.info(f"\nManifest written: {manifest_path}")

    # Summary by tier
    tier1 = [s for s in all_stats if s["tier"] == 1]
    tier2 = [s for s in all_stats if s["tier"] == 2]

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Targets attempted  : {len(targets)}")
    logger.info(f"Failures           : {n_failed}")
    logger.info(f"Seeds              : {args.seeds}")
    logger.info(f"Files per target   : {len(args.seeds) * 3}")
    logger.info(f"Tier 1 entries     : {len(tier1)} (across all seeds)")
    logger.info(f"Tier 2 entries     : {len(tier2)} (across all seeds)")
    if args.dry_run:
        logger.info("\nDRY RUN - no files were written")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

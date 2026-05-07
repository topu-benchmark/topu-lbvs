"""
create_setting3_data.py

Create Setting 3 data files for TopU-LBVS benchmark.

Setting 3 Design:
- Creates 3 SEPARATE files per seed: train, val, test
- Actives: Taken from Setting 1 split indices (same scaffold splits)
- Inactives: Shuffled from combined pool and redistributed

Process for each seed:
1. Load train/val split indices from Setting 1
2. Extract actives for train, val, test
3. Pool ALL inactives (final + topu) = 11,600 total
4. Shuffle inactives with seed
5. Distribute: 8,220 train + 1,460 val + 1,920 test
6. Create 3 files:
   - CHEMBL*_train_s3_seed{seed}_ecfp4.npz
   - CHEMBL*_val_s3_seed{seed}_ecfp4.npz
   - CHEMBL*_test_s3_seed{seed}_ecfp4.npz

Usage:
    # Create for all targets
    python create_setting3_data.py
    
    # Create for specific targets
    python create_setting3_data.py --targets akt2 xiap
    
    # Use specific seeds
    python create_setting3_data.py --seeds 2026 2027
    
    # Dry run (check without creating)
    python create_setting3_data.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATASET_DIR = "./topu_dataset"
DEFAULT_SPLITS_DIR = "./splits"
DEFAULT_SEEDS = [2026, 2027, 2028]


# ============================================================================
# Helper Functions
# ============================================================================

def discover_targets(dataset_dir: str) -> List[str]:
    """Find all targets with _final_ecfp4.npz files."""
    targets = []
    for name in sorted(os.listdir(dataset_dir)):
        target_dir = Path(dataset_dir) / name
        if not target_dir.is_dir():
            continue
        
        # Check if has final npz
        final_files = list(target_dir.glob("CHEMBL*_final_ecfp4.npz"))
        if final_files:
            targets.append(name)
    
    return targets


def find_npz(dataset_dir: str, target: str, suffix: str) -> Path:
    """Find NPZ file matching pattern."""
    matches = list(Path(dataset_dir).glob(f"{target}/CHEMBL*{suffix}"))
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching CHEMBL*{suffix} in {dataset_dir}/{target}/")
    if len(matches) > 1:
        raise ValueError(f"Multiple files matching CHEMBL*{suffix} in {dataset_dir}/{target}/")
    
    return matches[0]


def load_split_indices(splits_dir: str, target: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load train and val indices from Setting 1 splits."""
    split_dir = Path(splits_dir) / f"seed_{seed}" / target
    
    train_idx_path = split_dir / "train_idx.npy"
    val_idx_path = split_dir / "val_idx.npy"
    
    if not train_idx_path.exists():
        raise FileNotFoundError(f"Train indices not found: {train_idx_path}")
    if not val_idx_path.exists():
        raise FileNotFoundError(f"Val indices not found: {val_idx_path}")
    
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    
    return train_idx, val_idx


# ============================================================================
# Main Function
# ============================================================================

def create_setting3_files_for_target(
    dataset_dir: str,
    splits_dir: str,
    target: str,
    seed: int,
    dry_run: bool = False
) -> dict:
    """
    Create Setting 3 files (train, val, test) for one target and one seed.
    
    Creates:
    - CHEMBL*_train_s3_seed{seed}_ecfp4.npz
    - CHEMBL*_val_s3_seed{seed}_ecfp4.npz
    - CHEMBL*_test_s3_seed{seed}_ecfp4.npz
    
    Returns:
    - Dictionary with statistics
    """
    logger.info(f"[{target}] Creating Setting 3 files (seed={seed})...")
    
    # ========================================================================
    # Step 1: Load Setting 1 data
    # ========================================================================
    
    logger.info(f"[{target}] Loading Setting 1 data...")
    final_path = find_npz(dataset_dir, target, "_final_ecfp4.npz")
    topu_path = find_npz(dataset_dir, target, "_topUnbiased_ecfp4.npz")
    
    final = np.load(final_path, allow_pickle=True)
    topu = np.load(topu_path, allow_pickle=True)
    
    # ========================================================================
    # Step 2: Load split indices
    # ========================================================================
    
    logger.info(f"[{target}] Loading split indices for seed {seed}...")
    train_idx, val_idx = load_split_indices(splits_dir, target, seed)
    
    logger.info(f"[{target}] Train indices: {len(train_idx)}, Val indices: {len(val_idx)}")
    
    # ========================================================================
    # Step 3: Extract actives using split indices
    # ========================================================================
    
    logger.info(f"[{target}] Extracting actives from splits...")
    
    # Train actives (from final.npz using train_idx)
    train_actives = {
        'X': final['X'][train_idx],
        'y': final['y'][train_idx],
        'smiles': final['smiles'][train_idx],
        'ids': final['ids'][train_idx],
    }
    train_active_mask = train_actives['y'] == 1
    n_train_actives = train_active_mask.sum()
    
    # Val actives (from final.npz using val_idx)
    val_actives = {
        'X': final['X'][val_idx],
        'y': final['y'][val_idx],
        'smiles': final['smiles'][val_idx],
        'ids': final['ids'][val_idx],
    }
    val_active_mask = val_actives['y'] == 1
    n_val_actives = val_active_mask.sum()
    
    # Test actives (from topu.npz)
    topu_active_mask = topu['y'] == 1
    test_actives = {
        'X': topu['X'][topu_active_mask],
        'y': topu['y'][topu_active_mask],
        'smiles': topu['smiles'][topu_active_mask],
        'ids': topu['ids'][topu_active_mask],
    }
    n_test_actives = len(test_actives['y'])
    
    logger.info(f"[{target}] Train actives: {n_train_actives}")
    logger.info(f"[{target}] Val actives: {n_val_actives}")
    logger.info(f"[{target}] Test actives: {n_test_actives}")
    
    # ========================================================================
    # Step 4: Pool ALL inactives
    # ========================================================================
    
    logger.info(f"[{target}] Pooling all inactives...")
    
    # Inactives from final.npz (all of them, not split by train/val)
    final_inactive_mask = final['y'] == 0
    final_inactives = {
        'X': final['X'][final_inactive_mask],
        'smiles': final['smiles'][final_inactive_mask],
        'ids': final['ids'][final_inactive_mask],
    }
    n_final_inactives = len(final_inactives['X'])
    
    # Inactives from topu.npz
    topu_inactive_mask = topu['y'] == 0
    topu_inactives = {
        'X': topu['X'][topu_inactive_mask],
        'smiles': topu['smiles'][topu_inactive_mask],
        'ids': topu['ids'][topu_inactive_mask],
    }
    n_topu_inactives = len(topu_inactives['X'])
    
    # Pool all inactives together
    all_inactives = {
        'X': np.vstack([final_inactives['X'], topu_inactives['X']]),
        'smiles': np.concatenate([final_inactives['smiles'], topu_inactives['smiles']]),
        'ids': np.concatenate([final_inactives['ids'], topu_inactives['ids']]),
    }
    n_total_inactives = len(all_inactives['X'])
    
    logger.info(f"[{target}] Final inactives: {n_final_inactives}")
    logger.info(f"[{target}] TopU inactives: {n_topu_inactives}")
    logger.info(f"[{target}] Total inactive pool: {n_total_inactives}")
    
    # ========================================================================
    # Step 5: Shuffle inactives with seed
    # ========================================================================
    
    logger.info(f"[{target}] Shuffling inactive pool with seed={seed}...")
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(n_total_inactives)
    
    all_inactives['X'] = all_inactives['X'][shuffle_idx]
    all_inactives['smiles'] = all_inactives['smiles'][shuffle_idx]
    all_inactives['ids'] = all_inactives['ids'][shuffle_idx]
    
    # ========================================================================
    # Step 6: Calculate inactive counts for each split (1:10 and 1:40 ratios)
    # ========================================================================
    
    n_train_inactives = n_train_actives * 10  # 1:10 ratio
    n_val_inactives = n_val_actives * 10      # 1:10 ratio
    n_test_inactives = n_test_actives * 40    # 1:40 ratio
    
    logger.info(f"[{target}] Train inactives needed: {n_train_inactives} (ratio 1:10)")
    logger.info(f"[{target}] Val inactives needed: {n_val_inactives} (ratio 1:10)")
    logger.info(f"[{target}] Test inactives needed: {n_test_inactives} (ratio 1:40)")
    logger.info(f"[{target}] Total inactives needed: {n_train_inactives + n_val_inactives + n_test_inactives}")
    
    # Verify we have enough
    if n_train_inactives + n_val_inactives + n_test_inactives > n_total_inactives:
        raise ValueError(f"Not enough inactives! Need {n_train_inactives + n_val_inactives + n_test_inactives}, have {n_total_inactives}")
    
    # ========================================================================
    # Step 7: Distribute shuffled inactives
    # ========================================================================
    
    logger.info(f"[{target}] Distributing shuffled inactives...")
    
    # Train inactives (first n_train_inactives)
    train_inactives = {
        'X': all_inactives['X'][:n_train_inactives],
        'smiles': all_inactives['smiles'][:n_train_inactives],
        'ids': all_inactives['ids'][:n_train_inactives],
        'y': np.zeros(n_train_inactives, dtype=np.int32),
    }
    
    # Val inactives (next n_val_inactives)
    val_inactives = {
        'X': all_inactives['X'][n_train_inactives:n_train_inactives + n_val_inactives],
        'smiles': all_inactives['smiles'][n_train_inactives:n_train_inactives + n_val_inactives],
        'ids': all_inactives['ids'][n_train_inactives:n_train_inactives + n_val_inactives],
        'y': np.zeros(n_val_inactives, dtype=np.int32),
    }
    
    # Test inactives (next n_test_inactives)
    test_inactives = {
        'X': all_inactives['X'][n_train_inactives + n_val_inactives:n_train_inactives + n_val_inactives + n_test_inactives],
        'smiles': all_inactives['smiles'][n_train_inactives + n_val_inactives:n_train_inactives + n_val_inactives + n_test_inactives],
        'ids': all_inactives['ids'][n_train_inactives + n_val_inactives:n_train_inactives + n_val_inactives + n_test_inactives],
        'y': np.zeros(n_test_inactives, dtype=np.int32),
    }
    
    # ========================================================================
    # Step 8: Create train file
    # ========================================================================
    
    logger.info(f"[{target}] Creating train_s3_seed{seed}_ecfp4.npz...")
    
    # Extract only actives from train split
    train_actives_only = {
        'X': train_actives['X'][train_active_mask],
        'y': train_actives['y'][train_active_mask],
        'smiles': train_actives['smiles'][train_active_mask],
        'ids': train_actives['ids'][train_active_mask],
    }
    
    # Combine actives + inactives
    train_data = {
        'X': np.vstack([train_actives_only['X'], train_inactives['X']]),
        'y': np.concatenate([train_actives_only['y'], train_inactives['y']]),
        'smiles': np.concatenate([train_actives_only['smiles'], train_inactives['smiles']]),
        'ids': np.concatenate([train_actives_only['ids'], train_inactives['ids']]),
    }
    
    # Save
    output_train_path = final_path.parent / final_path.name.replace(
        '_final_ecfp4.npz',
        f'_train_s3_seed{seed}_ecfp4.npz'
    )
    
    if not dry_run:
        np.savez_compressed(
            output_train_path,
            X=train_data['X'],
            y=train_data['y'],
            smiles=train_data['smiles'],
            ids=train_data['ids']
        )
        logger.info(f"[{target}] ? Saved: {output_train_path.name} ({len(train_data['y'])} compounds)")
    else:
        logger.info(f"[{target}] [DRY RUN] Would save: {output_train_path.name} ({len(train_data['y'])} compounds)")
    
    # ========================================================================
    # Step 9: Create val file
    # ========================================================================
    
    logger.info(f"[{target}] Creating val_s3_seed{seed}_ecfp4.npz...")
    
    # Extract only actives from val split
    val_actives_only = {
        'X': val_actives['X'][val_active_mask],
        'y': val_actives['y'][val_active_mask],
        'smiles': val_actives['smiles'][val_active_mask],
        'ids': val_actives['ids'][val_active_mask],
    }
    
    # Combine actives + inactives
    val_data = {
        'X': np.vstack([val_actives_only['X'], val_inactives['X']]),
        'y': np.concatenate([val_actives_only['y'], val_inactives['y']]),
        'smiles': np.concatenate([val_actives_only['smiles'], val_inactives['smiles']]),
        'ids': np.concatenate([val_actives_only['ids'], val_inactives['ids']]),
    }
    
    # Save
    output_val_path = final_path.parent / final_path.name.replace(
        '_final_ecfp4.npz',
        f'_val_s3_seed{seed}_ecfp4.npz'
    )
    
    if not dry_run:
        np.savez_compressed(
            output_val_path,
            X=val_data['X'],
            y=val_data['y'],
            smiles=val_data['smiles'],
            ids=val_data['ids']
        )
        logger.info(f"[{target}] ? Saved: {output_val_path.name} ({len(val_data['y'])} compounds)")
    else:
        logger.info(f"[{target}] [DRY RUN] Would save: {output_val_path.name} ({len(val_data['y'])} compounds)")
    
    # ========================================================================
    # Step 10: Create test file
    # ========================================================================
    
    logger.info(f"[{target}] Creating test_s3_seed{seed}_ecfp4.npz...")
    
    # Combine actives + inactives
    test_data = {
        'X': np.vstack([test_actives['X'], test_inactives['X']]),
        'y': np.concatenate([test_actives['y'], test_inactives['y']]),
        'smiles': np.concatenate([test_actives['smiles'], test_inactives['smiles']]),
        'ids': np.concatenate([test_actives['ids'], test_inactives['ids']]),
    }
    
    # Save
    output_test_path = topu_path.parent / topu_path.name.replace(
        '_topUnbiased_ecfp4.npz',
        f'_test_s3_seed{seed}_ecfp4.npz'
    )
    
    if not dry_run:
        np.savez_compressed(
            output_test_path,
            X=test_data['X'],
            y=test_data['y'],
            smiles=test_data['smiles'],
            ids=test_data['ids']
        )
        logger.info(f"[{target}] ? Saved: {output_test_path.name} ({len(test_data['y'])} compounds)")
    else:
        logger.info(f"[{target}] [DRY RUN] Would save: {output_test_path.name} ({len(test_data['y'])} compounds)")
    
    # ========================================================================
    # Return statistics
    # ========================================================================
    
    return {
        'target': target,
        'seed': seed,
        'n_train': len(train_data['y']),
        'n_train_actives': int(n_train_actives),
        'n_train_inactives': int(n_train_inactives),
        'n_val': len(val_data['y']),
        'n_val_actives': int(n_val_actives),
        'n_val_inactives': int(n_val_inactives),
        'n_test': len(test_data['y']),
        'n_test_actives': int(n_test_actives),
        'n_test_inactives': int(n_test_inactives),
        'train_path': str(output_train_path),
        'val_path': str(output_val_path),
        'test_path': str(output_test_path),
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Setting 3 data files (train, val, test) with shuffled inactives"
    )
    parser.add_argument(
        '--dataset_dir',
        default=DEFAULT_DATASET_DIR,
        help=f"Path to topu_dataset directory (default: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        '--splits_dir',
        default=DEFAULT_SPLITS_DIR,
        help=f"Path to Setting 1 splits directory (default: {DEFAULT_SPLITS_DIR})"
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=None,
        help="Specific targets to process (default: all targets)"
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Seeds to use (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be created without actually creating files"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Creating Setting 3 Data Files (Train, Val, Test)")
    logger.info("=" * 80)
    logger.info(f"Dataset dir: {args.dataset_dir}")
    logger.info(f"Splits dir:  {args.splits_dir}")
    logger.info(f"Seeds:       {args.seeds}")
    logger.info(f"Dry run:     {args.dry_run}")
    logger.info("=" * 80)
    
    # Get targets
    if args.targets:
        targets = args.targets
        logger.info(f"Processing {len(targets)} specified targets")
    else:
        targets = discover_targets(args.dataset_dir)
        logger.info(f"Discovered {len(targets)} targets")
    
    # Process each target
    all_stats = []
    
    for i, target in enumerate(targets, 1):
        logger.info(f"\n[{i}/{len(targets)}] Processing {target}...")
        
        try:
            # Create files for EACH seed
            for seed in args.seeds:
                stats = create_setting3_files_for_target(
                    dataset_dir=args.dataset_dir,
                    splits_dir=args.splits_dir,
                    target=target,
                    seed=seed,
                    dry_run=args.dry_run
                )
                all_stats.append(stats)
            
            logger.info(f"[{target}] ? Complete")
            
        except Exception as e:
            logger.error(f"[{target}] ? Failed: {e}", exc_info=True)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Targets processed: {len(targets)}")
    logger.info(f"Seeds used: {args.seeds}")
    logger.info(f"Files created per target: {len(args.seeds) * 3} NPZ files")
    
    if args.dry_run:
        logger.info("\n??  DRY RUN - No files were actually created")
    else:
        logger.info(f"\n? Created Setting 3 data in: {args.dataset_dir}")
        logger.info(f"\nFor each seed, created:")
        logger.info(f"  - CHEMBL*_train_s3_seed{{seed}}_ecfp4.npz")
        logger.info(f"  - CHEMBL*_val_s3_seed{{seed}}_ecfp4.npz")
        logger.info(f"  - CHEMBL*_test_s3_seed{{seed}}_ecfp4.npz")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

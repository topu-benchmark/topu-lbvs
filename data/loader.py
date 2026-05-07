"""
data/loader.py
TopU-LBVS Setting 1, 2, & 3 - Data loading utilities.

NPZ arrays (both _final and _topUnbiased):
    X          : (n, 2048) uint8  - pre-computed ECFP4 fingerprints
    y          : (n,)      int    - labels (1=active, 0=inactive)
    ids        : (n,)             - compound IDs
    smiles     : (n,)             - SMILES strings

Splits directory structure (Setting 1 only):
    splits/seed_{seed}/{target}/train_idx.npy
    splits/seed_{seed}/{target}/val_idx.npy
    splits/seed_{seed}/{target}/split_info.json

Setting 2 NPZ filename pattern (TopU few-shot):
    CHEMBL*_train_s2_seed{seed}_ecfp4.npz
    CHEMBL*_val_s2_seed{seed}_ecfp4.npz
    CHEMBL*_test_s2_seed{seed}_ecfp4.npz

Setting 3 NPZ filename pattern (random ChEMBL* decoys):
    CHEMBL*_train_s3_seed{seed}_ecfp4.npz
    CHEMBL*_val_s3_seed{seed}_ecfp4.npz
    CHEMBL*_test_s3_seed{seed}_ecfp4.npz
"""

import json
import os
from glob import glob

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_npz(dataset_dir: str, target: str, suffix: str) -> str:
    """
    Find exactly one file matching CHEMBL*{suffix} inside dataset_dir/target/.
    Raises FileNotFoundError if zero or more than one match found.
    """
    pattern = os.path.join(dataset_dir, target, f"CHEMBL*{suffix}")
    matches = glob(pattern)
    if len(matches) == 0:
       raise FileNotFoundError(
          f"No file matching '{pattern}' found for target '{target}'.")
    if len(matches) > 1:
       raise ValueError(
          f"Multiple files matching '{pattern}' found for target '{target}': {matches}")

    return matches[0]


def _load_npz(path: str, include_smiles: bool):
    """
    Load X, y, ids, and optionally smiles from an npz file.

    Returns
    -------
    include_smiles=False : X, y, ids
    include_smiles=True  : X, y, ids, smiles
    """
    data = np.load(path, allow_pickle=True)

    X   = data["X"]  # (n, 2048)  # uint8, cast to float32 in each model as needed

    y   = data["y"].astype(np.int32)     # (n,)
    ids = data["ids"]                    # (n,)

    if include_smiles:
        smiles = data["smiles"]          # (n,)
        return X, y, ids, smiles

    return X, y, ids


# ---------------------------------------------------------------------------
# Public API - Setting 1 loaders
# ---------------------------------------------------------------------------

def load_final_npz(
    dataset_dir: str,
    target: str,
    include_smiles: bool = False,
):
    """
    Load the full _final_ecfp4.npz (training pool) for a target.

    Returns
    -------
    include_smiles=False : X, y, ids
    include_smiles=True  : X, y, ids, smiles
    """
    path = _find_npz(dataset_dir, target, "_final_ecfp4.npz")
    return _load_npz(path, include_smiles=include_smiles)


def load_topu_npz(
    dataset_dir: str,
    target: str,
    include_smiles: bool = False,
):
    """
    Load the full _topUnbiased_ecfp4.npz (test set) for a target.
    This file is NEVER used for training - test only.

    Returns
    -------
    include_smiles=False : X_test, y_test, ids_test
    include_smiles=True  : X_test, y_test, ids_test, smiles_test
    """
    path = _find_npz(dataset_dir, target, "_topUnbiased_ecfp4.npz")
    return _load_npz(path, include_smiles=include_smiles)


def load_split(splits_dir: str, target: str, seed: int):
    """
    Load train/val split indices and metadata for a target + seed.

    Returns
    -------
    train_idx  : np.ndarray of int64 - indices into _final_ecfp4.npz
    val_idx    : np.ndarray of int64 - indices into _final_ecfp4.npz
    split_info : dict                - full metadata from split_info.json
    """
    seed_dir = os.path.join(splits_dir, f"seed_{seed}", target)

    train_path = os.path.join(seed_dir, "train_idx.npy")
    val_path   = os.path.join(seed_dir, "val_idx.npy")
    info_path  = os.path.join(seed_dir, "split_info.json")

    for p in (train_path, val_path, info_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Split file not found: '{p}'. "
                f"Run create_splits.py first for target='{target}', seed={seed}."
            )

    train_idx = np.load(train_path).astype(np.int64)
    val_idx   = np.load(val_path).astype(np.int64)

    with open(info_path, "r") as f:
        split_info = json.load(f)

    return train_idx, val_idx, split_info


def get_train_val(
    dataset_dir: str,
    splits_dir: str,
    target: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load split indices and apply them to _final_ecfp4.npz to produce
    ready-to-use train and val arrays.

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    splits_dir     : path to splits/
    target         : target name, e.g. 'egfr'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles_train and smiles_val
                     (needed for RDKitRF descriptor computation)

    Returns
    -------
    include_smiles=False:
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val,
        split_info

    include_smiles=True:
        X_train, y_train, ids_train, smiles_train,
        X_val,   y_val,   ids_val,   smiles_val,
        split_info
    """
    train_idx, val_idx, split_info = load_split(splits_dir, target, seed)

    if include_smiles:
        X, y, ids, smiles = load_final_npz(dataset_dir, target, include_smiles=True)
        return (
            X[train_idx], y[train_idx], ids[train_idx], smiles[train_idx],
            X[val_idx],   y[val_idx],   ids[val_idx],   smiles[val_idx],
            split_info,
        )

    X, y, ids = load_final_npz(dataset_dir, target, include_smiles=False)
    return (
        X[train_idx], y[train_idx], ids[train_idx],
        X[val_idx],   y[val_idx],   ids[val_idx],
        split_info,
    )


def get_test(
    dataset_dir: str,
    target: str,
    include_smiles: bool = False,
):
    """
    Load the full TopU test set for a target.
    No split indices needed - the entire _topUnbiased_ecfp4.npz is the test set.

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    include_smiles : if True, also return smiles_test
                     (needed for RDKitRF descriptor computation)

    Returns
    -------
    include_smiles=False : X_test, y_test, ids_test
    include_smiles=True  : X_test, y_test, ids_test, smiles_test
    """
    return load_topu_npz(dataset_dir, target, include_smiles=include_smiles)


# ---------------------------------------------------------------------------
# Public API - Setting 2 loaders (TopU few-shot)
# ---------------------------------------------------------------------------

def load_setting2_file(
    dataset_dir: str,
    target: str,
    split_type: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load a Setting 2 file directly (train, val, or test).

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    split_type     : 'train', 'val', or 'test'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False : X, y, ids
    include_smiles=True  : X, y, ids, smiles
    """
    suffix = f"_{split_type}_s2_seed{seed}_ecfp4.npz"
    path = _find_npz(dataset_dir, target, suffix)
    return _load_npz(path, include_smiles=include_smiles)


def get_train_val_setting2(
    dataset_dir: str,
    target: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load Setting 2 train and val files (already split, no indices needed).

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False:
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val

    include_smiles=True:
        X_train, y_train, ids_train, smiles_train,
        X_val,   y_val,   ids_val,   smiles_val
    """
    if include_smiles:
        X_train, y_train, ids_train, smiles_train = load_setting2_file(
            dataset_dir, target, 'train', seed, include_smiles=True
        )
        X_val, y_val, ids_val, smiles_val = load_setting2_file(
            dataset_dir, target, 'val', seed, include_smiles=True
        )
        return (
            X_train, y_train, ids_train, smiles_train,
            X_val,   y_val,   ids_val,   smiles_val,
        )

    X_train, y_train, ids_train = load_setting2_file(
        dataset_dir, target, 'train', seed, include_smiles=False
    )
    X_val, y_val, ids_val = load_setting2_file(
        dataset_dir, target, 'val', seed, include_smiles=False
    )
    return (
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val,
    )


def get_test_setting2(
    dataset_dir: str,
    target: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load Setting 2 test file.

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False : X_test, y_test, ids_test
    include_smiles=True  : X_test, y_test, ids_test, smiles_test
    """
    return load_setting2_file(dataset_dir, target, 'test', seed, include_smiles=include_smiles)


# ---------------------------------------------------------------------------
# Public API - Setting 3 loaders
# ---------------------------------------------------------------------------

def load_setting3_file(
    dataset_dir: str,
    target: str,
    split_type: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load a Setting 3 file directly (train, val, or test).

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    split_type     : 'train', 'val', or 'test'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False : X, y, ids
    include_smiles=True  : X, y, ids, smiles
    """
    suffix = f"_{split_type}_s3_seed{seed}_ecfp4.npz"
    path = _find_npz(dataset_dir, target, suffix)
    return _load_npz(path, include_smiles=include_smiles)


def get_train_val_setting3(
    dataset_dir: str,
    target: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load Setting 3 train and val files (already split, no indices needed).

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False:
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val

    include_smiles=True:
        X_train, y_train, ids_train, smiles_train,
        X_val,   y_val,   ids_val,   smiles_val
    """
    if include_smiles:
        X_train, y_train, ids_train, smiles_train = load_setting3_file(
            dataset_dir, target, 'train', seed, include_smiles=True
        )
        X_val, y_val, ids_val, smiles_val = load_setting3_file(
            dataset_dir, target, 'val', seed, include_smiles=True
        )
        return (
            X_train, y_train, ids_train, smiles_train,
            X_val,   y_val,   ids_val,   smiles_val,
        )

    X_train, y_train, ids_train = load_setting3_file(
        dataset_dir, target, 'train', seed, include_smiles=False
    )
    X_val, y_val, ids_val = load_setting3_file(
        dataset_dir, target, 'val', seed, include_smiles=False
    )
    return (
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val,
    )


def get_test_setting3(
    dataset_dir: str,
    target: str,
    seed: int,
    include_smiles: bool = False,
):
    """
    Load Setting 3 test file.

    Parameters
    ----------
    dataset_dir    : path to topu_dataset/
    target         : target name, e.g. 'egfr'
    seed           : one of 2026, 2027, 2028
    include_smiles : if True, also return smiles

    Returns
    -------
    include_smiles=False : X_test, y_test, ids_test
    include_smiles=True  : X_test, y_test, ids_test, smiles_test
    """
    return load_setting3_file(dataset_dir, target, 'test', seed, include_smiles=include_smiles)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def discover_targets(dataset_dir: str) -> list:
    """
    Discover all valid targets in dataset_dir.
    A valid target is a subdirectory containing at least one
    CHEMBL*_final_ecfp4.npz file.

    Returns
    -------
    Sorted list of target name strings.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: '{dataset_dir}'")

    targets = []
    for name in os.listdir(dataset_dir):
        target_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(target_dir):
            continue
        if glob(os.path.join(target_dir, "CHEMBL*_final_ecfp4.npz")):
            targets.append(name)

    if not targets:
        raise FileNotFoundError(
            f"No valid targets found in '{dataset_dir}'. "
            "Expected subdirectories containing CHEMBL*_final_ecfp4.npz files."
        )

    return sorted(targets)

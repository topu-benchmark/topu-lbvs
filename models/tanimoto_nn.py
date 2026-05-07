"""
models/tanimoto_nn.py
TopU-LBVS Setting 1 - Tanimoto Nearest-Neighbour baseline.

Model description (paper Appendix A.1):
    For each test compound, compute the maximum Tanimoto coefficient
    against all training actives using 2048-bit Morgan fingerprints,
    and use this maximum similarity as the ranking score. No learning
    is involved - the method ranks compounds purely by structural
    proximity to known actives.

Tanimoto similarity (binary fingerprints):
    T(a, b) = |a n b| / |a ? b|
            = dot(a, b) / (sum(a) + sum(b) - dot(a, b))
    Range: [0, 1]. T=1 means identical fingerprint. T=0 means no overlap.

Implementation:
    fit()          -extracts and stores fingerprints of training actives only.
                     Decoys are discarded. Val set is ignored entirely.
    predict_proba() - computes full (n_test * n_actives) Tanimoto matrix
                     via vectorized matrix multiplication, then takes row-max.
                     No loop over test compounds.

is_deterministic = False:
    TanimotoNN itself has no random state (fit() is a pure array slice,
    predict_proba() is pure matrix arithmetic), but each seed produces
    a different scaffold-based split, which means a different set of
    training actives is stored in fit(). Different training actives
    -> different Tanimoto scores at test time. Runner runs all 3 seeds
    (2026, 2027, 2028) and reports mean +/- std, capturing split-induced
    variance like any other model.


Edge cases handled:
    - No actives in training set ? predict_proba returns all zeros
    - All-zero fingerprint      ? denom=0 ? T=0.0 (via np.where guard)
    - smiles args               ? accepted and ignored (fingerprints used)
"""

from typing import Optional

import numpy as np

from models.base import BaseModel


class TanimotoNN(BaseModel):
    """
    Max Tanimoto similarity to training actives as the ranking score.

    Parameters
    ----------
    seed : int
        Accepted for API consistency with other models. Not used -
        TanimotoNN has no random state.
    """

    def __init__(self, seed: int = 2026):
        self._seed           = seed        # stored but never used
        self._X_actives      = None        # (n_train_actives, 2048) float32
        self._n_train_actives = 0

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:      np.ndarray,
        y_train:      np.ndarray,
        X_val:        np.ndarray,
        y_val:        np.ndarray,
        smiles_train: Optional[np.ndarray] = None,
        smiles_val:   Optional[np.ndarray] = None,
    ) -> None:
        """
        Store fingerprints of training actives only.

        No learning occurs. Decoys are discarded immediately.
        Val set is ignored entirely - TanimotoNN has no model selection.
        smiles_train and smiles_val are accepted but ignored.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, 2048), float32
            Pre-computed ECFP4 fingerprints for training compounds.
        y_train : np.ndarray, shape (n_train,), int32
            Binary labels. Actives (y=1) are extracted and stored.
        X_val, y_val : ignored
        smiles_train, smiles_val : ignored
        """
        y_train = np.asarray(y_train, dtype=np.int32)

        # Extract training actives only — decoys are never used
        active_mask          = y_train == 1
        self._X_actives      = X_train[active_mask].astype(np.float32)
        self._n_train_actives = int(active_mask.sum())

    def predict_proba(
        self,
        X_test:      np.ndarray,
        smiles_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score each test compound by max Tanimoto similarity to training actives.

        Vectorized over all (n_test * n_actives) pairs simultaneously.
        smiles_test is accepted but ignored.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, 2048), float32
            Pre-computed ECFP4 fingerprints for test compounds.

        Returns
        -------
        np.ndarray, shape (n_test,), float64
            Max Tanimoto similarity in [0, 1] for each test compound.
            Higher = more similar to training actives = ranked higher.
        """
        if self._X_actives is None:
            raise RuntimeError(
                "TanimotoNN.predict_proba() called before fit(). "
                "Call fit() first."
            )

        # Degenerate case: no actives in training set
        if self._n_train_actives == 0:
            return np.zeros(len(X_test), dtype=np.float64)

        X_test    = np.asarray(X_test,          dtype=np.float64)
        X_actives = np.asarray(self._X_actives, dtype=np.float64)

        # Intersection counts — vectorized dot products
        # inter[i, j] = number of bits set in BOTH X_test[i] AND X_actives[j]
        # shape: (n_test, n_actives)
        inter = X_test @ X_actives.T

        # Bit counts per compound
        sum_test    = X_test.sum(axis=1, keepdims=True)     # (n_test, 1)
        sum_actives = X_actives.sum(axis=1, keepdims=True)  # (n_actives, 1)

        # Union counts: |a ? b| = sum(a) + sum(b) - |a n b|
        # shape: (n_test, n_actives)  via broadcasting
        union = sum_test + sum_actives.T - inter

        # Tanimoto: inter / union, guarded against division by zero
        # (union=0 only if both fingerprints are all-zeros ? T=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tanimoto = np.where(union > 0, inter / union, 0.0)

        # Max similarity over all training actives — the ranking score
        return tanimoto.max(axis=1).astype(np.float64)

    @property
    def is_deterministic(self) -> bool:
        """
        False - different seeds use different scaffold-based splits, meaning
        different training actives are stored in fit(). This produces different
        Tanimoto scores across seeds even though TanimotoNN has no random state.
         Runner runs all 3 seeds and averages results like any other model.
        """
        return False

    @property
    def name(self) -> str:
        return "tanimoto_nn"
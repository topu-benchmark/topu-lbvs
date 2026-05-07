"""
models/morgan_rf.py
TopU-LBVS Setting 1 - Morgan Fingerprint + Random Forest baseline.

Model description (paper Appendix A.1):
    Compute 2048-bit Morgan (ECFP4) fingerprints with radius r=2 using
    RDKit, then train a Random Forest classifier with up to 500 trees,
    max_features=sqrt(d), and balanced class weights to account for the
    active-to-decoy imbalance. Early stopping is applied using validation
    PR-AUC, adding trees in steps of `step` and stopping after `patience`
    steps without improvement. The predicted probability of the active
    class is used as the ranking score.

Implementation notes:
    - Fingerprints are pre-computed and stored in _final_ecfp4.npz and
      _topUnbiased_ecfp4.npz. MorganRF uses X directly - no recomputation.
    - class_weight='balanced': w_j = n / (n_classes * n_j)
      With 1:10 ratio: w_active=5.5, w_inactive=0.55 (exact values
      depend on split counts, sklearn computes automatically)
    - max_features='sqrt': sqrt(2048) ~ 45 features per split
    - n_jobs=-1: uses all available CPU cores, does not affect scores
    - Seed controls both tree bootstrap sampling and feature selection
    - warm_start=True used during early stopping loop to incrementally
      add trees without refitting from scratch each step
    - Final model is refit from scratch at best_n (no warm_start) for
      a clean, reproducible model state
    - smiles_train / smiles_val / smiles_test are accepted but ignored
      (fingerprints already computed in npz)

Edge cases handled:
    - Training set contains only one class (degenerate split):
      predict_proba returns zeros (no actives predicted)
    - sklearn RF classes_ ordering not guaranteed to be [0,1]:
      always look up class 1 index explicitly
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.base import BaseModel
from metrics.screening import prauc


class MorganRF(BaseModel):
    """
    Random Forest trained on pre-computed 2048-bit ECFP4 fingerprints.
    Uses validation PR-AUC for early stopping - adds trees in steps,
    stops when val PR-AUC stops improving.

    Parameters
    ----------
    seed : int
        Random seed for the RF. Controls bootstrap sampling and
        feature selection at each split. One of 2026, 2027, 2028.
    n_estimators : int
        Maximum number of trees. Default 500 (paper spec).
    n_jobs : int
        Parallel jobs for fitting and prediction. Default -1 (all cores).
    patience : int
        Number of steps without val PR-AUC improvement before stopping.
        Default 5 (50 trees without improvement).
    step : int
        Number of trees to add per early stopping check. Default 10.
    """

    def __init__(
        self,
        seed:         int = 2026,
        n_estimators: int = 500,
        n_jobs:       int = -1,
        patience:     int = 5,
        step:         int = 10,
    ):
        self._seed         = seed
        self._n_estimators = n_estimators
        self._n_jobs       = n_jobs
        self._patience     = patience
        self._step         = step
        self._clf          = None   # set after fit()

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
        Train the Random Forest on pre-computed ECFP4 fingerprints.
        Uses val PR-AUC for early stopping - trees are added in steps
        of `step`, stopping after `patience` steps without improvement.
        Max trees capped at n_estimators (default 500).
        smiles_train and smiles_val are accepted but ignored.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, 2048), float32
            Pre-computed ECFP4 fingerprints for training compounds.
        y_train : np.ndarray, shape (n_train,), int32
            Binary labels. Expected 1:10 active:inactive ratio.
        X_val : np.ndarray, shape (n_val, 2048), float32
            Used for early stopping via val PR-AUC.
        y_val : np.ndarray, shape (n_val,), int32
            Binary labels for validation. Used for early stopping only.
        smiles_train : ignored
        smiles_val   : ignored
        """
        best_score = -1.0
        best_n     = self._step
        no_improve = 0
        # NEW - added these lines
        n_total      = len(y_train)
        n_active     = int(y_train.sum())
        n_inactive   = n_total - n_active
        class_weight = {
        0: n_total / (2.0 * n_inactive) if n_inactive > 0 else 1.0,
        1: n_total / (2.0 * n_active)   if n_active   > 0 else 1.0,
        }

        # warm_start=True lets us add trees incrementally without refitting
        self._clf = RandomForestClassifier(
            n_estimators = self._step,
            max_features = "sqrt",           # sqrt(2048) ~ 45 features/split
            class_weight = class_weight,       # upweights rare actives
            random_state = self._seed,
            n_jobs       = self._n_jobs,
            warm_start   = True,             # incremental tree addition
        )

        for n in range(self._step, self._n_estimators + 1, self._step):
            self._clf.n_estimators = n
            self._clf.fit(X_train, y_train)

            if 1 in self._clf.classes_:
                idx   = list(self._clf.classes_).index(1)
                score = prauc(y_val, self._clf.predict_proba(X_val)[:, idx])
                if score > best_score:
                    best_score = score
                    best_n     = n
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self._patience:
                        break

        # Refit from scratch at best_n - clean final model without warm_start
        self._clf = RandomForestClassifier(
            n_estimators = best_n,
            max_features = "sqrt",
            class_weight = class_weight,
            random_state = self._seed,
            n_jobs       = self._n_jobs,
        )
        self._clf.fit(X_train, y_train)

    def predict_proba(
        self,
        X_test:      np.ndarray,
        smiles_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return predicted probability of active class for each test compound.

        Uses the RF's vote fraction across all trees as the ranking score.
        smiles_test is accepted but ignored.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, 2048), float32
            Pre-computed ECFP4 fingerprints for test compounds.

        Returns
        -------
        np.ndarray, shape (n_test,), float64
            P(active | compound) in [0, 1].
            Higher = ranked higher = more likely active.
        """
        if self._clf is None:
            raise RuntimeError(
                "MorganRF.predict_proba() called before fit(). "
                "Call fit() first."
            )

        # Guard: if training set had only one class, class 1 may be absent
        if 1 not in self._clf.classes_:
            # Model never saw an active - return zeros (all compounds ranked equally low)
            return np.zeros(len(X_test), dtype=np.float64)

        proba      = self._clf.predict_proba(X_test)         # shape (n_test, n_classes)
        class1_idx = list(self._clf.classes_).index(1)       # index of class 1 column
        return proba[:, class1_idx].astype(np.float64)

    @property
    def is_deterministic(self) -> bool:
        """
        False - RF uses random bootstrap sampling and random feature
        selection, both controlled by seed. Different seeds produce
        different models and different scores.
        """
        return False

    @property
    def name(self) -> str:
        return "morgan_rf"
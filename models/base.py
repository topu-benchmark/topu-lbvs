"""
models/base.py
TopU-LBVS Setting 1 - Abstract base class for all models.

Every model in the pipeline (classical, GNN, pretrained) must subclass
BaseModel and implement all abstract methods and properties.

The runner (training/runner.py) only ever interacts with BaseModel -
it never imports or checks specific model classes directly. This means:
    - Adding a new model never requires changing the runner
    - Missing method implementations are caught at instantiation time,
      not silently at runtime after hours of training

Abstract interface
------------------
fit(X_train, y_train, X_val, y_val,
    smiles_train=None, smiles_val=None)     must be implemented

predict_proba(X_test,
              smiles_test=None)             must be implemented

is_deterministic    bool property           must be implemented
name                str  property           must be implemented

SMILES arguments
----------------
smiles_train, smiles_val, smiles_test are always passed by the runner
but are None by default. Models that do not need SMILES (MorganRF,
TanimotoNN, all GNNs) simply ignore them. Only RDKitRF uses them to
compute 200-dim physicochemical descriptors via RDKit.

Concrete models
---------------
models/morgan_rf.py     MorganRF        is_deterministic=False
models/tanimoto_nn.py   TanimotoNN      is_deterministic=True
models/rdkit_rf.py      RDKitRF         is_deterministic=False
models/gin.py           GIN, GINWithFP  is_deterministic=False
models/gat.py           GAT, GATWithFP  is_deterministic=False
models/gps.py           GPS, GPSWithFP  is_deterministic=False
models/dmpnn.py         DMPNN           is_deterministic=False
models/molformer.py     MolFormer       is_deterministic=False
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all TopU-LBVS Setting 1 models.

    Subclasses must implement:
        fit()
        predict_proba()
        is_deterministic (property)
        name (property)

    Usage in runner
    ---------------
        model = SomeModel(seed=2026)
        model.fit(X_train, y_train, X_val, y_val,
                  smiles_train=smiles_train,
                  smiles_val=smiles_val)
        scores = model.predict_proba(X_test, smiles_test=smiles_test)
        metrics = compute_all(y_test, scores)
    """

    # ------------------------------------------------------------------
    # Abstract methods - must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        X_val:         np.ndarray,
        y_val:         np.ndarray,
        smiles_train:  Optional[np.ndarray] = None,
        smiles_val:    Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, 2048), float32
            Pre-computed ECFP4 fingerprints for training compounds.
            GNN models may ignore X and build graphs from smiles_train.

        y_train : np.ndarray, shape (n_train,), int32
            Binary labels - 1=active, 0=inactive.
            Class ratio is 1:10 (active:inactive) after splitting.

        X_val : np.ndarray, shape (n_val, 2048), float32
            Pre-computed ECFP4 fingerprints for validation compounds.

        y_val : np.ndarray, shape (n_val,), int32
            Binary labels for validation set.
            Used ONLY for early stopping / model selection (val PR-AUC).
            Never reported as a result metric.

        smiles_train : np.ndarray of str, shape (n_train,), optional
            SMILES strings for training compounds.
            Required by RDKitRF. Ignored by all other models.

        smiles_val : np.ndarray of str, shape (n_val,), optional
            SMILES strings for validation compounds.
            Required by RDKitRF. Ignored by all other models.

        Returns
        -------
        None. Model state is stored internally (e.g. self._clf, self._model).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        X_test:       np.ndarray,
        smiles_test:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score all test compounds. Higher score = more likely active.

        Must be called AFTER fit(). Behaviour before fit() is undefined.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, 2048), float32
            Pre-computed ECFP4 fingerprints for test compounds.

        smiles_test : np.ndarray of str, shape (n_test,), optional
            SMILES strings for test compounds.
            Required by RDKitRF. Ignored by all other models.

        Returns
        -------
        np.ndarray, shape (n_test,), float64
            Real-valued scores - one per test compound.
            Higher score = compound ranked higher = more likely active.
            Scale does not matter (EF, PR-AUC, ROC-AUC are all rank-based).
            Must not contain NaN or Inf.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract properties - must be implemented by every subclass
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """
        Whether this model produces identical results regardless of seed.

        True  ? TanimotoNN only (no learned parameters, pure similarity search)
                Runner will skip seeds 2027 and 2028 for this model.

        False ? All other models (RF, GNN, pretrained). Runner runs all
                3 seeds and averages per-target metrics.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique string identifier for this model.

        Used for:
            - wandb run group:     wandb.init(group=model.name)
            - results directory:   results/setting1/{model.name}/
            - CSV column headers:  df["model"] = model.name
            - logging messages

        Must be lowercase with underscores. Must be unique across all models.

        Expected values
        ---------------
        "morgan_rf"
        "tanimoto_nn"
        "rdkit_rf"
        "gin"
        "gin_fp"
        "gat"
        "gat_fp"
        "gps"
        "gps_fp"
        "dmpnn"
        "molformer"

        Returns
        -------
        str
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete helper - shared across all subclasses, not overridden
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        det = "deterministic" if self.is_deterministic else "stochastic"
        return f"{self.__class__.__name__}(name='{self.name}', {det})"
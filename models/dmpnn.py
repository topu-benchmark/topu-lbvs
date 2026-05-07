"""
models/dmpnn.py
TopU-LBVS Setting 1 - Directed Message Passing Neural Network (D-MPNN) baseline.

Model description (paper Appendix A.3):
    Directed Message Passing Neural Network implemented via Chemprop v2.
    Passes messages along directed edges rather than nodes, mitigating
    oversmoothing and enabling precise encoding of local chemical environments.
    The molecular embedding is augmented with 200-dimensional RDKit
    physicochemical descriptors before the FFN classification head.

Architecture (paper spec):
    - 3 message-passing steps (depth=3)
    - Hidden dimension 300
    - 2-layer FFN, hidden size 300
    - Dropout p=0.0
    - ReLU activations
    - RDKit 200-dim physicochemical descriptors concatenated to molecular embedding

Training:
- Binary cross-entropy loss (chemprop default cnn.BCELoss)
    - NormAggregation pooling (scaled sum, per Chemprop v2 paper 4.1)
    - Descriptor StandardScaler fit on train, applied to val/test, and
      passed to MPNN as X_d_transform (chemprop reference pattern from
      examples/extra_features_descriptors.ipynb)
    - Adam optimizer with Chemprop v2's default Noam LR schedule:
        init_lr = 1e-4
        max_lr  = 1e-3  (reached after 2 warmup epochs)
        final_lr = 1e-4 
    - Edge update uses preactivation initial edge hidden states
      (Chemprop v2 carries this forward from v1; differs from
       Yang et al. 2019 which used postactivation)
    - Early stopping on val PR-AUC (patience=10 epochs)
    - Max 50 epochs
    - Batch size 64 (Chemprop default)

Implementation notes:
    - Uses Chemprop v2 Python API with PyTorch Lightning backend
    - Lightning trainer output suppressed during benchmark runs
    - Requires SMILES strings (not pre-computed fingerprints)
    - runner.py _needs_smiles() recognises "dmpnn" class name - smiles always passed
    - GPU used automatically if available
    - Model checkpoint saved to temp dir, cleaned up after prediction
"""

import logging
import os
import tempfile
import warnings
from typing import Optional
import shutil  
import numpy as np
import torch
import torch.serialization
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Suppress lightning and chemprop verbosity during benchmark runs
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


def _compute_rdkit_descriptors(smiles_list: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute 200-dim RDKit physicochemical descriptors for each SMILES.
    Returns array of shape (n, 200) or None if descriptastorus not available.
    Invalid SMILES get a zero vector.
    """
    try:
        from descriptastorus.descriptors import rdNormalizedDescriptors
        generator = rdNormalizedDescriptors.RDKit2DNormalized()

        descs = []
        for smi in smiles_list:
            try:
                result = generator.process(smi)
                if result is None or result[0] is False:
                    descs.append(np.zeros(200, dtype=np.float32))
                else:
                    # result[0] is success flag, result[1:] are descriptors
                    arr = np.array(result[1:], dtype=np.float32)
                    # Replace NaN/Inf with 0
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    descs.append(arr[:200])
            except Exception:
                descs.append(np.zeros(200, dtype=np.float32))

        return np.stack(descs, axis=0)   # (n, 200)

    except ImportError:
        logger.warning(
            "descriptastorus not available - running D-MPNN without RDKit descriptors. "
            "Install with: pip install git+https://github.com/bp-kelley/descriptastorus"
        )
        return None


class DMPNN(BaseModel):
    """
    Directed Message Passing Neural Network via Chemprop v2.

    Takes SMILES strings as input. Augments learned graph embedding with
    200-dim RDKit physicochemical descriptors before the classification head.

    Parameters
    ----------
    seed : int
        Random seed for model initialisation and training. One of 2026, 2027, 2028.
    hidden_dim : int
        Hidden dimension for message passing and FFN. Default 300 (paper spec).
    depth : int
        Number of message passing steps. Default 3 (paper spec).
    ffn_num_layers : int
        Number of FFN layers. Default 2 (paper spec).
    dropout : float
        Dropout probability. Default 0.0 (paper spec).
    max_epochs : int
        Maximum training epochs. Default 50 (paper spec).
    patience : int
        Early stopping patience on val PR-AUC. Default 10 (paper spec).
    batch_size : int
        Batch size. Default 64 -
    use_wandb : bool
        If True, attaches a Lightning WandbLogger to the existing wandb run
        initialized by runner.py. Logs train_loss, val_loss and val/prc
        per epoch. Default False.
    """

    def __init__(
        self,
        seed:          int   = 2026,
        hidden_dim:    int   = 300,
        depth:         int   = 3,
        ffn_num_layers:int   = 2,
        dropout:       float = 0.0,
        max_epochs:    int   = 50,
        patience:      int   = 20,
        batch_size:    int   = 64,
        use_wandb:     bool  = False,
        
    ):
        self._seed           = seed
        self._hidden_dim     = hidden_dim
        self._depth          = depth
        self._ffn_num_layers = ffn_num_layers
        self._dropout        = dropout
        self._max_epochs     = max_epochs
        self._patience       = patience
        self._batch_size     = batch_size
        self._model          = None    # set after fit()
        self._ckpt_path      = None    # best checkpoint path
        self._tmpdir         = None    # temp dir for checkpoints
        self._use_wandb      = use_wandb  # whether to log train/va curves to wandb
        self._x_d_scaler = None

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
        Train D-MPNN on SMILES strings using Chemprop v2 + Lightning.

        X_train / X_val are accepted for interface consistency but ignored.
        smiles_train and smiles_val are required.

        Parameters
        ----------
        X_train : ignored (fingerprints not used by D-MPNN)
        y_train : np.ndarray, shape (n_train,), int32 - binary labels
        X_val   : ignored
        y_val   : np.ndarray, shape (n_val,), int32 - binary labels
        smiles_train : np.ndarray of str, shape (n_train,) - required
        smiles_val   : np.ndarray of str, shape (n_val,)   - required
        """
        if smiles_train is None or smiles_val is None:
            raise ValueError(
                "DMPNN requires smiles_train and smiles_val. "
                "Ensure runner passes include_smiles=True."
            )

        from chemprop import data, models, nn as cnn
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        # Set seed for reproducibility
        pl.seed_everything(self._seed, workers=True)

        # -- Compute RDKit descriptors (optional augmentation) ----------------
        x_d_train = _compute_rdkit_descriptors(smiles_train)
        x_d_val   = _compute_rdkit_descriptors(smiles_val)

        # -- Build datasets ---------------------------------------------------
        y_train_2d = y_train.astype(np.float32).reshape(-1, 1)
        y_val_2d   = y_val.astype(np.float32).reshape(-1, 1)

          # D3: Build datapoints with defensive error handling.
        # Chemprop's from_smi can raise on rare SMILES that pass RDKit's
        # MolFromSmiles but fail Chemprop's featurizer (exotic stereochemistry,
        # unusual atom types, etc.). Drop such rows atomically to keep
        # smiles/label/descriptor indices aligned.
        def _build_datapoints(smis, ys, xds, split_name):
            dps, skipped = [], 0
            for i in range(len(smis)):
                smi = smis[i]
                y   = ys[i]
                xd  = xds[i] if xds is not None else None
                try:
                    if xd is not None:
                        dps.append(MoleculeDatapoint.from_smi(smi, y=y, x_d=xd))
                    else:
                        dps.append(MoleculeDatapoint.from_smi(smi, y=y))
                except Exception as e:
                    skipped += 1
                    if skipped <= 3:
                        logger.warning(
                            f"[dmpnn] {split_name}: dropping invalid SMILES "
                            f"{str(smi)[:80]!r}: {type(e).__name__}: {e}"
                        )
            if skipped:
                logger.warning(
                    f"[dmpnn] {split_name}: dropped {skipped}/{len(smis)} "
                    f"invalid SMILES ({100*skipped/len(smis):.2f}%)"
                )
            return dps

        train_data = _build_datapoints(smiles_train, y_train_2d, x_d_train, "train")
        val_data   = _build_datapoints(smiles_val,   y_val_2d,   x_d_val,   "val")

        train_dset = MoleculeDataset(train_data)
        val_dset   = MoleculeDataset(val_data)
        
        if x_d_train is not None:
            self._x_d_scaler = train_dset.normalize_inputs("X_d")
            val_dset.normalize_inputs("X_d", self._x_d_scaler)
            from chemprop.nn import ScaleTransform
            X_d_transform = ScaleTransform.from_standard_scaler(self._x_d_scaler)
        else:
            self._x_d_scaler = None
            X_d_transform = None
        
        train_loader = build_dataloader(
            train_dset, batch_size=self._batch_size, shuffle=True,
            num_workers=0, seed=self._seed,
        )
        val_loader = build_dataloader(
            val_dset, batch_size=self._batch_size, shuffle=False,
            num_workers=0,
        )

        # -- Build model ------------------------------------------------------
        mp = cnn.BondMessagePassing(
            d_h        = self._hidden_dim,
            depth      = self._depth,
            dropout    = self._dropout,
            activation = "relu",
        )
        agg = cnn.NormAggregation()
        # -- pos_weight class balance (matching previous code) ---------------
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight_scalar = n_neg / max(n_pos, 1)
        logger.info(
            f"[dmpnn] train class balance: {n_pos} active / {n_neg} inactive "
            f"(pos_weight={pos_weight_scalar:.2f})"
        )

        import torch.nn.functional as F
        class BCELossWithPosWeight(cnn.BCELoss):
            def __init__(self, pos_weight, task_weights=1.0):
                super().__init__(task_weights=task_weights)
                self.register_buffer(
                    "pos_weight",
                    torch.as_tensor(pos_weight, dtype=torch.float32),
                )
            def _calc_unreduced_loss(self, preds, targets, *args):
                return F.binary_cross_entropy_with_logits(
                    preds, targets,
                    pos_weight=self.pos_weight,
                    reduction="none",
                )

        # x_d_train adds 200 RDKit descriptor dims to molecular embedding
        d_xd      = x_d_train.shape[1] if x_d_train is not None else 0
        ffn_input = self._hidden_dim + d_xd

        ffn = cnn.BinaryClassificationFFN(
            input_dim  = ffn_input,
            hidden_dim = self._hidden_dim,
            n_layers   = self._ffn_num_layers,
            dropout    = self._dropout,
            criterion  = BCELossWithPosWeight(pos_weight=pos_weight_scalar),
            
        )

        # Add PR-AUC as validation metric - matches paper model selection criterion
        from chemprop.nn.metrics import BinaryAUPRC
       # self._model = models.MPNN(mp, agg, ffn, metrics=[BinaryAUPRC()])
        self._model = models.MPNN(
            mp, agg, ffn,
            X_d_transform = X_d_transform,
            metrics       = [BinaryAUPRC()],
        )
        # -- Callbacks --------------------------------------------------------
        self._tmpdir = tempfile.mkdtemp(prefix="dmpnn_")
        ckpt_callback = ModelCheckpoint(
            dirpath   = self._tmpdir,
            filename  = "best",
            monitor   = "val/prc",          # val PR-AUC - higher is better (paper spec)
            mode      = "max",
            save_top_k= 1,
        )
        early_stop = EarlyStopping(
            monitor  = "val/prc",
            patience = self._patience,
            mode     = "max",
        )

        # -- Wandb logger (attaches to existing run from runner.py) ----------
        if self._use_wandb:
            from lightning.pytorch.loggers import WandbLogger
            import wandb as wandb_module
            pl_logger = WandbLogger(experiment=wandb_module.run, log_model=False)
        else:
            pl_logger = False

        # -- Trainer ----------------------------------------------------------
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(
            max_epochs          = self._max_epochs,
            accelerator         = accelerator,
            devices             = 1,
            callbacks           = [ckpt_callback, early_stop],
            enable_progress_bar = False,
            enable_model_summary= False,
            logger              = pl_logger,
            deterministic       = True,
        )

        trainer.fit(self._model, train_loader, val_loader)

        # -- Load best checkpoint ---------------------------------------------
        # Use torch.load with weights_only=False to handle numpy scalars
        # in checkpoint (safe since we wrote this checkpoint ourselves)
        self._ckpt_path = ckpt_callback.best_model_path
        if self._ckpt_path and os.path.exists(self._ckpt_path):
            ckpt = torch.load(self._ckpt_path, weights_only=False)
            self._model.load_state_dict(ckpt["state_dict"])
            logger.info(
                f"[dmpnn] Loaded best checkpoint: val/prc="
                f"{ckpt_callback.best_model_score:.4f}"
            )
        else:
            logger.warning("[dmpnn] No checkpoint found - using final epoch weights.")

        self._model.eval()

    def predict_proba(
        self,
        X_test:      np.ndarray,
        smiles_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Score each test compound. Returns P(active) in [0, 1].

        X_test is ignored. smiles_test is required.

        Returns
        -------
        np.ndarray, shape (n_test,), float64
            Higher = more likely active = ranked higher.
        """
        if self._model is None:
            raise RuntimeError(
                "DMPNN.predict_proba() called before fit(). Call fit() first."
            )
        if smiles_test is None:
            raise ValueError("DMPNN requires smiles_test for prediction.")

        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

        # Compute RDKit descriptors for test set
        x_d_test = _compute_rdkit_descriptors(smiles_test)

        # D3: Build test datapoints with defensive error handling.
        # Track which original indices survive so scores can be scattered
        # back to full length (runner expects len(scores) == len(y_test)).
        test_data = []
        valid_idx = []
        skipped = 0
        for i in range(len(smiles_test)):
            smi = smiles_test[i]
            xd  = x_d_test[i] if x_d_test is not None else None
            try:
                if xd is not None:
                    test_data.append(MoleculeDatapoint.from_smi(smi, x_d=xd))
                else:
                    test_data.append(MoleculeDatapoint.from_smi(smi))
                valid_idx.append(i)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning(
                        f"[dmpnn] test: dropping invalid SMILES "
                        f"{str(smi)[:80]!r}: {type(e).__name__}: {e}"
                    )
        if skipped:
            logger.warning(
                f"[dmpnn] test: dropped {skipped}/{len(smiles_test)} "
                f"invalid SMILES ({100*skipped/len(smiles_test):.2f}%)"
            )

        test_dset   = MoleculeDataset(test_data)
        

        # Apply the same scaler that was fit on training descriptors (CLI behavior)
       
        test_loader = build_dataloader(
            test_dset, batch_size=self._batch_size, shuffle=False, num_workers=0
        )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(
            accelerator         = accelerator,
            devices             = 1,
            enable_progress_bar = False,
            enable_model_summary= False,
            logger              = False,
        )

        with torch.inference_mode():
            preds = trainer.predict(self._model, test_loader)

         # preds is a list of tensors, each shape (batch, 1)
        valid_scores = torch.cat(preds, dim=0).squeeze(1).cpu().numpy()

        # D3: Scatter scores back to full length. Positions with invalid
        # SMILES get -1e9 so they rank last. Using a finite value (not -inf)
        # respects BaseModel's "no NaN or Inf" contract.
        scores = np.full(len(smiles_test), -1e9, dtype=np.float64)
        scores[valid_idx] = valid_scores.astype(np.float64)
        return scores

    @property
    def is_deterministic(self) -> bool:
        """
        False - model weights depend on random seed via Xavier initialisation
        and minibatch ordering. Runner runs all 3 seeds.
        """
        return False

    @property
    def name(self) -> str:
        return "dmpnn"

    def __del__(self):
        """Clean up temp checkpoint directory on garbage collection."""
        if self._tmpdir and os.path.exists(self._tmpdir):
            try:
                shutil.rmtree(self._tmpdir)
            except Exception:
                pass
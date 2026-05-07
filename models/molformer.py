"""
models/molformer_new.py

MolFormer implementation aligned with the official IBM reference for
TopU-LBVS benchmark, while keeping screening-specific adaptations.

Pretrained weights:  ibm/MoLFormer-XL-both-10pct (HuggingFace, 1.1B-mol pretraining)
Architecture:        IBM MoLFormer custom code, loaded via trust_remote_code=True.

Changes vs. the previous molformer.py (see review for full discussion):

  Bug fixes:
    - predict_proba now applies sigmoid (was returning raw logits).
    - SMILES canonicalization cache now uses a dict {smi: canonical_smi} so the
      same SMILES set in different orders cannot return mis-aligned cached data

  Reference alignment (IBM molformer-main / finetune_pubchem_light_classification.py):
    - Pooling: attention-masked mean over last_hidden_state (not pooler_output).
    - Head: 3-layer MLP with two residual skip connections, dims [768,768,768,1].
    - Optimizer: torch_optimizer.Lamb (pure-PyTorch implementation of the
      LAMB algorithm, equivalent to apex.optimizers.FusedLAMB used in the
      IBM reference). Both groups use weight_decay=0.0 to match reference
      lines 213-216 exactly. Parameter grouping is preserved (Linear weights
      in "decay" group; LayerNorm/Embedding weights and all biases in
      "no_decay") so the structure mirrors the reference even though the
      numerical effect of grouping is currently null.
    - Single learning rate (default 3e-5, per all reference run scripts).
    - Canonicalization: isomericSmiles=False (matches reference preprocessing).

  Adaptations kept (appropriate for TopU virtual screening, not in reference):
    - BCE with pos_weight class balancing (1:10 train, 1:40 test imbalance).
    - SMILES truncation at max_length=202 (MolFormer max_position_embeddings).
    - AMP / GradScaler optional (reference relies on FusedLAMB instead).
    - ReduceLROnPlateau scheduler optional (reference uses no scheduler).
    - Best-model selection by max val PR-AUC (with early stopping). Required
      by the TopU-LBVS benchmark protocol; supersedes the IBM reference's
       no-selection / final-epoch policy.

  Style:
    - torch.amp (new API) replacing deprecated torch.cuda.amp.
    - Bare excepts replaced with logged Exception handlers.
    "We use BCE-with-logits loss with class-weighted positive examples      (pos_weight = n_neg / n_pos) rather than the IBM reference's unweighted CrossEntropy.       We tested unweighted BCE and it underperformed; the weighted variant is necessary for the imbalanced screening task."
"""

import hashlib
import logging
import pickle
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random

from models.base import BaseModel
from metrics.screening import prauc

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Model configuration
MOLFORMER_HF_ID = "ibm/MoLFormer-XL-both-10pct"
HIDDEN_SIZE = 768
MAX_TOKENS = 202   # MolFormer-XL max_position_embeddings

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "molformer_smiles"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def suppress_rdkit_logging():
    """Suppress RDKit warnings."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        yield
    finally:
        RDLogger.EnableLog('rdApp.*')


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string.

    Matches IBM reference: canonical=True, isomericSmiles=False (strips stereo).
    The encoder was pretrained on stereo-stripped SMILES, so feeding stereo
    here would push tokens out of distribution.
    """
    from rdkit import Chem
    with suppress_rdkit_logging():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def compute_smiles_hash(smiles_list: List[str]) -> str:
    """Compute MD5 hash for cache key. Order-invariant (sorts before hashing)
    so that two calls with the same SMILES set in different orders share a
    cache entry. The cache itself is a {smi: canonical_smi} dict, so ordering
    is reconstructed at lookup time, not from the cache file.
    """
    return hashlib.md5('|'.join(sorted(smiles_list)).encode()).hexdigest()[:8]


class ClassificationHead(nn.Module):
    """3-layer MLP with residual skip connections (IBM MolFormer reference Net).

    Reference: finetune_pubchem_light_classification.py:88-123, dims [768,768,768,1].
    Skip connection 1: input -> after fc1+GELU
    Skip connection 2: x_out -> input of final layer
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.gelu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu2 = nn.GELU()

        self.final = nn.Linear(hidden_size, 1)   # binary classification, single logit

        # Init: matches reference _init_weights (normal(0, 0.02), zero bias)
        for m in (self.fc1, self.fc2, self.final):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, smiles_emb: torch.Tensor) -> torch.Tensor:
        # smiles_emb: (B, H)
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.gelu1(x_out)
        x_out = x_out + smiles_emb               # skip from input

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.gelu2(z)

        logits = self.final(z + x_out)           # skip into final
        return logits.squeeze(-1)


class SMILESDataset(Dataset):
    """Dataset for SMILES strings."""

    def __init__(self, smiles: np.ndarray, labels: Optional[np.ndarray] = None):
        self.smiles = list(smiles)
        self.labels = labels
        if labels is not None:
            assert len(self.smiles) == len(labels)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Tuple:
        if self.labels is not None:
            return self.smiles[idx], float(self.labels[idx])
        return self.smiles[idx]


class MolFormer(BaseModel):
    """MolFormer fine-tuning for binary virtual screening.

    Loads ibm/MoLFormer-XL-both-10pct pretrained weights via AutoModel +
    trust_remote_code=True (which is what activates the IBM custom MoLFormer
    architecture, including rotary attention).
    """

    def __init__(
        self,
        seed: int = 2026,
        # Optimization
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        # Training
        batch_size: int = 32,
        max_epochs: int = 30,
        patience: int = 5,
        dropout: float = 0.1,
        max_length: int = MAX_TOKENS,
        grad_clip: float = 1.0,
        lr_scheduler: bool = False,   
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        use_amp: bool = True,
        use_wandb: bool = False,
        cache_smiles: bool = True,
        show_progress: bool = True,
    ):
        self._seed = seed
        self._lr = lr
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._dropout = dropout
        self._max_length = max_length
        self._grad_clip = grad_clip
        self._lr_scheduler = lr_scheduler
        self._lr_patience = lr_patience
        self._lr_factor = lr_factor
        self._use_amp = use_amp
        self._use_wandb = use_wandb
        self._cache_smiles = cache_smiles
        self._show_progress = show_progress

        self._tokenizer = None
        self._encoder = None
        self._head = None
        self._device = None
        self._scaler = None

    # ---- BaseModel interface --------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,           # unused; required by BaseModel interface
        y_train: np.ndarray,
        X_val:   np.ndarray,           # unused; required by BaseModel interface
        y_val:   np.ndarray,
        smiles_train: Optional[np.ndarray] = None,
        smiles_val:   Optional[np.ndarray] = None,
    ) -> None:
        """Train the model. X_* are required for BaseModel compat but unused
        here — MolFormer consumes SMILES strings directly via smiles_train/val.
        """
        if smiles_train is None or smiles_val is None:
            raise ValueError("MolFormer requires SMILES strings")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[molformer] Using device: {self._device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[molformer] GPU: {torch.cuda.get_device_name(0)}")

        self._set_random_seeds()

        logger.info("[molformer] Canonicalizing SMILES...")
        smiles_train_canon = self._canonicalize_smiles_batch(
            smiles_train, f"train_{self._seed}"
        )
        smiles_val_canon = self._canonicalize_smiles_batch(
            smiles_val, f"val_{self._seed}"
        )

        self._load_pretrained_model()

        if self._use_amp and self._device.type == 'cuda':
            self._scaler = GradScaler(device='cuda')
            logger.info("[molformer] Using AMP")

        train_dataset = SMILESDataset(smiles_train_canon, y_train)
        val_dataset = SMILESDataset(smiles_val_canon, y_val)

        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)

        criterion = self._create_loss_function(y_train)
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer) if self._lr_scheduler else None

        best_val_prauc = -np.inf
        best_model_state = None
        epochs_without_improvement = 0

        logger.info("[molformer] Starting training...")
        logger.info(f"[molformer] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        logger.info(f"[molformer] Batch size: {self._batch_size}")

        for epoch in range(1, self._max_epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer, epoch)
            val_prauc = self._validate(val_loader)

            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step(val_prauc)

            self._log_metrics(epoch, train_loss, val_prauc, current_lr)

            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_model_state = self._save_model_state()
                epochs_without_improvement = 0
                logger.info(f"[molformer] New best! Val PR-AUC: {best_val_prauc:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._patience:
                logger.info(f"[molformer] Early stop at epoch {epoch}")
                break

        if best_model_state:
            self._load_model_state(best_model_state)
            logger.info(f"[molformer] Restored best (PR-AUC: {best_val_prauc:.4f})")

        self._encoder.eval()
        self._head.eval()
        logger.info("[molformer] Training complete")

    def predict_proba(
        self,
        X_test: np.ndarray,                          # unused; required by BaseModel interface
        smiles_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict activity probabilities (sigmoid of logits)."""
        if self._encoder is None or smiles_test is None:
            raise RuntimeError("Model not trained or SMILES missing")

        smiles_test_canon = self._canonicalize_smiles_batch(smiles_test, "test")
        test_dataset = SMILESDataset(smiles_test_canon)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)

        self._encoder.eval()
        self._head.eval()

        all_preds = []

        with torch.no_grad():
            for batch_smiles in tqdm(
                test_loader, desc="Predicting",
                disable=not self._show_progress, leave=False
            ):
                inputs = self._tokenize_smiles(batch_smiles)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                if self._scaler:
                    with autocast(device_type='cuda'):
                        embeddings = self._encode_smiles(inputs)
                        logits = self._head(embeddings)
                else:
                    embeddings = self._encode_smiles(inputs)
                    logits = self._head(embeddings)

                # Apply sigmoid for actual probabilities (predict_proba contract)
                probs = torch.sigmoid(logits.float())
                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds).astype(np.float64)

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "molformer"

    # ---- Model construction ---------------------------------------------------

    def _load_pretrained_model(self) -> None:
        """Load the IBM MolFormer-XL custom model (rotary attention etc.)
        via AutoModel with trust_remote_code=True.
        """
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"[molformer] Loading: {MOLFORMER_HF_ID}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            MOLFORMER_HF_ID, trust_remote_code=True
        )
        logger.info(f"[molformer] Tokenizer loaded (vocab: {len(self._tokenizer)})")

        self._encoder = AutoModel.from_pretrained(
            MOLFORMER_HF_ID, trust_remote_code=True,
            embedding_dropout_prob=0.1,   # IBM finetune ref uses d_dropout=0.1; HF default is 0.2
        )
        self._encoder.to(self._device)

        n_enc = sum(p.numel() for p in self._encoder.parameters())
        logger.info(f"[molformer] Encoder loaded ({n_enc:,} parameters)")

        self._head = ClassificationHead(HIDDEN_SIZE, self._dropout)
        self._head.to(self._device)

        n_head = sum(p.numel() for p in self._head.parameters())
        logger.info(f"[molformer] Head created ({n_head:,} parameters)")
        logger.info(f"[molformer] Total: {n_enc + n_head:,} parameters")

    # ---- Loss / optimizer / scheduler ----------------------------------------

    def _create_loss_function(self, y_train: np.ndarray) -> nn.Module:
        """BCE with pos_weight class balancing.

        Adaptation for TopU screening (1:10 train imbalance) — IBM reference
        uses CrossEntropy with no rebalancing, which underweights the minority
        class for our setting.
        """
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        logger.info(
            f"[molformer] Class balance: {n_pos} active, {n_neg} inactive "
            f"(pos_weight={pos_weight:.3f})"
        )
        pos_weight_tensor = torch.tensor(
            [pos_weight], dtype=torch.float32, device=self._device
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """- Optimizer: torch_optimizer.Lamb (pure-PyTorch implementation of the
         LAMB algorithm, equivalent to apex.optimizers.FusedLAMB used in the
        IBM reference). Both groups use weight_decay=0.0 to match reference
        lines 213-216 exactly. Parameter grouping is preserved (Linear weights
        in "decay" group; LayerNorm/Embedding weights and all biases in
        "no_decay") so the structure mirrors the reference even though the
        numerical effect of grouping is currently null.
        """
        decay_params, no_decay_params = [], []
        decay_names, no_decay_names = set(), set()

        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for prefix, model in [("encoder", self._encoder), ("head", self._head)]:
            for module_name, module in model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    full_name = (
                        f"{prefix}.{module_name}.{param_name}"
                        if module_name else f"{prefix}.{param_name}"
                    )
                    if full_name in decay_names or full_name in no_decay_names:
                        continue   # already counted under a parent path

                    if param_name.endswith('bias'):
                        no_decay_params.append(param)
                        no_decay_names.add(full_name)
                    elif isinstance(module, whitelist):
                        decay_params.append(param)
                        decay_names.add(full_name)
                    elif isinstance(module, blacklist):
                        no_decay_params.append(param)
                        no_decay_names.add(full_name)
                    else:
                        # Catch-all: rotary buffers, raw embeddings, etc. — no decay
                        no_decay_params.append(param)
                        no_decay_names.add(full_name)

        overlap = decay_names & no_decay_names

        assert not overlap, f"Params in both groups: {overlap}"
        optim_groups = [
          {"params": decay_params,    "weight_decay": 0.0},
          {"params": no_decay_params, "weight_decay": 0.0},
        ]
        import torch_optimizer as topt
        optimizer = topt.Lamb(
           optim_groups, lr=self._lr, betas=(0.9, 0.99), eps=1e-6
        )

        logger.info(
           f"[molformer] Optimizer: Lamb(lr={self._lr:.2e}, "
           f"decay={len(decay_params)} params, no_decay={len(no_decay_params)} params)"
        )
        return optimizer

    def _create_scheduler(self, optimizer):
        """ReduceLROnPlateau on val PR-AUC.

        Note: IBM reference uses no scheduler; this is a screening adaptation.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=self._lr_factor,
            patience=self._lr_patience
        )
        logger.info(
            f"[molformer] Scheduler: ReduceLROnPlateau(patience={self._lr_patience})"
        )
        return scheduler

    # ---- Data loading ---------------------------------------------------------

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create dataloader with SMILES tokenization in collate_fn."""
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self._seed)

        if isinstance(dataset, SMILESDataset) and dataset.labels is not None:
            collate_fn = lambda b: (
                self._tokenize_smiles([x[0] for x in b]),
                torch.tensor([x[1] for x in b], dtype=torch.float32)
            )
        else:
            collate_fn = lambda b: b

        return DataLoader(
            dataset, batch_size=self._batch_size, shuffle=shuffle,
            generator=generator, num_workers=0,
            pin_memory=(self._device.type == 'cuda'),
            collate_fn=collate_fn
        )

    # ---- Train / validate -----------------------------------------------------

    def _train_epoch(self, dataloader, criterion, optimizer, epoch) -> float:
       
        self._encoder.train()
        self._head.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{self._max_epochs}",
            disable=not self._show_progress, leave=False
        )

        

        for batch_idx, (inputs, labels) in enumerate(pbar):
            optimizer.zero_grad()
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            labels = labels.to(self._device)

            if self._scaler:
                with autocast(device_type='cuda'):
                    embeddings = self._encode_smiles(inputs)
                    logits = self._head(embeddings)
                    loss = criterion(logits, labels)
                self._scaler.scale(loss).backward()
            else:
                embeddings = self._encode_smiles(inputs)
                logits = self._head(embeddings)
                loss = criterion(logits, labels)
                loss.backward()

            
            self._optimizer_step(optimizer)
            total_loss += loss.item()
            n_batches += 1

            if self._show_progress:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(n_batches, 1)

    def _optimizer_step(self, optimizer) -> None:
        """Unscale (if AMP), grad-clip, step, zero_grad."""
        if self._grad_clip > 0:
            if self._scaler:
                self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self._encoder.parameters()) + list(self._head.parameters()),
                max_norm=self._grad_clip
            )

        if self._scaler:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    def _validate(self, dataloader) -> float:
        """Compute val PR-AUC."""
        self._encoder.eval()
        self._head.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                if self._scaler:
                    with autocast(device_type='cuda'):
                        embeddings = self._encode_smiles(inputs)
                        logits = self._head(embeddings)
                else:
                    embeddings = self._encode_smiles(inputs)
                    logits = self._head(embeddings)

                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(labels.numpy())

        if not all_logits:
            return 0.0

        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_labels).astype(int)

        if np.isnan(logits).any():
            logger.error(
                f"[molformer] NaN in predictions: "
                f"{np.isnan(logits).sum()}/{len(logits)} — "
                f"excluding epoch from best-model selection"
            )
            return -np.inf

        if len(np.unique(labels)) < 2:
            return 0.0

        return prauc(labels, logits)

    # ---- Checkpointing --------------------------------------------------------

    def _save_model_state(self) -> Dict:
        """Save best-model state dict to CPU memory.

        Note: copies ~180MB at fp32; happens once per "best" epoch. For
        memory-constrained runs, consider saving to disk and reloading.
        """
        return {
            'encoder': {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()},
            'head':    {k: v.cpu().clone() for k, v in self._head.state_dict().items()},
        }

    def _load_model_state(self, state_dict: Dict) -> None:
        """Restore best-model state."""
        self._encoder.load_state_dict(state_dict['encoder'])
        self._head.load_state_dict(state_dict['head'])
        self._encoder.to(self._device)
        self._head.to(self._device)

    # ---- SMILES processing ----------------------------------------------------

    def _canonicalize_smiles_batch(
        self, smiles_list: np.ndarray, cache_key: str
    ) -> np.ndarray:
        """Canonicalize SMILES with order-safe caching.

        The cache stores a {smi: canonical_smi} dict, NOT an ordered array. This
        means the same SMILES set in any order will hit the cache correctly,
        and the returned array is always rebuilt in the input order from the
        dict. The previous implementation cached the canonical array in input
        order, which silently mis-aligned with labels if the input order ever
        changed.
        """
        if not self._cache_smiles:
            return np.array([canonicalize_smiles(s) for s in smiles_list])

        smiles_hash = compute_smiles_hash(list(smiles_list))
        cache_path = CACHE_DIR / f"{cache_key}_{smiles_hash}.pkl"

        canonical_map: Optional[Dict[str, str]] = None
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    canonical_map = pickle.load(f)
                # Defensive: confirm cache covers everything we need
                if not all(s in canonical_map for s in smiles_list):
                    logger.warning(
                        f"[molformer] Cache {cache_path.name} incomplete — recomputing"
                    )
                    canonical_map = None
                else:
                    logger.info(f"[molformer] Cached SMILES loaded: {cache_path.name}")
            except Exception as e:
                logger.warning(f"[molformer] Cache load failed ({e}) — recomputing")
                canonical_map = None

        if canonical_map is None:
            unique = list(set(smiles_list))
            logger.info(
                f"[molformer] Canonicalizing {len(unique)} unique SMILES "
                f"({len(smiles_list)} total)..."
            )
            canonical_map = {s: canonicalize_smiles(s) for s in unique}
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(canonical_map, f)
            except Exception as e:
                logger.warning(f"[molformer] Cache save failed: {e}")

        # Always rebuild in input order from the dict
        return np.array([canonical_map[s] for s in smiles_list])

    def _tokenize_smiles(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of SMILES."""
        return self._tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=self._max_length, return_tensors='pt',
            return_token_type_ids=False,
        )

    def _encode_smiles(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode SMILES via attention-masked mean pooling.

        Reference: finetune_pubchem_light_classification.py:238-241.
            input_mask_expanded = mask.unsqueeze(-1).expand(emb.size()).float()
            sum_emb = (emb * input_mask_expanded).sum(1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_emb / sum_mask

        Uses last_hidden_state (real token reps), not pooler_output (a tanh
        layer over the BOS token that wasn't trained against any downstream
        objective during MLM pretraining).
        """
        outputs = self._encoder(**inputs)
        hidden = outputs.last_hidden_state                  # (B, T, H)
        mask = inputs['attention_mask'].unsqueeze(-1).float()   # (B, T, 1)
        sum_emb = (hidden * mask).sum(dim=1)                # (B, H)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
        return sum_emb / sum_mask

    # ---- Misc -----------------------------------------------------------------

    def _set_random_seeds(self) -> None:
        """Set torch / numpy / cuda / Python random seeds.

        Mirrors pytorch_lightning.seed.seed_everything used by the IBM
        reference at line 578 of finetune_pubchem_light_classification.py.
        """
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
        logger.info(f"[molformer] Seed: {self._seed}")

    def _log_metrics(self, epoch, train_loss, val_prauc, lr) -> None:
        """Log metrics to console and (optionally) wandb."""
        logger.info(
            f"[molformer] Epoch {epoch}/{self._max_epochs} | "
            f"Loss: {train_loss:.4f} | Val PR-AUC: {val_prauc:.4f} | LR: {lr:.2e}"
        )

        if self._use_wandb:
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        'epoch':         epoch,
                        'train_loss':    train_loss,
                        'val_prauc':     val_prauc,
                        'learning_rate': lr,
                    })
            except Exception as e:
                logger.debug(f"[molformer] wandb log failed: {e}")

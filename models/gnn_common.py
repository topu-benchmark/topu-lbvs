from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool

from models.base import BaseModel
from utils.mol_utils import NODE_FEAT_DIM, smiles_list_to_graphs




def _collate(batch: List[Data]) -> Batch:
    return Batch.from_data_list(batch)


class MolGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graphs: List[Optional[Data]],
        labels: np.ndarray,
        fps: Optional[np.ndarray] = None,
    ):
        valid_records = []
        for idx, graph in enumerate(graphs):
            if graph is None:
                continue
            record = {"graph": graph, "label": float(labels[idx])}
            if fps is not None:
                record["fp"] = fps[idx].astype(np.float32, copy=False)
            valid_records.append(record)

        if not valid_records:
            raise ValueError("No valid graphs in dataset.")

        self._records = valid_records
        self._has_fp = fps is not None

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Data:
        item = self._records[idx]
        graph = deepcopy(item["graph"])
        graph.y = torch.tensor([item["label"]], dtype=torch.float32)
        if self._has_fp:
            graph.fp = torch.from_numpy(item["fp"])
        return graph


class GraphEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        embedding_dim: int = 32,
        heads: int = 4,
    ):
        super().__init__()
        self.backbone = backbone.lower()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(self._build_conv(hidden_dim, heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, embedding_dim)

    def _build_conv(self, hidden_dim: int, heads: int) -> nn.Module:
        if self.backbone == "gcn":
            return GCNConv(hidden_dim, hidden_dim)
        if self.backbone == "gat":
            return GATConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=self.dropout,
            )
        if self.backbone == "gin":
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            return GINConv(mlp, train_eps=True)
        if self.backbone == "gps":
            # Imported lazily so GCN/GAT/GIN still work in envs without GPSConv.
            from torch_geometric.nn import GPSConv

            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINConv(local_mlp, train_eps=True)
            return GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=self.dropout,
            )
        raise ValueError(f"Unsupported backbone: {self.backbone}")

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.input_proj(batch.x)
        x = F.relu(x)

        for conv, bn in zip(self.convs, self.bns):
            if self.backbone == "gps":
                x = conv(x, batch.edge_index, batch=batch.batch)
            else:
                x = conv(x, batch.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_add_pool(x, batch.batch)
        return self.out_proj(pooled)


class GraphOnlyHead(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        return self.head(graph_emb).squeeze(-1)


class FpProjector(nn.Module):
    def __init__(self, fp_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, fp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fp_hidden_dim, 32),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


class GNNGraphClassifier(nn.Module):
    def __init__(self, encoder: GraphEncoder, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.graph_head = GraphOnlyHead(dropout=dropout)

    def forward(self, batch: Batch) -> torch.Tensor:
        graph_emb = self.encoder(batch)
        return self.graph_head(graph_emb)


class GNNLateFusionClassifier(nn.Module):
    def __init__(
        self,
        encoder: GraphEncoder,
        fp_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.fp_proj = FpProjector(fp_hidden_dim=fp_hidden_dim, dropout=dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(64, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        graph_emb = self.encoder(batch)
        fp = batch.fp.float()
        # PyG may collate custom tensor attributes as a flattened 1D vector.
        # Restore per-molecule FP rows expected by the projector.
        fp = fp.reshape(-1, 2048)
        fp_emb = self.fp_proj(fp)
        fused = torch.cat([graph_emb, fp_emb], dim=-1)
        return self.fusion_head(fused).squeeze(-1)


class BaseGNNModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        backbone: str,
        with_fp: bool,
        seed: int = 2026,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        fp_hidden_dim: int = 256,
        fusion_hidden_dim: int = 128,
        use_wandb: bool = True,
    ):
        self._name = model_name
        self._backbone = backbone
        self._with_fp = with_fp

        self._seed = seed
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dropout = dropout
        self._heads = heads
        self._lr = lr
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr_patience = lr_patience
        self._lr_factor = lr_factor
        self._fp_hidden_dim = fp_hidden_dim
        self._fusion_hidden_dim = fusion_hidden_dim
        self._use_wandb = use_wandb

        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        smiles_train: Optional[np.ndarray] = None,
        smiles_val: Optional[np.ndarray] = None,
    ) -> None:
        if smiles_train is None or smiles_val is None:
            raise ValueError(f"{self._name} requires SMILES in fit().")

        self._set_seed()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fp_train = X_train.astype(np.float32, copy=False) if self._with_fp else None
        fp_val = X_val.astype(np.float32, copy=False) if self._with_fp else None

        graph_start = time.time()
        train_graphs = smiles_list_to_graphs(smiles_train)
        val_graphs = smiles_list_to_graphs(smiles_val)

        train_dataset = MolGraphDataset(train_graphs, y_train, fps=fp_train)
        val_dataset = MolGraphDataset(val_graphs, y_val, fps=fp_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        encoder = GraphEncoder(
            backbone=self._backbone,
            hidden_dim=self._hidden_dim,
            n_layers=self._n_layers,
            dropout=self._dropout,
            heads=self._heads,
            embedding_dim=32,
        )
        if self._with_fp:
            self._model = GNNLateFusionClassifier(
                encoder=encoder,
                fp_hidden_dim=self._fp_hidden_dim,
                fusion_hidden_dim=self._fusion_hidden_dim,
                dropout=self._dropout,
            ).to(self._device)
        else:
            self._model = GNNGraphClassifier(
                encoder=encoder,
                dropout=self._dropout,
            ).to(self._device)

        n_active = int(np.sum(y_train))
        n_decoy = int(len(y_train) - n_active)
        pos_weight = n_decoy / max(n_active, 1)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self._device, dtype=torch.float32)
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self._lr_factor,
            patience=self._lr_patience,
        )

        best_val_prauc = -1.0
        best_state = None
        stale_epochs = 0
        total_train_batches = 0
        epochs_ran = 0
        train_start = time.time()

        for epoch in range(1, self._max_epochs + 1):
            epochs_ran = epoch
            self._model.train()
            loss_sum = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                logits = self._model(batch)
                loss = criterion(logits, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
                n_batches += 1
                total_train_batches += 1

            train_loss = loss_sum / max(n_batches, 1)
            val_prauc = self._evaluate_prauc(val_loader)
            scheduler.step(val_prauc)

            if self._use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_prauc": val_prauc,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

    def predict_proba(
        self,
        X_test: np.ndarray,
        smiles_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._model is None or self._device is None:
            raise RuntimeError(f"{self._name}.predict_proba called before fit().")
        if smiles_test is None:
            raise ValueError(f"{self._name} requires smiles_test in predict_proba().")

        test_graphs = smiles_list_to_graphs(smiles_test)
        valid_idx = [i for i, g in enumerate(test_graphs) if g is not None]
        scores = np.full(len(smiles_test), -1e9, dtype=np.float64)
        if not valid_idx:
            return scores

        labels = np.zeros(len(valid_idx), dtype=np.int32)
        valid_graphs = [test_graphs[i] for i in valid_idx]
        fp_valid = None
        if self._with_fp:
            fp_valid = X_test.astype(np.float32, copy=False)[valid_idx]

        dataset = MolGraphDataset(valid_graphs, labels, fps=fp_valid)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        preds = []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                logits = self._model(batch)
                preds.append(logits.detach().cpu().numpy())

        valid_scores = np.concatenate(preds).astype(np.float64)
        for j, original_idx in enumerate(valid_idx):
            scores[original_idx] = valid_scores[j]
        return scores

    def _evaluate_prauc(self, loader: torch.utils.data.DataLoader) -> float:
        self._model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                logits = self._model(batch)
                y_score.append(logits.detach().cpu().numpy())
                y_true.append(batch.y.view(-1).detach().cpu().numpy().astype(np.int32))

        y_true_arr = np.concatenate(y_true)
        y_score_arr = np.concatenate(y_score)
        if np.unique(y_true_arr).shape[0] < 2:
            return 0.0
        return float(average_precision_score(y_true_arr, y_score_arr))

    def _set_seed(self) -> None:
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name

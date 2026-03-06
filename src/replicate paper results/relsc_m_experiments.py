import os
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from torch_geometric.loader import DataLoader
import random
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, GraphConv, TransformerConv, Linear
from relsc_m import RelSCM
from typing import List

DEVICE = "cpu"         # set to "cuda" if available
HIDDEN = 30            # as in the paper
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.01
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
SEEDS = (0, 1, 2, 3, 4)

# metrics used everywhere
METRICS = ["MAE", "RMSE", "MAPE", "RHO", "MRE"]
# file where incremental results are saved
RESULTS_PATH = "all_results_relsc_m.pt"


# ====================== DATASETS ======================

relsc_m_dataset3 = RelSCM(project_name="H2")
relsc_m_dataset1 = RelSCM(project_name="rdf")
relsc_m_dataset2 = RelSCM(project_name="dubbo")
relsc_m_dataset4 = RelSCM(project_name="hadoop")
relsc_m_dataset5 = RelSCM(project_name="systemds")
relsc_m_dataset6 = RelSCM(project_name="ossbuilds")

# ------- DATASETS (column order as in the tables) -------
datasets = {
    "Hadoop":    relsc_m_dataset4,
    "RDF4J":     relsc_m_dataset1,
    "SystemDS":  relsc_m_dataset5,
    "H2":        relsc_m_dataset3,
    "Dubbo":     relsc_m_dataset2,
    "OssBuilds": relsc_m_dataset6,
}


# ====================== UTILITY: POOLING & METADATA ======================

def _mean_max_pool_by_type(x_dict, batch_dict, node_types: List[str],
                           hidden_channels: int, device: str):
    """Per-graph pooling: for each node type compute mean and max, then concat across all types (mean||max)."""
    if len(batch_dict) == 0:
        # fallback: a single graph
        parts = [torch.zeros(2 * hidden_channels, device=device) for _ in node_types]
        return torch.stack([torch.cat(parts, dim=0)], dim=0)

    B = int(batch_dict[next(iter(batch_dict))].max().item() + 1)
    graph_reps = []
    for b in range(B):
        chunks = []
        for ntype in node_types:
            if ntype in x_dict:
                mask = (batch_dict[ntype] == b)
                if mask.any():
                    x = x_dict[ntype][mask]
                    mean_pool = x.mean(dim=0)
                    max_pool = x.max(dim=0).values
                    chunks.append(torch.cat([mean_pool, max_pool], dim=0).to(device))
                else:
                    chunks.append(torch.zeros(2 * hidden_channels, device=device))
            else:
                chunks.append(torch.zeros(2 * hidden_channels, device=device))
        graph_reps.append(torch.cat(chunks, dim=0))
    return torch.stack(graph_reps, dim=0)


def extract_metadata(dataset):
    """Collect sorted node and edge types from a heterogeneous dataset."""
    node_types = set()
    edge_types = set()
    for data in dataset:
        node_types.update(list(data.x_dict.keys()))
        edge_types.update(list(data.edge_index_dict.keys()))
    node_types = list(sorted(node_types))
    edge_types = list(sorted(edge_types))
    return (node_types, edge_types)


# ====================== MODELS ======================

class HeteroGNN_GraphConv(nn.Module):
    """
    Heterogeneous GNN with GraphConv per relation.
    - Initial projection per node type (-1 -> hidden)
    - GraphConv-based HeteroConv over each edge type
    - Per-type BatchNorm + LeakyReLU + Dropout
    - Mean+max pooling per node type, concatenated across all types
    """
    def __init__(self, dataset, hidden_channels=30, out_channels=1,
                 num_layers=2, device="cpu", dropout=0.2):
        super().__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.node_types, self.edge_types = extract_metadata(dataset)
        self.dropout = dropout

        # Per-node-type projection + per-type BatchNorm
        self.proj = nn.ModuleDict({nt: Linear(-1, hidden_channels) for nt in self.node_types})
        self.bn   = nn.ModuleDict({nt: nn.BatchNorm1d(hidden_channels) for nt in self.node_types})

        # Heterogeneous stack
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: GraphConv((-1, -1), hidden_channels) for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Readout: two poolings (mean+max) per node type
        readout_dim = 2 * hidden_channels * len(self.node_types)
        self.mlp = Linear(readout_dim, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # --- Projection + safe BN + activation ---
        new_x_dict = {}
        for k, x in x_dict.items():
            x = x.float()
            if x.size(0) == 0:
                # no nodes of this type in the batch
                new_x_dict[k] = x
                continue

            h = self.proj[k](x)
            # In training, apply BN only if batch_size > 1; in eval always (running stats)
            if self.training:
                if h.size(0) > 1:
                    h = self.bn[k](h)
            else:
                h = self.bn[k](h)

            new_x_dict[k] = h

        x_dict = {k: F.leaky_relu(v) for k, v in new_x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # --- Heterogeneous convolutions + safe BN ---
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            new_x_dict = {}
            for k, x in x_dict.items():
                if x.size(0) == 0:
                    new_x_dict[k] = x
                    continue

                if self.training:
                    if x.size(0) > 1:
                        x = self.bn[k](x)
                else:
                    x = self.bn[k](x)

                new_x_dict[k] = x

            x_dict = {k: F.leaky_relu(v) for k, v in new_x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # --- Mean+max pooling per type and concatenation ---
        x = _mean_max_pool_by_type(x_dict, batch_dict, self.node_types,
                                   self.hidden_channels, self.device)
        x = self.mlp(x)
        return self.lin(x)


class HeteroGNN_Transformer(nn.Module):
    """
    Heterogeneous GNN with TransformerConv (multi-head) per relation.
    Parameters: heads=2, concat=False (dimension stays hidden), beta=True.
    Remaining pipeline is identical to HeteroGNN_GraphConv.
    """
    def __init__(self, dataset, hidden_channels=30, out_channels=1,
                 num_layers=2, device="cpu", dropout=0.2):
        super().__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.node_types, self.edge_types = extract_metadata(dataset)
        self.dropout = dropout

        self.proj = nn.ModuleDict({nt: Linear(-1, hidden_channels) for nt in self.node_types})
        self.bn   = nn.ModuleDict({nt: nn.BatchNorm1d(hidden_channels) for nt in self.node_types})

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=2,
                    concat=False,
                    beta=True
                ) for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        readout_dim = 2 * hidden_channels * len(self.node_types)
        self.mlp = Linear(readout_dim, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # --- Projection + safe BN + activation ---
        new_x_dict = {}
        for k, x in x_dict.items():
            x = x.float()
            if x.size(0) == 0:
                new_x_dict[k] = x
                continue

            h = self.proj[k](x)
            if self.training:
                if h.size(0) > 1:
                    h = self.bn[k](h)
            else:
                h = self.bn[k](h)

            new_x_dict[k] = h

        x_dict = {k: F.leaky_relu(v) for k, v in new_x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # --- Heterogeneous convolutions + safe BN ---
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            new_x_dict = {}
            for k, x in x_dict.items():
                if x.size(0) == 0:
                    new_x_dict[k] = x
                    continue

                if self.training:
                    if x.size(0) > 1:
                        x = self.bn[k](x)
                else:
                    x = self.bn[k](x)

                new_x_dict[k] = x

            x_dict = {k: F.leaky_relu(v) for k, v in new_x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # --- Pooling + final MLP ---
        x = _mean_max_pool_by_type(x_dict, batch_dict, self.node_types,
                                   self.hidden_channels, self.device)
        x = self.mlp(x)
        return self.lin(x)


# ------- MODEL BUILDERS: ONLY THE TWO NEW CLASSES -------
def make_graphconv(ds):
    return HeteroGNN_GraphConv(ds, hidden_channels=HIDDEN, out_channels=1,
                               num_layers=NUM_LAYERS, device=DEVICE, dropout=DROPOUT)


def make_transformer(ds):
    return HeteroGNN_Transformer(ds, hidden_channels=HIDDEN, out_channels=1,
                                 num_layers=NUM_LAYERS, device=DEVICE, dropout=DROPOUT)


model_registry = {
    "HeteroGraphConv":   make_graphconv,
    "HeteroTransformer": make_transformer,
}


# ====================== METRICS ======================

def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mre(y, p, eps=1e-12):
    return float(np.max(np.abs((y - p) / (np.abs(y) + eps))))


def _mape(y, p, eps=1e-12):
    val = np.mean(np.abs((y - p) / (np.abs(y) + eps)))
    return float(val) if np.isfinite(val) else None


def _spearman(y, p):
    def _rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r

    rx, ry = _rank(y), _rank(p)
    cov = np.cov(rx, ry, bias=True)[0, 1]
    sx, sy = np.std(rx), np.std(ry)
    return float(cov / (sx * sy)) if (sx > 0 and sy > 0) else 0.0


@torch.no_grad()
def _eval_model(model, loader, device=DEVICE):
    """Run inference and compute all regression metrics on a loader split."""
    model.eval()
    preds, reals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict).view(-1)
        preds.append(out.detach().cpu().numpy())
        reals.append(data.y.view(-1).detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(reals, axis=0)
    return dict(
        MAE=_mae(y_true, y_pred),
        RMSE=_rmse(y_true, y_pred),
        MAPE=_mape(y_true, y_pred),
        RHO=_spearman(y_true, y_pred),
        MRE=_mre(y_true, y_pred),
    )


# ====================== TRAINING & EARLY STOPPING ======================

def _fix_seed(seed):
    """Fix Python/NumPy/PyTorch seeds for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.p = patience
        self.best = float('inf')
        self.cnt = 0
        self.state = None

    def step(self, val, model):
        if val < self.best - 1e-12:
            self.best = val
            self.cnt = 0
            self.state = deepcopy(model.state_dict())
            return False
        self.cnt += 1
        return self.cnt >= self.p


def _train_val(model, train_loader, val_loader, device=DEVICE):
    """Train with MAE loss and stop early based on validation MAE."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss()
    stopper = _EarlyStop(patience=PATIENCE)
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict,
                        batch.batch_dict).view(-1)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # validate on MAE
        val_metrics = _eval_model(model, val_loader, device=device)
        if stopper.step(val_metrics["MAE"], model):
            break
    if stopper.state is not None:
        model.load_state_dict(stopper.state)


# ====================== TABLES ======================

def _format(values, digits=2, dash_for_none=False):
    """Format metric lists as mean(std) while handling missing/invalid values."""
    if values is None:
        return "–" if dash_for_none else "-"
    arr = [
        v for v in values
        if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
    ]
    if len(arr) == 0:
        return "–" if dash_for_none else "-"
    return f"{np.mean(arr):.{digits}f}(±{np.std(arr):.{digits}f})"


def _build_tables(results):
    """Build publication-style metric tables from aggregated run results."""
    order = ["Hadoop", "RDF4J", "SystemDS", "H2", "Dubbo", "OssBuilds"]

    def mk(metric, split, digits=2, dash=False):
        rows = []
        for model in results.keys():
            row = {"Model": model}
            for ds in order:
                vals = results[model][ds][split][metric]
                row[ds] = _format(vals, digits=digits, dash_for_none=dash)
            rows.append(row)
        return pd.DataFrame(rows)[["Model"] + order]

    return {
        "Table3_TEST_MAE":  mk("MAE", "TEST", digits=2, dash=False),
        "Table5_VAL_MAE":   mk("MAE", "VAL",  digits=2, dash=False),
        "Table6_TEST_RMSE": mk("RMSE", "TEST", digits=2, dash=False),
        "Table7_VAL_RMSE":  mk("RMSE", "VAL",  digits=2, dash=False),
        "Table8_TEST_MAPE": mk("MAPE", "TEST", digits=2, dash=True),
        "Table9_VAL_MAPE":  mk("MAPE", "VAL",  digits=2, dash=True),
        "Table10_TEST_RHO": mk("RHO", "TEST", digits=2, dash=False),
        "Table11_VAL_RHO":  mk("RHO", "VAL",  digits=2, dash=False),
        "Table12_TEST_MRE": mk("MRE", "TEST", digits=0, dash=False),
        "Table13_VAL_MRE":  mk("MRE", "VAL",  digits=0, dash=False),
    }


def _init_result_entry(all_results, model_name, ds_name):
    """Ensure the empty structure exists for (model, dataset)."""
    if model_name not in all_results:
        all_results[model_name] = {}
    if ds_name not in all_results[model_name]:
        all_results[model_name][ds_name] = {
            "TEST": {m: [] for m in METRICS},
            "VAL":  {m: [] for m in METRICS},
        }


# ====================== RUN WITH CHECKPOINT ======================

print("Starting RelSCM experiments...")

# 1) Load partial results if present
if os.path.exists(RESULTS_PATH):
    print(f"Loading partial results from '{RESULTS_PATH}'")
    all_results = torch.load(RESULTS_PATH)
else:
    all_results = {}

# 2) Loop over models and datasets, resuming per seed
for model_name, ctor in model_registry.items():
    for ds_name, ds in datasets.items():
        print(f"\n=== {model_name} / {ds_name} ===")

        # Initialize the structure for this (model, dataset) if needed
        _init_result_entry(all_results, model_name, ds_name)
        entry = all_results[model_name][ds_name]

        # DataLoaders created once for this dataset
        tr = DataLoader(ds.load_split("train"), batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=ds.hetero_collate)
        va = DataLoader(ds.load_split("val"), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=ds.hetero_collate)
        te = DataLoader(ds.load_split("test"), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=ds.hetero_collate)

        # How many seeds have already been completed?
        # (use TEST MAE list length as reference)
        done_seeds = len(entry["TEST"]["MAE"])
        if done_seeds > 0:
            print(f"  -> found {done_seeds} completed seeds, resuming from seed index {done_seeds}")

        # Continue from missing seeds
        for seed_idx in range(done_seeds, len(SEEDS)):
            s = SEEDS[seed_idx]
            print(f"  Seed {s}")
            _fix_seed(s)

            model = ctor(ds).to(DEVICE)
            _train_val(model, tr, va, device=DEVICE)

            te_metrics = _eval_model(model, te, device=DEVICE)
            va_metrics = _eval_model(model, va, device=DEVICE)

            for m in METRICS:
                entry["TEST"][m].append(te_metrics[m])
                entry["VAL"][m].append(va_metrics[m])

            # Save partial results immediately
            torch.save(all_results, RESULTS_PATH)
            print(f"  -> saved partial state to '{RESULTS_PATH}'")

# 3) Build tables once all seeds are completed
tables = _build_tables(all_results)

# ------- MAIN OUTPUT (Table 3) -------
print("=== Table 3 (TEST MAE) — RelSC-M / RelSCM — New models only ===")
print(tables["Table3_TEST_MAE"])

# ------- SAVE RESULTS TO CSV -------
for name, df in tables.items():
    csv_name = f"{name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved table '{name}' to '{csv_name}'")

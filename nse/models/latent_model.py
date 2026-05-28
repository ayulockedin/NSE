"""Latent Transition Model — Layer 2 (PyTorch GNN ensemble).

Architecture (blueprint section 6.2):
    CPG-lite node features -> GNN encoder (GAT, 2 layers, mean readout)
    patch features -> patch encoder (MLP)
    concat -> shared MLP -> M independent heads -> [p_t, r_long] per head

Ensemble variance across the M heads' p_t gives the epistemic uncertainty ``u``.

torch / torch_geometric are imported lazily. If unavailable (e.g. before the
heavy install finishes), :class:`LatentEnsemble` still answers via a
deterministic heuristic so the orchestrator runs end-to-end. The neural path is
used automatically once the deps are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pathlib import Path

from nse.config import ROOT, SETTINGS
from nse.orchestrator.schemas import LatentPrediction, PlannerBranch

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

try:
    from torch_geometric.nn import GATConv, global_mean_pool  # type: ignore

    _PYG = True
except ImportError:  # pragma: no cover
    _PYG = False


PATCH_FEATURE_DIM = 6  # [files, added, removed, tokens_changed, complexity_delta, depth]


# ───────────────────────────── neural model ─────────────────────────────

if _TORCH:

    class GNNEncoder(nn.Module):
        def __init__(self, in_dim: int, hidden: int) -> None:
            super().__init__()
            self._pyg = _PYG
            if _PYG:
                self.conv1 = GATConv(in_dim, hidden)
                self.conv2 = GATConv(hidden, hidden)
            else:
                # Fallback when PyG missing: simple node-MLP + mean pool.
                self.lin1 = nn.Linear(in_dim, hidden)
                self.lin2 = nn.Linear(hidden, hidden)

        def forward(self, x, edge_index, batch):  # type: ignore[no-untyped-def]
            if self._pyg:
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                return global_mean_pool(x, batch)
            x = self.lin1(x).relu()
            x = self.lin2(x).relu()
            # Batch-aware mean pool (one row per graph). A naive global mean
            # would collapse the whole batch to a single row and corrupt
            # batched training when torch_geometric is absent.
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
            pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
            pooled.index_add_(0, batch, x)
            counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1)
            return pooled / counts.unsqueeze(1).to(x.dtype)

    class LatentModel(nn.Module):
        def __init__(
            self,
            emb_dim: int = 32,
            patch_dim: int = PATCH_FEATURE_DIM,
            hidden: int = 256,
            M: int = SETTINGS.hp.M,
        ) -> None:
            super().__init__()
            self.gnn = GNNEncoder(emb_dim, hidden)
            self.patch_encoder = nn.Sequential(
                nn.Linear(patch_dim, hidden), nn.ReLU()
            )
            self.shared_mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU()
            )
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))
                    for _ in range(M)
                ]
            )

        def forward(self, g_x, edge_index, batch, patch_emb):  # type: ignore[no-untyped-def]
            g_emb = self.gnn(g_x, edge_index, batch)
            p_emb = self.patch_encoder(patch_emb)
            joint = torch.cat([g_emb, p_emb], dim=-1)
            h = self.shared_mlp(joint)
            # each head -> sigmoid([p_t, r_long]); shape (M, batch, 2)
            return [torch.sigmoid(head(h)) for head in self.heads]


@dataclass
class _HeuristicWeights:
    """Deterministic stand-in until the ensemble is trained."""

    base: float = 0.6
    complexity_penalty: float = 0.4


class LatentEnsemble:
    """Inference wrapper. Uses the neural model when available, else heuristic."""

    def __init__(self, model: Optional["LatentModel"] = None) -> None:  # type: ignore[name-defined]
        self.model = model
        self.M = SETTINGS.hp.M
        self._h = _HeuristicWeights()

    # ── feature extraction ────────────────────────────────────────────
    @staticmethod
    def patch_features(branch: PlannerBranch) -> list[float]:
        diff = branch.patch_preview or ""
        added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        tokens_changed = len(diff.split())
        return [
            float(len(branch.edited_files)),
            float(added),
            float(removed),
            float(tokens_changed),
            branch.expected_complexity,
            0.0,  # branch depth placeholder
        ]

    # ── inference ──────────────────────────────────────────────────────
    def predict(self, branch: PlannerBranch) -> LatentPrediction:
        """Score a planner branch. Delegates to :meth:`predict_from_features`."""
        feats = self.patch_features(branch)
        return self.predict_from_features(feats, branch_id=branch.branch_id)

    def predict_from_features(
        self, feats: list[float], branch_id: str = ""
    ) -> LatentPrediction:
        """Score a raw 6-dim patch-feature vector.

        Decoupled from ``PlannerBranch`` so the eval harness and training loop
        can score serialised feature vectors directly. ``feats[4]`` is the
        branch's ``expected_complexity``.
        """
        if _TORCH and self.model is not None:
            with torch.no_grad():  # type: ignore[union-attr]
                # Minimal single-node graph: feature vector from complexity.
                g_x = torch.zeros((1, 32))
                g_x[0, 0] = feats[4]
                edge_index = torch.empty((2, 0), dtype=torch.long)
                batch = torch.zeros(1, dtype=torch.long)
                patch_emb = torch.tensor([feats], dtype=torch.float32)
                outs = self.model(g_x, edge_index, batch, patch_emb)
                p_ts = [float(o[0, 0]) for o in outs]
                r_longs = [float(o[0, 1]) for o in outs]
            p_t_latent = sum(p_ts) / len(p_ts)
            r_long = sum(r_longs) / len(r_longs)
            mean = p_t_latent
            u = sum((p - mean) ** 2 for p in p_ts) / len(p_ts)
            return LatentPrediction(
                branch_id=branch_id,
                p_t_latent=p_t_latent,
                r_long=r_long,
                u=u,
                per_head_p_t=p_ts,
            )

        return self._heuristic_predict(feats, branch_id)

    def _heuristic_predict(self, feats: list[float], branch_id: str = "") -> LatentPrediction:
        """Deterministic, bounded fallback. Encodes 'simpler == safer'."""
        complexity = feats[4]
        center = self._h.base - self._h.complexity_penalty * complexity
        center = min(0.95, max(0.05, center))
        # Synthesize M slightly perturbed heads so u is non-degenerate and
        # grows with complexity (more uncertain on harder changes).
        spread = 0.02 + 0.10 * complexity
        per_head = [
            min(0.99, max(0.01, center + spread * (i - (self.M - 1) / 2)))
            for i in range(self.M)
        ]
        mean = sum(per_head) / self.M
        u = sum((p - mean) ** 2 for p in per_head) / self.M
        r_long = min(1.0, 0.2 + 0.6 * complexity)
        return LatentPrediction(
            branch_id=branch_id,
            p_t_latent=mean,
            r_long=r_long,
            u=u,
            per_head_p_t=per_head,
        )


def is_neural_available() -> bool:
    return _TORCH


# ───────────────────────────── persistence ─────────────────────────────

WEIGHTS_DIR = ROOT / "nse" / "models" / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "latent.pt"


def save_model(model: "LatentModel", path: Optional[Path] = None) -> Path:  # type: ignore[name-defined]
    """Persist a trained ensemble's weights. Returns the path written."""
    if not _TORCH:
        raise RuntimeError("torch not installed")
    path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)  # type: ignore[union-attr]
    return path


def load_ensemble(path: Optional[Path] = None, strict: bool = True) -> "LatentEnsemble":
    """Load trained weights into a :class:`LatentEnsemble`.

    Falls back to the heuristic ensemble (``model=None``) when torch is missing
    or the weights file is absent — the orchestrator keeps running either way.
    """
    path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    if not _TORCH or not path.exists():
        if strict and _TORCH and not path.exists():
            raise FileNotFoundError(f"no latent weights at {path}")
        return LatentEnsemble()
    model = LatentModel()  # type: ignore[call-arg]
    model.load_state_dict(torch.load(path, map_location="cpu"))  # type: ignore[union-attr]
    model.eval()
    return LatentEnsemble(model=model)

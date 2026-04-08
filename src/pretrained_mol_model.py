"""
Unified pre-trained molecular foundation model predictor.

Wraps any HuggingFace SMILES-based transformer (ChemBERTa-2, MoLFormer-XL,
ChemBERTa-ZINC, …) with a shared multi-task classification head, so a single
training script covers the entire family of models.

Supported checkpoints (tested):
    DeepChem/ChemBERTa-77M-MTR      (ChemBERTa-2, hidden=384, max_len=128)
    ibm/MoLFormer-XL-both-10pct     (MoLFormer-XL, hidden=768, max_len=202)
    seyonec/PubChem10M_SMILES_BPE_450k (ChemBERTa-ZINC, hidden=768, max_len=128)
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple


# ── Per-checkpoint defaults ────────────────────────────────────────────────

CHECKPOINT_DEFAULTS = {
    "DeepChem/ChemBERTa-77M-MTR": {
        "max_length": 128,
        "trust_remote_code": False,
        "cls_source": "last_hidden_state",   # use last_hidden_state[:, 0, :]
    },
    "ibm/MoLFormer-XL-both-10pct": {
        "max_length": 202,
        "trust_remote_code": True,
        "cls_source": "pooler_output",       # use pooler_output directly
    },
    "seyonec/PubChem10M_SMILES_BPE_450k": {
        "max_length": 128,
        "trust_remote_code": False,
        "cls_source": "last_hidden_state",
    },
}


def get_checkpoint_defaults(checkpoint: str) -> dict:
    """Return per-checkpoint defaults, falling back to generic settings."""
    return CHECKPOINT_DEFAULTS.get(checkpoint, {
        "max_length": 128,
        "trust_remote_code": False,
        "cls_source": "last_hidden_state",
    })


class PretrainedMolPredictor(nn.Module):
    """
    Pre-trained SMILES transformer with a multi-task classification head.

    The backbone CLS representation is extracted via either
    ``last_hidden_state[:, 0, :]`` or ``pooler_output`` depending on the
    checkpoint, then projected to task logits through dropout + linear layer.

    Args:
        pretrained_model: HuggingFace checkpoint name or local path.
        num_tasks:        Number of output tasks (12 for Tox21).
        dropout:          Dropout probability before the head.
        trust_remote_code: Required for MoLFormer-XL.
        cls_source:       "last_hidden_state" or "pooler_output".
    """

    def __init__(
        self,
        pretrained_model: str,
        num_tasks: int = 1,
        dropout: float = 0.1,
        trust_remote_code: bool = False,
        cls_source: str = "last_hidden_state",
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model, trust_remote_code=trust_remote_code
        )
        self.cls_source = cls_source
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_tasks)
        self.num_tasks = num_tasks

    def _get_cls(self, backbone_output) -> torch.Tensor:
        if self.cls_source == "pooler_output" and backbone_output.pooler_output is not None:
            return backbone_output.pooler_output
        return backbone_output.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls    = self._get_cls(out)
        logits = self.head(self.dropout(cls))
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def get_token_importance(
        self,
        smiles: str,
        tokenizer: AutoTokenizer,
        task_idx: int = 0,
        device: str = "cpu",
        max_length: int = 128,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Gradient × L2-norm token importance for a single SMILES.

        Returns:
            tokens:     List of token strings.
            importance: Numpy array of shape (n_tokens,).
        """
        self.eval()
        enc = tokenizer(
            smiles, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        embeds = self.backbone.embeddings.word_embeddings(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        out    = self.backbone(inputs_embeds=embeds, attention_mask=attention_mask)
        cls    = self._get_cls(out)
        logits = self.head(self.dropout(cls))
        score  = logits[0, task_idx] if logits.dim() == 2 else logits[0]
        score.backward()

        importance = embeds.grad[0].norm(dim=-1).detach().cpu().numpy()
        tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens, importance


def create_pretrained_mol_model(
    pretrained_model: str,
    num_tasks: int = 12,
    dropout: float = 0.1,
) -> PretrainedMolPredictor:
    """
    Factory that auto-detects per-checkpoint settings.

    Args:
        pretrained_model: HuggingFace checkpoint identifier.
        num_tasks:        Number of output tasks.
        dropout:          Dropout probability.

    Returns:
        PretrainedMolPredictor (not yet moved to device).
    """
    defaults = get_checkpoint_defaults(pretrained_model)
    return PretrainedMolPredictor(
        pretrained_model  = pretrained_model,
        num_tasks         = num_tasks,
        dropout           = dropout,
        trust_remote_code = defaults["trust_remote_code"],
        cls_source        = defaults["cls_source"],
    )

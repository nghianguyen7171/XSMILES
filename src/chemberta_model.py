"""
ChemBERTa-2 model for multi-task molecular toxicity prediction.

ChemBERTa-2 (Ahmad et al., 2022) is a RoBERTa-based transformer pre-trained
on 77M PubChem SMILES strings.  We use the MTR (multi-task regression)
checkpoint, which learns richer chemical representations than the MLM variant
and consistently outperforms it on downstream property-prediction benchmarks.

The [CLS] token embedding is projected to a 12-task linear head for Tox21.

Reference:
    Ahmad et al., "ChemBERTa-2: Towards Chemical Foundation Models",
    NeurIPS 2022 AI4Science Workshop.

HuggingFace checkpoint: DeepChem/ChemBERTa-77M-MTR
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple


class ChemBERTaPredictor(nn.Module):
    """
    ChemBERTa-2 backbone with a multi-task linear prediction head.

    The [CLS] token from the last hidden state is passed through dropout
    then a single linear layer that outputs one logit per toxicity task.

    Args:
        pretrained_model: HuggingFace model identifier (or local path).
        num_tasks:        Number of output tasks (12 for Tox21).
        dropout:          Dropout probability applied before the head.
    """

    def __init__(
        self,
        pretrained_model: str = "DeepChem/ChemBERTa-77M-MTR",
        num_tasks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_tasks)
        self.num_tasks = num_tasks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      Token IDs, shape (B, L).
            attention_mask: Padding mask, shape (B, L).

        Returns:
            Logits of shape (B,) for single-task or (B, T) for multi-task.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]        # (B, hidden)
        logits = self.head(self.dropout(cls))        # (B, T)
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)                # (B,)
        return logits                                # (B, T)

    def get_token_importance(
        self,
        smiles: str,
        tokenizer: AutoTokenizer,
        task_idx: int = 0,
        device: str = "cpu",
        max_length: int = 128,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Compute gradient-based token importance for a single SMILES string.

        Uses the L2 norm of the gradient w.r.t. the input token embeddings
        as a proxy for per-token importance.

        Args:
            smiles:    SMILES string to explain.
            tokenizer: Fitted HuggingFace tokenizer.
            task_idx:  Index of the task to explain.
            device:    "cpu" or "cuda".
            max_length: Tokenizer max sequence length.

        Returns:
            tokens:     List of token strings (including special tokens).
            importance: Numpy array of shape (n_tokens,).
        """
        self.eval()
        enc = tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Embed tokens and enable gradients on embeddings
        embeds = self.backbone.embeddings.word_embeddings(input_ids)
        embeds = embeds.detach().requires_grad_(True)

        # Forward through backbone using embeddings directly
        out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(self.dropout(cls))

        score = logits[0, task_idx] if logits.dim() == 2 else logits[0]
        score.backward()

        # Importance = L2 norm of gradient over embedding dimension
        importance = embeds.grad[0].norm(dim=-1).detach().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        return tokens, importance


def create_chemberta_model(
    pretrained_model: str = "DeepChem/ChemBERTa-77M-MTR",
    num_tasks: int = 12,
    dropout: float = 0.1,
) -> ChemBERTaPredictor:
    """
    Factory function for ChemBERTaPredictor.

    Args:
        pretrained_model: HuggingFace checkpoint name or local path.
        num_tasks:        Number of output tasks.
        dropout:          Dropout probability before the head.

    Returns:
        ChemBERTaPredictor (not yet moved to device).
    """
    return ChemBERTaPredictor(
        pretrained_model=pretrained_model,
        num_tasks=num_tasks,
        dropout=dropout,
    )

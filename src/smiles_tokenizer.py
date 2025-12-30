"""
SMILES tokenizer for sequence-based molecular representations.

Converts SMILES strings to token IDs and attention masks for transformer models.
"""

from typing import List, Tuple, Optional
import re
import torch
from collections import Counter


class SMILESTokenizer:
    """
    Simple SMILES tokenizer that tokenizes SMILES strings into subword tokens.
    
    Supports:
    - Character-level tokenization
    - Subword tokenization (BRICS-like patterns)
    - Special tokens (PAD, UNK, SOS, EOS)
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        max_length: int = 128,
        padding: str = "right",
        truncation: bool = True,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize SMILES tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            padding: Padding side ("right" or "left")
            truncation: Whether to truncate long sequences
            special_tokens: List of special tokens to add
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Special tokens
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.special_tokens = special_tokens
        
        # Vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_initialized = False
        
        # Token patterns (regex patterns for common SMILES tokens)
        self.token_patterns = [
            # Rings (e.g., c1ccccc1)
            r'\d+',
            # Common atoms (single characters)
            r'[BCOPSFClBrI]',
            # Bonds
            r'[=#\-:]',
            # Branching
            r'[\(\)\[\]]',
            # Aromatic atoms
            r'[cnops]',
            # Aliphatic atoms
            r'[CNOP]',
            # Others
            r'.'
        ]
    
    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string into tokens.
        
        Args:
            smiles: SMILES string
        
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        while i < len(smiles):
            matched = False
            for pattern in self.token_patterns:
                match = re.match(pattern, smiles[i:])
                if match:
                    token = match.group(0)
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            
            if not matched:
                # Default: single character
                tokens.append(smiles[i])
                i += 1
        
        return tokens
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1):
        """
        Build vocabulary from a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency for a token to be included
        """
        # Tokenize all SMILES
        all_tokens = []
        for smiles in smiles_list:
            tokens = self._tokenize_smiles(smiles)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary (special tokens first)
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Add most common tokens (up to vocab_size)
        next_id = len(self.special_tokens)
        for token, count in token_counts.most_common(self.vocab_size - len(self.special_tokens)):
            if count >= min_freq:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        
        self.vocab_initialized = True
    
    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Encode a SMILES string to token IDs and attention mask.
        
        Args:
            smiles: SMILES string
            add_special_tokens: Whether to add SOS/EOS tokens
        
        Returns:
            Tuple of (token_ids, attention_mask)
        """
        if not self.vocab_initialized:
            raise ValueError("Vocabulary not initialized. Call build_vocab() first.")
        
        # Tokenize
        tokens = self._tokenize_smiles(smiles)
        
        # Convert to IDs
        token_ids = []
        if add_special_tokens and "<SOS>" in self.token_to_id:
            token_ids.append(self.token_to_id["<SOS>"])
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            elif "<UNK>" in self.token_to_id:
                token_ids.append(self.token_to_id["<UNK>"])
            # Otherwise skip (shouldn't happen if vocab is built correctly)
        
        if add_special_tokens and "<EOS>" in self.token_to_id:
            token_ids.append(self.token_to_id["<EOS>"])
        
        # Truncate if necessary
        if self.truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad if necessary
        pad_id = self.token_to_id.get("<PAD>", 0)
        if len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            if self.padding == "right":
                token_ids = token_ids + [pad_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            else:  # left
                token_ids = [pad_id] * pad_length + token_ids
                attention_mask = [0] * pad_length + attention_mask
        
        return token_ids, attention_mask
    
    def encode_batch(
        self,
        smiles_list: List[str],
        add_special_tokens: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            add_special_tokens: Whether to add SOS/EOS tokens
        
        Returns:
            Tuple of (token_ids_tensor, attention_mask_tensor)
        """
        batch_token_ids = []
        batch_attention_masks = []
        
        for smiles in smiles_list:
            token_ids, attention_mask = self.encode(smiles, add_special_tokens)
            batch_token_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)
        
        return (
            torch.tensor(batch_token_ids, dtype=torch.long),
            torch.tensor(batch_attention_masks, dtype=torch.long)
        )
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to SMILES string.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            SMILES string
        """
        if not self.vocab_initialized:
            raise ValueError("Vocabulary not initialized. Call build_vocab() first.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return "".join(tokens)


def create_tokenizer_from_smiles(
    smiles_list: List[str],
    vocab_size: int = 100,
    max_length: int = 128,
    min_freq: int = 1
) -> SMILESTokenizer:
    """
    Create and initialize a SMILES tokenizer from a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        min_freq: Minimum frequency for tokens
    
    Returns:
        Initialized SMILESTokenizer
    """
    tokenizer = SMILESTokenizer(
        vocab_size=vocab_size,
        max_length=max_length
    )
    tokenizer.build_vocab(smiles_list, min_freq=min_freq)
    return tokenizer


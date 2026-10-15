"""Architecture matching the original PhoMT state dictionaries."""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Seq2SeqTransformerOptimized(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 53685,
        tgt_vocab_size: int = 53685,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 53684
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.fc_out.weight = self.tgt_embedding.weight

    def encode(self, src: torch.Tensor):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        src_padding_mask = (src == self.pad_idx)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask), src_padding_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_padding_mask: torch.Tensor):
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device)
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)

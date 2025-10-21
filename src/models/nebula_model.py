import math, torch
import torch.nn as nn
from .chunked_attention import MultiheadSelfAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        n = x.size(1)
        return x + self.pe[:, :n, :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, chunk_size=0):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, chunk_size=chunk_size)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = src
        x2 = self.self_attn(x, src_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

class NebulaTiny(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=1, dim_feedforward=512, max_len=512, num_classes=2, chunk_size=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])
        self.cls = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        h = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        h = self.pos(h)
        for layer in self.layers:
            h = layer(h, mask)
        pooled = h[:, 0, :]  # CLS-like first token pooling
        return self.cls(pooled)

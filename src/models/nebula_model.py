# src/models/nebula_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]

# === Custom Multi-Head Self-Attention with separate q/k/v/o (old-key compatible) ===
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        H = self.nhead
        q = self.q(x).view(B, T, H, D // H).transpose(1, 2)  # [B,H,T,head_dim]
        k = self.k(x).view(B, T, H, D // H).transpose(1, 2)
        v = self.v(x).view(B, T, H, D // H).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        att = att.softmax(dim=-1)
        out = torch.matmul(att, v)                               # [B,H,T,head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)     # [B,T,D]
        return self.o(out)

# === Transformer Block (old-key compatible) ===
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff):
        super().__init__()
        self.self_attn = SelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff)
        self.linear2 = nn.Linear(ff, d_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x

# === NebulaTiny (old-key compatible + forward_from_embeddings) ===
class NebulaTiny(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                 max_len=512, num_classes=2, chunk_size=0):
        super().__init__()
        # Names match the old checkpoint:
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

        # IMPORTANT: hidden size must match old checkpoint (64 when d_model=128)
        hidden = d_model // 2   # ensures shapes [64,128] and [2,64] for cls.* when d_model=128
        self.cls = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward_from_embeddings(self, emb):
        x = self.pos(emb)          # [B,T,D]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)          # [B,D] (mean pool over T)
        return self.cls(x)         # [B,num_classes]

    def forward(self, input_ids):
        emb = self.embed(input_ids)    # [B,T,D]
        return self.forward_from_embeddings(emb)

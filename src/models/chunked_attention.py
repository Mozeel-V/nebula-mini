import torch
import torch.nn as nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.chunk_size = chunk_size  # 0 means full attention

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.dk).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.dk).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.dk).transpose(1, 2)

        if self.chunk_size and self.chunk_size > 0:
            S = self.chunk_size
            assert N % S == 0, "N must be divisible by chunk_size"
            M = N // S
            q = q.view(B, self.num_heads, M, S, self.dk)
            k = k.view(B, self.num_heads, M, S, self.dk)
            v = v.view(B, self.num_heads, M, S, self.dk)
            out_chunks = []
            for m in range(M):
                qi = q[:, :, m]   # [B,h,S,dk]
                ki = k[:, :, m]
                vi = v[:, :, m]
                scores = torch.matmul(qi, ki.transpose(-2, -1)) / (self.dk ** 0.5)
                if mask is not None:
                    mi = mask[:, :, :, m*S:(m+1)*S]
                    scores = scores.masked_fill(mi == 0, -1e9)
                attn = torch.softmax(scores, dim=-1)
                outi = torch.matmul(attn, vi)
                out_chunks.append(outi)
            out = torch.cat(out_chunks, dim=2)  # [B,h,N,dk]
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o(out)

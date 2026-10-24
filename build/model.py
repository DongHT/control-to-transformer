"""Small causal Transformer policy: state history (H x 6) -> force command.

Pre-LN encoder blocks, learned positional embedding, causal self-attention,
final token -> LayerNorm -> linear -> u_max*tanh(.).
No plant knowledge is used: only measured states + its own previous action.
"""
import math
import torch
import torch.nn as nn

H = 24
FEAT = 6
U_MAX = 12.0


class Block(nn.Module):
    def __init__(self, d, nh, dff, p=0.0):
        super().__init__()
        self.nh = nh
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, dff)
        self.fc2 = nn.Linear(dff, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.nh, D // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, D // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, D // self.nh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.nh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        h = self.fc2(self.act(self.fc1(self.ln2(x))))
        return x + self.drop(h)


class Policy(nn.Module):
    def __init__(self, d=32, nh=4, dff=64, layers=2, h=H, feat=FEAT, umax=U_MAX):
        super().__init__()
        self.h, self.feat, self.umax = h, feat, umax
        self.inp = nn.Linear(feat, d)
        self.pos = nn.Parameter(torch.randn(h, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nh, dff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x):                     # x: (B,H,FEAT)
        z = self.inp(x) + self.pos[None, :, :]
        for b in self.blocks:
            z = b(z)
        z = self.ln(z[:, -1, :])
        return self.umax * torch.tanh(self.head(z)).squeeze(-1)

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#-----------------

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd: int = 384

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, n_embd)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) #sometimes abbreviated to FFN!

    def forward(self, x):
        # We add residual pathways (not in AIAYN), and make sure its done after the layer norm & attention blocks (or layer norm and MLP)
        # We prefer this "clean" residual pathway.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # nn.Embedding is just a glorified tensor with easier indexing access!
            wpe = nn.Embedding(config.block_size, config,n_embd), # Number of position tokens equals context length
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd) # We end the transformer with a final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)


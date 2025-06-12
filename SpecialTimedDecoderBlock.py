import torch
import torch.nn as nn
from torch.nn import functional as F
# from g_mlp_pytorch import gMLP
# from Data import load_tokenizer
# import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpecialTimedDecoderBlock(nn.Module):
    
    def __init__(self, timesteps, num_heads, context_length, embed_size):
       super().__init__()
       self.embeddings = nn.Embedding(num_embeddings=timesteps, embedding_dim=embed_size)
       
       self.LLMBlock = TLMBlock(num_heads, context_length, embed_size)
       
       self.timer = torch.tensor([0])
       self.timer.requires_grad_(False)
       
       self.time_steps = timesteps
       
    def forward(self, current_embs):
        
        new_embs = current_embs
        for i in range(self.time_steps):
            self.timer[0] = i
            # print(self.timer)
            current_embs = current_embs + new_embs + self.embeddings(self.timer)
            new_embs = self.LLMBlock(current_embs)
        
        return new_embs



class Head(nn.Module):

    def __init__(self, context_length, embed_size, head_dim):
        super().__init__()

        self.queries = nn.Linear(embed_size, head_dim, bias = False)
        self.keys = nn.Linear(embed_size, head_dim, bias=False)
        self.values = nn.Linear(embed_size, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        # self.ln = nn.LayerNorm(head_dim)
        # self.ln2 = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
    def forward(self, X):

        B,T,C = X.shape
        q = self.queries(X)
        k = self.keys(X)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        v = self.values(X)

        out = wei@v
        # print(out.shape)
        return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, context_length, embed_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(context_length, embed_size, embed_size//num_heads) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_size,embed_size)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.fc(out)
        return out

class TLMBlock(nn.Module):

  def __init__(self, num_heads, context_length, embed_size):
    super().__init__()

    self.ln_1 = nn.LayerNorm(embed_size)
    self.sa_head = MultiHeadedAttention(num_heads, context_length,embed_size)
    self.dropout = nn.Dropout(p=0.2)
    self.ln_2 = nn.LayerNorm(embed_size)
    # self.silu = nn.SiLU()
    self.mlp = nn.Sequential(
       nn.Linear(embed_size, 2*embed_size),
       nn.Linear(2*embed_size,embed_size),
       nn.Linear(embed_size,embed_size),
       nn.GELU(),
       nn.Dropout(p=0.1)
    )
  def forward(self, x):

    # B,T = x.shape
    # print(B,T)
    x = x+self.sa_head(self.ln_1(x))
    # print(x.shape)
    x = x + self.mlp(self.ln_2(x))

    # print(x.shape)
    return x


if __name__ == "__main__":
    
    prev_embs = torch.randn((1, 768, 1024))
    
    specialblock = SpecialTimedDecoderBlock(5, 4, 768, 1024)
    
    out = specialblock(prev_embs)
    print(out)
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import TLMBlock
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



if __name__ == "__main__":
    
    prev_embs = torch.randn((1, 768, 1024))
    
    specialblock = SpecialTimedDecoderBlock(5, 4, 768, 1024)
    
    out = specialblock(prev_embs)
    print(out)
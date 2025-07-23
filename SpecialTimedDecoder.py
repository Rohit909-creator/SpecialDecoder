import torch
import torch.nn as nn
from torch.nn import functional as F
# from g_mlp_pytorch import gMLP
# from Data import load_tokenizer
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# torch.set_default_device('cpu')

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        # SwiGLU: Swish(xW1) ⊙ (xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # More numerically stable RMSNorm
        # Calculate mean of squares
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # RMS normalization
        rms = torch.sqrt(mean_square + self.eps)
        return self.weight * x / rms


class TLMBlock(nn.Module):

  def __init__(self, num_heads, context_length, embed_size):
    super().__init__()

    self.ln_1 = RMSNorm(embed_size)
    # self.sa_head = MultiHeadedAttention(num_heads, context_length,embed_size)
    
    self.dropout = nn.Dropout(p=0.2)
    self.ln_2 = RMSNorm(embed_size)
    # self.silu = nn.SiLU()
    self.mlp = nn.Sequential(
      nn.Linear(embed_size, 2*embed_size),
      SwiGLU(2*embed_size,2*embed_size),
      nn.Dropout(p=0.1),
      nn.Linear(2*embed_size, embed_size),
    )
  def forward(self, x):

    # B,T = x.shape
    # print(B,T)
    # x = x+self.sa_head(self.ln_1(x))
    x = self.ln_1(x)
    x = x + nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True, dropout_p=0.2)
    # print(x.shape)
    x = x + self.mlp(self.ln_2(x))

    # print(x.shape)
    return x

class SpecialTimedDecoderBlock(nn.Module):

    def __init__(self, timesteps, num_heads, context_length, embed_size, device='cpu'):
      
      super().__init__()

      self.device = device
      self.embeddings = nn.Embedding(num_embeddings=timesteps, embedding_dim=embed_size) # Did u know that u could use embeddings as a clock signal
      self.LLMBlock = TLMBlock(num_heads, context_length, embed_size)
      #self.timer = torch.tensor([0]).to(device=device)
      #self.timer.requires_grad_(False)

      self.time_steps = timesteps

    def forward(self, current_embs):
      timer_embs = self.embeddings(torch.arange(self.time_steps, device=self.device))
      
      x = current_embs
      for i in range(self.time_steps):
          x = x + timer_embs[i].unsqueeze(0)  # Assuming batch dimension
          x = self.LLMBlock(x)
      return x  


    # Next Time Baby
    # def forward(self, current_embs):
    #     timer = torch.tensor([i for i in range(self.time_steps)]).to(self.device) # this controls the timer
    #     prev_embs = None
    #     hx, cx = torch.zeros(current_embs.shape), torch.zeros(current_embs.shape)
    #     for i in range(self.time_steps):
            
    #         if i == 0:
    #           new_embs = current_embs + self.embeddings(timer[i].unsqueeze(0))
              
    #         else:
    #           hx, cx = self.LSTMCELL(prev_embs, (hx, cx))
    #           current_embs = current_embs + hx + self.embeddings(timer[i].unsqueeze(0)) # and this is how u do it
    #           new_embs = self.LLMBlock(current_embs)

    #         prev_embs = new_embs
    #     return new_embs


class RecurrentLM(nn.Module):

  def __init__(self, vocab_size, context_length, embed_size, num_blocks = 5, device = 'cpu'):
    super().__init__()
    self.token_embeddings = nn.Embedding(vocab_size, embed_size)
    self.positional_embeddings = nn.Embedding(context_length, embed_size)
    self.word_embeddings = nn.Embedding(context_length, embed_size)
    self.context_length = context_length

    self.blocks = SpecialTimedDecoderBlock(num_blocks, 4, context_length, embed_size, device=device)

    # self.block = nn.ModuleList()
    # for _ in range(num_blocks):
    #   self.block.append(TLMBlock( 4, context_length,embed_size))
    self.device = device
    self.lm_head = nn.Linear(embed_size,vocab_size)
    self.silu = nn.SiLU()

  def forward(self, idx, targets=None):
    B,T = idx.shape
    # print(idx.shape)
    # print(B,T)
    tok_emb = self.token_embeddings(idx)
    # print(tok_emb.shape)
    pos_emb = self.positional_embeddings(torch.arange(T, device=self.device))
    # word_emb = self.word_embeddings(idx)
    x = tok_emb + pos_emb
    # print(x.shape)
    # for _,layer in enumerate(self.block):
    #   x = layer(x)
    x = self.blocks(x)

    logits = self.lm_head(x)
    if targets == None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      # print(logits.shape)
      targets = targets.view(B*T)
      # print(targets.shape)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
      for _ in range(max_new_tokens):
          idx_cond = idx[:, -self.context_length:]
          logits, _ = self(idx_cond)
          logits = logits[:, -1, :]  # last time step

          # Apply temperature
          logits = logits / temperature

          # Top-k filtering
          if top_k is not None:
              values, indices = torch.topk(logits, top_k)
              mask = torch.full_like(logits, float('-inf'))
              logits = mask.scatter(1, indices, values)

          # Top-p filtering
          if top_p is not None:
              sorted_logits, sorted_indices = torch.sort(logits, descending=True)
              cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

              sorted_mask = cum_probs > top_p
              sorted_mask[..., 1:] = sorted_mask[..., :-1]
              sorted_mask[..., 0] = 0

              mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
              logits = logits.masked_fill(mask, float('-inf'))

          # Sample from final distribution
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)

          idx = torch.cat((idx, idx_next), dim=1)

      return idx


def generate(string):

  for s in string:
    print(s, end="")
    time.sleep(0.05)

if __name__ == "__main__":

    # prev_embs = torch.randn((1, 768, 1024)).to(device="cuda")

    # specialblock = SpecialTimedDecoderBlock(5, 4, 768, 1024, device='cuda')
    # specialblock = specialblock.to(device='cuda')
    # out = specialblock(prev_embs)
    # print(out)
    # print(out.shape)

    tlm = RecurrentLM(4,8,32,10,device=device)
    tlm = tlm.to(device)
    x = torch.ones((8,4),dtype=torch.long).to(device)
    out = tlm(x)
    print(out[0].shape)
    
    # from Data import load_tokenizer
    
    # model_path = r".\special_decoder_logs\new_model.pt"
    # cache_dir = "./cache"

    # vocab_size = 11799
    # context_length = 768
    # n_embs = 512
    
    # m = RecurrentLM(vocab_size,context_length,n_embs,12, device=device)
    # m.load_state_dict(torch.load(model_path))
    # # print(f"Model:{m.named_modules}\n\n")
    
    
    # # named_children = m.named_children()
    # # print(named_children)
    
    
    # # m.block = m.block[:5]
    # # print(len(m.block))
    # m = m.to(device)
    
    # tokenizer = load_tokenizer(cache_dir)
    
    # # initial_text = "and go to sleep. But the twins didn't want to sleep yet. They wanted to"
    # # context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)
    
    
    
    # with torch.no_grad():
    #   initial_text = "what are you doing?"
    #   context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)
    #   # m.generate(context, 100)
    #   generated_tokens = m.generate(context, 100, 0.2, top_k=50)[0].tolist()
    #   generated_text = tokenizer.decode(generated_tokens)
    #   print(f"Generated: {generated_text[:1000]}")
    
    # # tlm = TLM(4,8,32,10).load_state_dict(torch.load(model_path))
    # # x = torch.ones((8,4),dtype=torch.long)
    # # out = tlm(x)
    # # print(out[0].shape)
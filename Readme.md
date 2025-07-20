## This is an experiment on LLMs, based on recent findings from mechanistic Interpretability

*Btw, this code is a continuition of my SimpleLLM repo,*

- So based on a recent video I saw, LLMs need deep layers to make complex circuits and get complex understandings,
so for that we need large number of the LLM-Blocks, so how about this, what if could make a specific layer or a new kind of block to model what the other layers do, basically said,

-if we actually had,

*embeddings -> block1 -> block2 -> block3 -> block4 -> block5 -> block6 -> ................... blockN -> lmhead*

-what if could do,

*embeddings -> block1 -> block2 -> block3 -> block4 -> SpecialBlock -> blockN*

-thus replacing the computation of blocks from 5 to N-1 with SpecialBlock,

-Where Special Block can be a recurrent transformer model or someother model which does computation (N-1 - 6) => (N-7) times to model deep representations.
-This is a work in progress, so please feel free to contribute or suggest improvements.

**So main the higlight of this experiment is to see if we can replace the computation of multiple blocks with a single block that can model the deep representations of the previous blocks, so for that I made this SpecialTimedDecoderBlock with a trick on using nn.Embedding to keep track of the time steps**

```
class SpecialTimedDecoderBlock(nn.Module):

    def __init__(self, timesteps, num_heads, context_length, embed_size, device='cpu'):
       super().__init__()

       self.device = device

       self.embeddings = nn.Embedding(num_embeddings=timesteps, embedding_dim=embed_size)
       self.LLMBlock = TLMBlock(num_heads, context_length, embed_size)
    #    self.timer = torch.tensor([0]).to(device=device)
    #    self.timer.requires_grad_(False)

       self.time_steps = timesteps

    def forward(self, current_embs):
        timer = torch.tensor([i for i in range(self.time_steps)]).to(self.device)
        new_embs = current_embs
        for i in range(self.time_steps):
            # timer[0] = i
            # print(timer[i].unsqueeze(0))
            current_embs = current_embs + new_embs + self.embeddings(timer[i].unsqueeze(0))
            new_embs = self.LLMBlock(current_embs)

        return new_embs
```
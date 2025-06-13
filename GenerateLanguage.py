import torch
import torch.nn as nn
from torch.nn import functional as F
# from g_mlp_pytorch import gMLP
from model import TLM
from Data import load_tokenizer
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model_path = r"C:\Users\Rohit Francis\Downloads\model2 (1).pt"
cache_dir = "./cache"

m = TLM(11799,768,1024,12)
m.load_state_dict(torch.load(model_path))
# print(f"Model:{m.named_modules}\n\n")


# named_children = m.named_children()
# print(named_children)


# m.block = m.block[:5]
# print(len(m.block))
m = m.to(device)

tokenizer = load_tokenizer(cache_dir)

# initial_text = "and go to sleep. But the twins didn't want to sleep yet. They wanted to"
# context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)



with torch.no_grad():
  initial_text = "and go to sleep. But the twins didn't want to sleep yet. They wanted to"
  context = torch.tensor([tokenizer.encode(initial_text)], dtype=torch.long).to(device)
  # m.generate(context, 100)
  generated_tokens = m.generate(context, 100)[0].tolist()
  generated_text = tokenizer.decode(generated_tokens)
  print(f"Generated: {generated_text[:1000]}")

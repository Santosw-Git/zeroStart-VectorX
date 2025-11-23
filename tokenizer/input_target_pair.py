

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


with open("the-verdict.txt", "r",encoding="utf-8") as f:
  text = f.read()

class Gpt2Dataset(Dataset):
  def __init__(self, text, tokenizer, context_length, stride):
    self.input_ids = []
    self.target_ids = []
    token_ids = tokenizer.encode(text)

    for i in range(0, len(token_ids) - context_length, stride):
      context = token_ids[i:i+context_length]
      desired = token_ids[i+1:i+1+context_length]
      self.input_ids.append(torch.tensor(context))
      self.target_ids.append(torch.tensor(desired))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]

def dataLoader(text, batch_size=4, context_length=256, stride=128, drop_last=True, shuffle=True, num_workers=4):

  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = Gpt2Dataset(text, tokenizer, context_length, stride)

  dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    drop_last=drop_last,
    shuffle=shuffle,
    num_workers=num_workers
    )

  return dataloader

dataloader = dataLoader(text, batch_size=4, context_length=6, stride=1, drop_last=True, shuffle=True, num_workers=2)

# for x, y in dataloader:
#   print("Input_tensor", x)
#   print(tokenizer.decode(x[0].tolist()))
#   print()
#   print("Target_tensor", y)
#   print(tokenizer.decode(y[0].tolist()))
#   break

vocab_size = 50257
output_dim = 256

embeddings = torch.nn.Embedding(vocab_size, output_dim)

vocab_size = 50257
output_dim = 256
context_length = 4

embeddings = torch.nn.Embedding(vocab_size, output_dim)
pos_encoding_layer = torch.nn.Embedding(context_length, output_dim)
pos_encoding = pos_encoding_layer(torch.arange(context_length))

for input, target in dataloader:
  token_embeddings = embeddings(input)
  input_embedding = token_embeddings + pos_encoding
  break


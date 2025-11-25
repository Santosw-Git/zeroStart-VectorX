from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
from config.config import GPT_CONFIG_124M
from dataset.data import pdf_text as text

class DatasetBatch(Dataset):
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
  dataset = DatasetBatch(text, tokenizer, context_length, stride)

  dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    drop_last=drop_last,
    shuffle=shuffle,
    num_workers=num_workers
    )

  return dataloader

train_ratio = 0.80
split_idx = int(train_ratio * len(text))
train_data = text[:split_idx]
val_data = text[split_idx:]


torch.manual_seed(123)

train_loader = dataLoader(
    train_data,
    batch_size=2,
    context_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = dataLoader(
    val_data,
    batch_size=2,
    context_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")

for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)
print(len(val_loader))
print(len(train_loader))

from models.model import Model
from logits.logits import text_to_token_ids, token_ids_to_text
from config.config import GPT_CONFIG_124M
import tiktoken
import torch
# inference test

def generateinference(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).any():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

model = Model(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)

token_ids = generateinference(
    model=model,
    idx=text_to_token_ids("I HAD always thought Jack Gisburn rather a ", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
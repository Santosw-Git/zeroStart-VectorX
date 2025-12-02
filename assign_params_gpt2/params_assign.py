
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config.config import CONFIG_124M 
from models.model import Model
from training.inference import generateinference
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

NEW_CONFIG = CONFIG_124M.copy()
# NEW_CONFIG["vocab_size"] = hf_model.config.vocab_size
NEW_CONFIG.update({"context_length": hf_model.config.n_positions, "n_embd": hf_model.config.n_embd, "qkv_bias": True,})
# print(hf_model.config.to_dict())


model = Model(NEW_CONFIG)
model.eval()

model.tok_emb.weight.data = hf_model.transformer.wte.weight.data.clone()
model.pos_emb.weight.data = hf_model.transformer.wpe.weight.data.clone()
model.out_head.weight.data = hf_model.lm_head.weight.data.clone()

for i in range(CONFIG_124M["n_layers"]):
    hf_block = hf_model.transformer.h[i]

    # QKV weights and biases
    qkv_w = hf_block.attn.c_attn.weight.data
    qkv_b = hf_block.attn.c_attn.bias.data
    q_w, k_w, v_w = qkv_w.chunk(3, dim=1)
    q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
    model.trf_blocks[i].att.W_query.weight.data = q_w.T.clone()
    model.trf_blocks[i].att.W_key.weight.data   = k_w.T.clone()
    model.trf_blocks[i].att.W_value.weight.data = v_w.T.clone()

    model.trf_blocks[i].att.W_query.bias.data   = q_b.clone()
    model.trf_blocks[i].att.W_key.bias.data     = k_b.clone()
    model.trf_blocks[i].att.W_value.bias.data   = v_b.clone()

    # Attention output projection
    model.trf_blocks[i].att.out_proj.weight.data = hf_block.attn.c_proj.weight.data.T.clone()
    model.trf_blocks[i].att.out_proj.bias.data   = hf_block.attn.c_proj.bias.data.clone()

    # Feed-forward MLP
    model.trf_blocks[i].ff.layers[0].weight.data = hf_block.mlp.c_fc.weight.data.T.clone()
    model.trf_blocks[i].ff.layers[0].bias.data   = hf_block.mlp.c_fc.bias.data.clone()
    model.trf_blocks[i].ff.layers[2].weight.data = hf_block.mlp.c_proj.weight.data.T.clone()
    model.trf_blocks[i].ff.layers[2].bias.data   = hf_block.mlp.c_proj.bias.data.clone()

  # LayerNorm 1
    model.trf_blocks[i].norm1.scale.data = hf_block.ln_1.weight.data.clone()
    model.trf_blocks[i].norm1.shift.data = hf_block.ln_1.bias.data.clone()

    # LayerNorm 2
    model.trf_blocks[i].norm2.scale.data = hf_block.ln_2.weight.data.clone()
    model.trf_blocks[i].norm2.shift.data = hf_block.ln_2.bias.data.clone()


# Final LayerNorm
model.final_norm.scale.data = hf_model.transformer.ln_f.weight.data.clone()
model.final_norm.shift.data = hf_model.transformer.ln_f.bias.data.clone()


model.to(device)


input_text = "Once upon a time in a small village, a young girl discovered"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

with torch.no_grad():
    token_ids = generateinference(
    model=model,
    idx=input_ids,
    max_new_tokens=15,
    context_size=CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
    print(tokenizer.decode(token_ids[0]))



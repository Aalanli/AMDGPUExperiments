# %%
import time
import torch
from matplotlib import pyplot as plt
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer

model_name = 'decapoda-research/llama-7b-hf'
tok = LlamaTokenizer.from_pretrained(model_name)
with torch.device('cuda'):
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# %%
def generate_torch(input_ids: str, tokenizer, torch_model, num_tokens, device='cuda', dtype=torch.float16):
    torch_model = torch_model.to(device=device, dtype=dtype)
    input_ids = tokenizer.encode(input_ids)
    input_ids = torch.tensor(input_ids).to(device=device).unsqueeze(0)
    config = torch_model.config

    attention_mask = torch.ones([1, config.max_position_embeddings]).to(device=device, dtype=dtype)
    # position_ids = torch.arange(0, config.max_position_embeddings, device='cuda').unsqueeze(0)
    make_past = lambda: torch.zeros(
        [1, config.num_key_value_heads, 0, config.hidden_size // config.num_key_value_heads]
    ).to(device=device, dtype=dtype)
    key_value_cache = [(make_past(), make_past()) for _ in range(config.num_hidden_layers)]
    outputs = []
    cur_len = input_ids.shape[-1]
    timings = []
    for _ in range(num_tokens):
        t1 = time.time()
        y = torch_model(
            input_ids,
            attention_mask=attention_mask[:, :cur_len],
            position_ids=None,
            past_key_values=key_value_cache,
            use_cache=True,
        )
        logits = y['logits']
        new_ids = torch.argmax(logits, -1, keepdim=False)
        new_ids = new_ids[:, -1:]
        t2 = time.time()
        timings.append(t2 - t1)
        outputs.append(new_ids[0, -1].item())
        input_ids = new_ids
        key_value_cache = y['past_key_values']
        cur_len += 1

    print(f"Average time: {sum(timings) / len(timings)}")
    quantiles = torch.quantile(torch.tensor(timings), torch.tensor([0.5, 0.2, 0.8]))
    print(f"Median time: {quantiles[0]}")
    plt.plot(timings)
    plt.show()
    return tokenizer.decode(outputs)


print(generate_torch('The cat', tok, model, 50))


# 50 tokens, RTX 3090
# Average time: 0.02653202533721924
# Median time: 0.026173830032348633

# 50 tokens, Mi210
# Average time: 0.025665740966796875
# Median time: 0.025589466094970703

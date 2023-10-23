# %%
from tqdm import tqdm
import time
import torch
from transformers.models.llama import LlamaForCausalLM, LlamaConfig


config = LlamaConfig(
    num_hidden_layers=6,
)
with torch.device('cuda'):
    model = LlamaForCausalLM(config)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

samples = 100
batch_size = 8
seq_len = config.max_length
dataset = [
    torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda', dtype=torch.int64) for i in range(samples)
]

forward = []
backward = []
grad_step = []
for x in tqdm(dataset):
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.autocast('cuda', torch.float16, enabled=True):
        y = model(x, labels=x)
    torch.cuda.synchronize()
    t2 = time.time()
    y.loss.backward()
    torch.cuda.synchronize()
    t3 = time.time()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t4 = time.time()
    forward.append(t2 - t1)
    backward.append(t3 - t2)
    grad_step.append(t4 - t3)

quantile = lambda x: torch.quantile(torch.tensor(x), torch.tensor([0.5, 0.2, 0.8]))

print(f"Forward: {quantile(forward)}")
print(f"Backward: {quantile(backward)}")
print(f"Grad Step: {quantile(grad_step)}")


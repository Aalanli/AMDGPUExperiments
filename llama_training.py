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


forward = [torch.cuda.Event(enable_timing=True) for _ in range(len(dataset))]
backward = [torch.cuda.Event(enable_timing=True) for _ in range(len(dataset))]
grad_step1 = [torch.cuda.Event(enable_timing=True) for _ in range(len(dataset))]
grad_step2 = [torch.cuda.Event(enable_timing=True) for _ in range(len(dataset))]
torch.cuda.synchronize()
for i, x in tqdm(enumerate(dataset)):
    t1 = time.time()
    forward[i].record()
    with torch.autocast('cuda', torch.float16, enabled=True):
        y = model(x, labels=x)
    backward[i].record()
    y.loss.backward()
    grad_step1[i].record()
    optimizer.step()
    optimizer.zero_grad()
    grad_step2[i].record()

torch.cuda.synchronize()
quantile = lambda x: torch.quantile(torch.tensor(x), torch.tensor([0.5, 0.2, 0.8]))

forward = [s.elapsed_time(e) for s, e in zip(forward, backward)]
backward = [s.elapsed_time(e) for s, e in zip(backward, grad_step1)]
grad_step = [s.elapsed_time(e) for s, e in zip(grad_step1, grad_step2)]
print(f"Forward: {quantile(forward)}")
print(f"Backward: {quantile(backward)}")
print(f"Grad Step: {quantile(grad_step)}")


# batch_size = 8, Mi210
# Forward: tensor([0.0212, 0.0211, 0.0214])
# Backward: tensor([0.0310, 0.0309, 0.0310])
# Grad Step: tensor([0.0291, 0.0290, 0.0292])

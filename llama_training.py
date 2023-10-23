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
# Forward: tensor([20.8165, 20.8043, 20.8263])
# Backward: tensor([30.4398, 30.4243, 30.4643])
# Grad Step: tensor([28.7892, 28.7251, 28.8487])

# batch_size = 8, RTX 3090
# Forward: tensor([23.7425, 23.1397, 24.2870])
# Backward: tensor([31.6831, 31.4556, 32.0506])
# Grad Step: tensor([24.2524, 23.8010, 25.0170])

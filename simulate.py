# %%
import torch
import triton
import triton.language as tl

# 0 .. n
# p = prob of moving left
@triton.jit
def simulate_kernel(
    output,
    n_steps,
    seed,
    p,
    start,
    block_size: tl.constexpr
):
    n_program = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    state = tl.full([block_size], start, dtype=tl.uint32)
    for _ in range(n_steps):
        this_seed = tl.randint(seed, pid)
        rand = tl.rand(this_seed, offsets)
        state = tl.where(state == 0, state, tl.where(rand < p, state - 1, state + 1))
        pid += n_program

    fall_off = state == 0
    prob = tl.sum(fall_off.to(tl.int64))
    tl.store(output + tl.program_id(0), prob)

def simulate(start, seed, p, n_steps, num_blocks):
    block_size = 1024
    output = torch.empty([num_blocks], dtype=torch.int64, device='cuda')
    simulate_kernel[(num_blocks,)](output, n_steps, seed, p, start, block_size=block_size)
    return output.sum()

num_blocks = 7000
pos = 1
n_fall = simulate(pos, 42, 1/3, 100000, num_blocks).item()
n_sim = num_blocks * 1024
print(n_fall)
print(n_sim)
print(n_fall / n_sim)
print(1 / (2 ** pos))

# %%
n = 50
probs = torch.zeros([n], dtype=torch.float64, device='cuda')
probs[0] = 1.0
probs = probs.reshape([1, 1, n])
state_transition = torch.tensor([2/3, 0, 1/3], dtype=torch.float64, device='cuda').unsqueeze(0).unsqueeze(0)
for i in range(1, n):
    probs = torch.conv1d(probs, state_transition, padding=1)
    print(probs)
    probs[0, 0, 0] = 1.0

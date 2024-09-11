import torch
import torch_xla2

env = torch_xla2.default_env()
env.config.debug_print_each_op = True
env.config.debug_accuracy_for_each_op = True

with env:
    input = torch.tensor([1, 5, 10], dtype=torch.float32)
    q = torch.tensor(0.5)
    print(torch.quantile(input, q))
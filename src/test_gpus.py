import torch
print(f'cuda: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
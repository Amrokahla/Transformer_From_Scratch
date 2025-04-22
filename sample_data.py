import torch

def get_dummy_input(seq_len=6, d_model=64):
    return torch.rand(1, seq_len, d_model)

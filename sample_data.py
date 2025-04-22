import torch

def get_dummy_inputs(vocab_size=100, seq_len=6, batch_size=1):
    return torch.randint(0, vocab_size, (batch_size, seq_len))

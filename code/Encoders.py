import torch
import torch.nn as nn
import numpy as np

class FreqEncoder(nn.Module):
    def __init__(self, max_freq=10, num_freqs=10, d=3): #TEMP
        super(FreqEncoder, self).__init__()
        periodics = [torch.sin, torch.cos]
        freq_bands = 2.**torch.linspace(0., max_freq-1, steps=num_freqs)
        encoder_fns = [lambda x, fn=fn, freq=freq: fn(x*freq)
                       for freq in freq_bands
                       for fn in periodics]
        out_dim = len(encoder_fns)*d
        self.encoder_fns = encoder_fns
        self.out_dim = out_dim
        
    def forward(self, x):
        return torch.cat([fn(x) for fn in self.encoder_fns], -1)


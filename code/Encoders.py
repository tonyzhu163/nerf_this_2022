import torch
import torch.nn as nn

class FreqEncoder(nn.Module):
    def __init__(self, max_freq=1000, num_freqs=10, d=3): #TEMP
        super(FreqEncoder, self).__init__()
        periodics = [torch.sin, torch.cos]
        freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=num_freqs)
        encoder_fns = [lambda x: fn(x*freq)
                       for fn in periodics
                       for freq in freq_bands]
        out_dim = len(encoder_fns)*d
        self.encoder_fns = encoder_fns
        self.out_dim = out_dim
        
    def forward(self, x):
        return torch.cat([fn(x) for fn in self.encoder_fns], -1)
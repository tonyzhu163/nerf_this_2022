import torch
import torch.nn as nn
import numpy as np

class FreqEncoder(nn.Module):
    def __init__(self, max_freq=9, num_freqs=10, d=3):
        super(FreqEncoder, self).__init__()
        periodics = [torch.sin, torch.cos]
        freq_bands = 2.**torch.linspace(0., max_freq, steps=num_freqs)
        encoder_fns = [lambda x, fn=fn, freq=freq: fn(x*freq)
                       for freq in freq_bands
                       for fn in periodics]
        out_dim = len(encoder_fns)*d
        self.encoder_fns = encoder_fns
        self.out_dim = out_dim
        
    def forward(self, x):
        return torch.cat([fn(x) for fn in self.encoder_fns], -1)


def get_embedder(multries, args, i_embed = 0):
    if i_embed == 0:
        embedder_obj = FreqEncoder(multries-1,multries)
        embed = lambda x, eo=embedder_obj : eo.forward(x)
        out_dim = embedder_obj.out_dim
    
    return embed, out_dim

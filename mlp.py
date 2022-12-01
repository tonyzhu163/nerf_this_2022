import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, posi_len=60, dir_len=24, skips=[4]):
        '''
        D specifies the number of hidder layers.
        W specifies the output shape of hidden layers.
        posi_len specifies the size of the position encoding vector.
        dir_len specifies the size of the direction encoding vector.
        Input of the forward function should be the concatenation of the position 
        encoding vector and the direction encoding vector.
        Thus, input of the forward function should be of size (*, posi_len + dir_len).
        skips specifies after which hidden layer(s), the position encoding vector should
        be concatenated with the output of the last hidden layer, and the concatenated         
        vector should be the input of the next hidden layer.
        (actually, if skips=4, then the concatenatation happens after the 5th layer.)
        '''
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.posi_len = posi_len
        self.dir_len = dir_len
        self.skips = skips

        self.hidden_layers = nn.ModuleList([nn.Linear(posi_len, W)])
        for i in range(D-1):
            if i in self.skips:
                self.hidden_layers.append([nn.Linear(W + posi_len, W)])
            else:
                self.hidden_layers.append([nn.Linear(W, W)])
        
        self.density_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.view_linear = nn.Linear(dir_len + W, W//2)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x):
        posi_encode, dir_encode = torch.split(x, [self.posi_len, self.dir_len], dim=-1)
        h = posi_encode
        for i,l in enumerate(self.hidden_layers):
            h = self.hidden_layers[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([posi_encode, h], -1)
        
        density = self.density_linear(h)
        ## density = F.relu(density)
        h = self.feature_linear(h)
        h = torch.cat([h, dir_encode], -1)
        h = self.view_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, density], -1)
        return outputs

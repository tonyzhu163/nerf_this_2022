import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders import get_embedder
import os

## We always assume that our model uses 5d inputs, so we will not use
## the "use_viewdir" argument as in the codes we reference.


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
                self.hidden_layers.append(nn.Linear(W + posi_len, W))
            else:
                self.hidden_layers.append(nn.Linear(W, W))
        
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
        h = self.feature_linear(h)
        h = torch.cat([h, dir_encode], -1)
        h = self.view_linear(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, density], -1)
        ## The paper says there should be a sigmoid for rgb and a relu for density,
        ## but those are implemented in raw2outputs
        return outputs

    def load_weights_from_keras(self, weights):
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.hidden_layers[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.hidden_layers[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_density_linear = 2 * self.D
        self.density_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_density_linear]))
        self.density_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_density_linear+1]))

        # Load views_linears
        idx_feature_linear = 2 * self.D + 2
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load rgb_linear
        idx_view_linear = 2 * self.D + 4
        self.view_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_view_linear]))
        self.view_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_view_linear+1]))

        # Load alpha_linear
        idx_rgb_linear = 2 * self.D + 6
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rgb_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rgb_linear+1]))
        

def chunkify(fn, chunk_sz):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk_sz != None:
        def ret(inputs):
            return torch.cat([fn(inputs[i:i+chunk_sz]) for i in range(0, inputs.shape[0], chunk_sz)], 0)
        return ret
    else:
        return fn


def run_network(inputs, viewdirs, network_fn, embed_fn_pos, embed_fn_dir, point_chunk_sz=1024*64, **kwargs):
    '''
    inputs: positions of points, dimension: x * y * 3
    (x: number of rays. y: number of points per ray.)
    viewdirs: directions of points, dimension: x * y
    (all points on the same ray are of the same direction)
    outputs: x * y * 4 (4: 3 rgb + 1 density)
    '''
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded_pos = embed_fn_pos(inputs_flat)

    input_dirs = viewdirs[:,None].expand(inputs.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embed_fn_dir(input_dirs_flat)
    embedded = torch.cat([embedded_pos, embedded_dirs], -1)

    outputs_flat = chunkify(network_fn, point_chunk_sz)(embedded)
    output_shape = list(inputs.shape[:-1])
    output_shape.append(outputs_flat.shape[-1])
    outputs = torch.reshape(outputs_flat,output_shape)
    return outputs

    

def create_nerf(args, device):
    embed_fn_pos, input_ch_pos = get_embedder(args.multires, args, i_embed=args.i_embed)
    ### if args.i_embed==1:
    embed_fn_dir, input_ch_dir = get_embedder(args.multires_views, args, i_embed=args.i_embed_views)
    skips = [4]
    if args.i_embed==1:
        ## instant NGP
        pass
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                posi_len=input_ch_pos, dir_len=input_ch_dir, skips=skips).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            ##
            pass
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                posi_len=input_ch_pos, dir_len=input_ch_dir, skips=skips).to(device)
        grad_vars += list(model_fine.parameters())
    
    #################### need to modify
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                        embed_fn_pos=embed_fn_pos, embed_fn_dir=embed_fn_dir, point_chunk_sz=args.point_chunk_sz)

    if args.i_embed==1:
        ##
        pass
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    
    start = 0
    
    #TEMP: add these back when implemented properly (also add to params)
    savedir = args.savedir
    object = args.object
    
    ##########################

    if not args.no_reload:
        sample_mode = 'all' if args.use_batching else 'single'
        coarse_fine = "coarse" if args.N_importance<=0 else "fine"
        # Load checkpoints
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join("..", *savedir, "weights", coarse_fine, sample_mode, object, f) for f in sorted(os.listdir(os.path.join("..", *savedir, "weights", coarse_fine, sample_mode, object))) if 'tar' in f]

        print('Found ckpts', [c[-10:] for c in ckpts])
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            '''
            if args.i_embed==1:
                embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])
            '''

    ##########################
    
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn_dir': embed_fn_dir,
        'embed_fn_pos': embed_fn_pos,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


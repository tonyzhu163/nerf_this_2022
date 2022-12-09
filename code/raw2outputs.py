import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    '''
    input
    raw: [N_rays, N_samples, 4], output of the mlp model.
    z_vals: [N_rays, N_samples]. Integration time t. (Recall r(t)=o+td)
    rays_d: [num_rays, 3]. Direction of each ray.

    output
    rgb_map: [N_rays, 3]. Estimated RGB color of a ray.
    disp_map: [N_rays]. Disparity map. Inverse of depth map.
    acc_map: [N_rays]. Sum of weights along each ray.
    weights: [N_rays, N_samples]. Weights assigned to each sampled color.
    depth_map: [N_rays]. Estimated distance to object.
    '''

    dists = z_vals[...,1:] - z_vals[...,:-1]
    temp = torch.Tensor([1e10]).expand(dists[...,0:1].shape)
    dists = torch.cat([dists,temp],-1)
    dir_norms = torch.norm(rays_d[...,None,:], dim=-1)
    dists = dists * dir_norms    # [N_rays, N_samples]

    rgb = torch.sigmoid(raw[...,:3])    # [N_rays, N_samples, 3]

    density = raw[...,3]    # [N_rays, N_samples]
    noise = 0.
    if raw_noise_std > 0.:
        if not pytest:
            noise = torch.randn(density.shape) * raw_noise_std
        else:
            np.random.seed(0)
            noise = np.random.rand(*list(density.shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    density = F.relu(density + noise)    # [N_rays, N_samples]

    exps = torch.exp(-density*dists)    # [N_rays, N_samples]
    alpha = 1.0 - exps
    temp = torch.cat([torch.ones((exps.shape[0], 1)), exps + 1e-10], -1)
    T = torch.cumprod(temp,-1)[:, :-1]    # [N_rays, N_samples]
    weights = alpha * T    # [N_rays, N_samples]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)    # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    temp = 1e-10 * torch.ones_like(depth_map)
    disp_map = 1./torch.max(temp, depth_map / torch.sum(weights, -1))

    if white_bkgd:
        rgb_map = rgb_map + (1.0-acc_map[...,None])

    ## sparsity loss

    return rgb_map, disp_map, acc_map, weights, depth_map
    

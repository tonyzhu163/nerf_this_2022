import torch
import numpy as np

def generate_rays(H, W, K, pose):
    focal = K[0][0]
    
    x, y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    x, y = x.t(), y.t()

    # equation from https://www.scratchapixel.com/lessons/3d-basic-rendering
    # /ray-tracing-generating-camera-rays/generating-camera-rays
    # may or may not need 0.5 shift

    dx = (x + 0.5 - W/2) / focal
    dy = -(y + 0.5 - H/2) / focal
    dir = torch.stack([dx, dy, -torch.ones_like(x)], -1)

    ray_d = dir @ pose[:3, :3].T
    ray_d = ray_d / (torch.norm(ray_d, dim=-1, keepdim=True) + 1e-10)
    ray_o = pose[:3, -1].expand(ray_d.shape)

    ray_d = ray_d.view(-1, 3)
    ray_o = ray_o.view(-1, 3)

    # normalized direction and origin
    # both [H*W, 3]
    return ray_d, ray_o


def render(H, W, K, max_rays, rays, near, far, pose=None):
    # section 4 in NERF
    if pose is not None:
        generate_rays(H, W, K, pose)
    else:
        ray_d, ray_o = rays

    # batchify_ray -> render_ray -> h_sampling

def h_sampling(bins, weights, n, det=False, tol=1e-5):
    # section 5.2 in NERF
    n_rays, n_samples = weights.shape
    weights = weights + tol
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0, 1, n)
        u = u.expand(n_rays, n)
    else:
        u = torch.rand(n_rays, n)
    u = u.contiguous()

    # invert CDF by finding domain for each value
    idx = torch.searchsorted(cdf, u, right=True)
    lower = torch.clamp_min(idx-1, 0)
    upper = torch.clamp_max(idx, n_samples)

    i_s = torch.stack([lower, upper], -1).view(n_rays, 2*n)
    cdf_g = torch.gather(cdf, 1, i_s).view(n_rays, n, 2)
    bins_g = torch.gather(bins, 1, i_s).view(n_rays, n, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < tol] = 1

    return bins_g[..., 0] + (u-cdf_g[..., 0])/denom * (bins_g[..., 1] - bins_g[..., 0])







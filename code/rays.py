import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_rays(H, W, K, pose):
    focal = K[0][0]
    x, y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    x, y = x.t(), y.t()

    x = x.to(device)
    y = y.to(device)
    # equation from https://www.scratchapixel.com/lessons/3d-basic-rendering
    # /ray-tracing-generating-camera-rays/generating-camera-rays
    # may or may not need 0.5 shift

    dx = (x + 0.5 - W/2) / focal
    dy = -(y + 0.5 - H/2) / focal
    dirs = torch.stack([dx, dy, -torch.ones_like(x)], -1)

    ray_d = dirs @ pose[:3, :3].T
    ray_d = ray_d / (torch.norm(ray_d, dim=-1, keepdim=True) + 1e-10)
    ray_o = pose[:3, -1].expand(ray_d.shape)

    ray_d = ray_d.view(-1, 3)
    ray_o = ray_o.view(-1, 3)

    # normalized direction and origin
    # both [H*W, 3]
    return ray_d, ray_o


def h_sampling(bins, weights, n, device, det=False, tol=1e-5):
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
    u = u.to(device)

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


def sample_coarse(z_vals, z_vals_mid, rays_o, rays_d, perturb, device):
    if perturb > 0.:
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
        perturb_rand = perturb * torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * perturb_rand

    pts_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    return pts_coarse_sampled


def sample_fine(z_vals, z_vals_mid, rays_o, rays_d, weights, n_importance, perturb, device):
    z_samples = h_sampling(z_vals_mid, weights[:, 1:-1], n_importance, device, det=(perturb == 0)).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    pts_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    return pts_fine_sampled, z_samples, z_vals













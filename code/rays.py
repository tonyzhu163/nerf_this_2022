import torch
import numpy as np


def generate_rays(H, W, focal, pose):
    
    x, y = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    x, y = x.t(), y.t()

    # equation from https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays

    dx = (x + 0.5 -W/2)/focal
    dy = -(y + 0.5 -H/2)/focal
    dir = torch.stack([dx, dy, -torch.ones_like(x)], -1)
    
    ray_d = ray_d / (torch.norm(dir, dim=-1, keepdim=True) + 1e-10)
    ray_d = torch.sum(ray_d[..., np.new_axis, :] * pose[:3, :3], -1)
    ray_o = pose[:3, -1].expand(ray_d.shape)

    return ray_d, ray_o


def render(H, W, focal, max_rays, rays, pose, near, far):

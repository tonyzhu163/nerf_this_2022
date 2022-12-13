import torch
from render import render

def validate(H, W, K, ray_chunk_sz, rays_val, target_rgb_val, device, **render_kwargs_train):
    img2mse = lambda x, y: torch.mean((x - y) ** 2)
    mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
    render_outputs_val, extras_val = render(
            H, W, K, ray_chunk_sz, rays_val, device, **render_kwargs_train
        )
    rgb_val, disp_val, acc_val = render_outputs_val

    loss_val = img2mse(rgb_val, target_rgb_val)  # * mean squared error as tensor
    psnr_val = mse2psnr(loss_val)  # * peak signal-to-noise ratio as tensor
    psnr0_val = None
    if "rgb0" in extras_val:
            loss0_val = img2mse(extras_val["rgb0"],target_rgb_val )
            loss_val = loss_val + loss0_val
            psnr0_val = mse2psnr(loss0_val)
    
    return loss_val, psnr_val, psnr0_val
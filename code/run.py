import os
from pathlib import Path

import torch
from tqdm import tqdm, trange

from load_blender import load_blender_data
from render import render
from params import get_params
from generate_output import generate_output
from model import create_nerf
from batching import BatchedRayLoader


img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
# TODO: why can't we use torch here
# to8b = lambda x : (255*clip(x,0,1)).astype(uint8)
# TODO: why can't we use torch here
to8b = lambda x: (255 * torch.clip(x, 0, 1)).astype(torch.uint8)


def update_lr(params, optimizer, global_step):
    decay_rate = 0.1
    decay_steps = params.lrate_decay * 1000
    new_lrate = params.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate


def main():
    params = get_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(Path.cwd().parent, *params.datadir, params.object)
    #TODO: right now the images outputted are 3 channel: RGB. However blender
    #TODO: images actually have a opacity layer. We could improve the model by
    #TODO: comparing density output with opacity from source image
    images, poses, render_poses, [H, W, F, K], near, far, i_split = load_blender_data(data_dir, params.white_bkgd)
    print("Loaded blender", images.shape, render_poses.shape, H, W, F, K, data_dir)
    i_train, i_val, i_test = i_split

    render_poses = torch.Tensor(render_poses).to(device)

    # TODO: can we attach optimizer to render_kwargs or smth damn
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        params,
        device=device,
    )
    global_step = start

    for kwargs in [render_kwargs_train, render_kwargs_test]:
        kwargs["near"] = near
        kwargs["far"] = far

    # Short circuit if only rendering out from trained model
    if params.render_only:
        #TODO
        # generate_output()
        return

    # ------------------------------------------------------------------------ #
    #                          Main Training Loop                              #
    # ------------------------------------------------------------------------ #
    
    # batches and creates rays from poses
    dataloader = BatchedRayLoader(images, poses, i_train, H, W, K, device, params, sample_mode='single')
    
    for epoch in trange(start + 1, params.epochs + 1):
        # ---- Forward Pass (Sampling, MLP, Volumetric Rendering) ------------ #
        
        rays, target_rgb = dataloader.get_rays()
        
        #TODO: could probably clean up the function call parameters
        #TODO: switch render form temp_code to rays.py
        rgb, disp, acc, extras = render(
            H, W, K, params.ray_chunk_sz, rays, **render_kwargs_train
        )
        optimizer.zero_grad()
        #rgb: (4096,3), target_rgb (4096, 4)
        loss = torch.mean((rgb - target_rgb) ** 2)  # * mean squared error as tensor
        psnr = mse2psnr(loss)  # * peak signal-to-noise ratio as tensor

        loss.backward()
        optimizer.step()

        update_lr(params, optimizer, global_step)

        # --------------- Saving Model Output / Weights ---------------------- #
        #TODO
        if epoch % params.i_weights == 0:
            pass
        if epoch % params.i_video == 0 and epoch > 0:
            pass
        if epoch % params.i_testset == 0 and epoch > 0:
            pass

        if epoch % params.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {epoch} Loss: {loss.item()}  PSNR: {psnr.item()}"
            )
        global_step += 1


if __name__ == "__main__":
    main()

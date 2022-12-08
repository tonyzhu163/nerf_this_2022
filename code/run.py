import os
from pathlib import Path

import torch
from tqdm import tqdm, trange

from load_blender import load_blender_data
from rays import generate_rays, render
from params import get_params
from generate_output import generate_output
from numpy import clip, uint8

def create_nerf():
    pass  # placeholder for create_nerf in model.py

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
# to8b = lambda x : (255*clip(x,0,1)).astype(uint8) #TODO: why can't we use torch here
to8b = lambda x : (255*torch.clip(x,0,1)).astype(torch.uint8) #TODO: why can't we use torch here



def update_lr(params, optimizer, global_step):
    decay_rate = 0.1
    decay_steps = params.lrate_decay * 1000
    new_lrate = params.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    
def main():
    params = get_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(Path.cwd().parent, *params.datadir)
    images, poses, render_poses, [H, W, F, K], near, far, i_split = load_blender_data(
        data_dir
    )
    print("Loaded blender", images.shape, render_poses.shape, H, W, F, K, data_dir)
    i_train, i_val, i_test = i_split

    # Short circuit if only rendering out from trained model
    if params.render_only:
        # generate_output()
        return

    
if __name__ == "__main__":
    main()
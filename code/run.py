import os
from pathlib import Path

import torch

from load_blender import load_blender_data
from rays import generate_rays, render
from params import get_params
from generate_output import generate_output
def main():
    params = get_params()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = Path.cwd().parent / params.datadir
    images, poses, render_poses, hwf, K, i_split = load_blender_data(data_dir)
    print('Loaded blender', images.shape, render_poses.shape, hwf, data_dir)
    i_train, i_val, i_test = i_split

    # Short circuit if only rendering out from trained model
    if params.render_only:
        # generate_output()
        return

    
if __name__ == "__main__":
    main()
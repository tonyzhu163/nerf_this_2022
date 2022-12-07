import os
from pathlib import Path

from .load_blender import load_blender_data
from .rays import generate_rays, render

def main():
    data_dir = Path.cwd() / '_data' / 'nerf_synthetic'
    images, poses, render_poses, hwf, i_split = load_blender_data(data_dir)
    print('Loaded blender', images.shape, render_poses.shape, hwf, data_dir)
    i_train, i_val, i_test = i_split

    # 

if __name__ == "__main__":
    main()
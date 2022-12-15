import os
import pickle
import imageio
import numpy as np
import torch
from tqdm import tqdm
from params import ModelParameters
from rays import generate_rays
from render import render


def generate_output(H, W, K, Rs, test_imgs, start, device, params: ModelParameters, **render_kwargs) -> None:
    
    print('SET TO RENDER ONLY')
    print('Rendering now...')
    with torch.no_grad():
        
        images = None
        if params.render_test:
            images = test_imgs
        
        savedir = params.savedir
        expname = params.expname
        outputdir = os.path.join('..', *savedir, 'experiments', expname, \
            f"{'test' if params.render_test else 'path'}_output_{start}")
        
        os.makedirs(outputdir, exist_ok=True)
        
        print('test poses shape', Rs.shape)

        rgbs, *_ = render_path(H, W, K, Rs, params.ray_chunk_sz, device, gt_imgs=images, render_factor=params.render_factor, **render_kwargs)
        
        
        print('Done rendering', outputdir)
        settings = {
            'macro_block_size': None,
            'fps':30,
            'quality':8
        }
        imageio.mimwrite(os.path.join(outputdir, 'video.mp4'), to8b(rgbs), **settings)
        return


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def render_path(H, W, K, Rs, ray_chunk_sz, device, gt_imgs=None, render_factor=1, **render_kwargs):
    H, W = int(H//render_factor), int(W//render_factor)

    rgbs = []
    disps = []
    psnrs = []
    

    print(f"generating rgbs for {Rs.shape[0]} poses")
    for i, R in tqdm(enumerate(Rs)):
        rays = generate_rays(H, W, K, R[:3,:4])
        (rgb, disp, acc), extras = render(H, W, K, ray_chunk_sz, rays, device, **render_kwargs)
        rgb = rgb.reshape((H,W,3))
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0: print(f"render output shapes | rgb: {rgb.shape}, disp: {disp.shape}")

        if gt_imgs is not None and render_factor==1:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            # print(p)
            psnrs.append(p)
        
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    avg_psnr = None
    if gt_imgs is not None and render_factor==1:
        avg_psnr = sum(psnrs)/len(psnrs)

    return rgbs, disps, avg_psnr
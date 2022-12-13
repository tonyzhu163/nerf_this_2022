import torch
import numpy as np
from pathlib import Path
from model import NeRF
from batching import BatchedRayLoader
from run import img2mse, mse2psnr
from render import render


# test code put in run
# -1 turns always turns off cropping

# total test size is test_size * repeat_test
def test_weights(network_fn: NeRF, network_fine: NeRF, optimizer, test_size, n, weights_path, test_loader: BatchedRayLoader
                 , ray_chunk_sz, device, H, W, K, i_test, test_all = True, **render_kwargs):
    weights = []
    ret = {}
    epochs = []

    p = Path(weights_path)

    for x in p.rglob('*.tar'):
        weights.append(x)

    loss_lst = []
    psnr_lst = []
    psnr0_lst = []

    for w in weights:
        epoch = int(w.stem)
        epochs.append(epoch)

        ckpt = torch.load(w)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        network_fn.load_state_dict(ckpt['network_fn_state_dict'])
        if network_fine is not None:
            network_fine.load_state_dict(ckpt['network_fine_state_dict'])

        loss_avg = []
        loss_0 = []
        loss_1 = []

        if test_all:
            ns = i_test
        else:
            ns = np.random.choice(i_test, n, replace=False)

        for x in range(ns):
            rays, target_rgb = test_loader.get_sample(test_size, x)
            render_outputs, extras = render(
                H, W, K, ray_chunk_sz, rays, device, **render_kwargs
            )
            rgb, disp, acc = render_outputs

            loss = img2mse(rgb, target_rgb)  # * mean squared error as tensor
            psnr = mse2psnr(loss)

            loss_1.append(loss)

            if "rgb0" in extras:
                loss0 = img2mse(extras["rgb0"], target_rgb)
                loss = loss + loss0

            loss_avg.append(loss)
            loss_0.append(loss0)


        loss_lst.append(np.mean(loss_avg))
        psnr_lst.append(mse2psnr(np.mean(loss_1)))
        psnr0_lst.append(mse2psnr(np.mean(loss_0)))

    ret['loss'] = loss_lst
    ret['psnr'] = psnr_lst
    ret['psnr0'] = psnr0_lst
    ret['epoch'] = epochs
    return ret













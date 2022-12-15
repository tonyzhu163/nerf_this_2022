import torch
import numpy as np
from pathlib import Path

from tqdm import tqdm
from model import NeRF
from batching import BatchedRayLoader
from render import render

# test code put in run
# -1 turns always turns off cropping

# total test size is test_size * repeat_test

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))


def test_weights(writer, test_size, n:int, weights_path, test_loader: BatchedRayLoader,
                 ray_chunk_sz, device, H, W, K, i_test, network_fn: NeRF, network_fine: NeRF, test_all=True,
                 ignore_before: int = 0,
                 **render_kwargs):

    assert network_fine and network_fn is not None

    weights = []
    ret = {}
    epochs = []

    p = Path(weights_path)

    for x in p.rglob('*.tar'):
        weights.append(x)

    loss_lst = []
    psnr_lst = []
    psnr0_lst = []

    for w in tqdm(weights):
        epoch = int(w.stem)

        if epoch < ignore_before:
            continue

        epochs.append(epoch)

        ckpt = torch.load(w)

        # Load model
        network_fn.load_state_dict(ckpt['network_fn_state_dict'])
        if network_fine is not None:
            network_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
        render_kwargs["network_fn"] = network_fn
        render_kwargs["network_fine"] = network_fine

        loss_avg = []
        loss_0 = []
        loss_1 = []

        if test_all:
            ns = i_test
        else:
            ns = np.random.choice(i_test, n, replace=False)

        for x in ns:
            rays, target_rgb = test_loader.get_sample(test_size, x)
            render_outputs, extras = render(
                H, W, K, ray_chunk_sz, rays, device, **render_kwargs
            )
            rgb, disp, acc = render_outputs

            loss = img2mse(rgb, target_rgb)  # * mean squared error as tensor

            loss_1.append(loss)

            if "rgb0" in extras:
                loss0 = img2mse(extras["rgb0"], target_rgb)
                loss = loss + loss0

            loss_avg.append(loss)
            loss_0.append(loss0)
        loss_write = torch.mean(torch.Tensor(loss_avg)).item()
        loss_lst.append(loss_write)
        psnr_write = mse2psnr(torch.mean(torch.Tensor(loss_1))).item()
        psnr_lst.append(psnr_write)
        psnr0_lst.append(mse2psnr(torch.mean(torch.Tensor(loss_0))).item())
        writer.add_scalar('Loss/test', loss_write, epoch)
        writer.add_scalar('PSNR/test', psnr_write, epoch)

    ret['loss'] = loss_lst
    ret['psnr'] = psnr_lst
    ret['psnr0'] = psnr0_lst
    ret['epoch'] = epochs
    return ret













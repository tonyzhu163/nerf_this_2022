import torch
import numpy as np
from pathlib import Path
from model import NeRF
from batching import BatchedRayLoader
from run import img2mse, mse2psnr

# test code put in run
# -1 turns always turns off cropping
test_loader = BatchedRayLoader(images, poses, i_test, H, W, K, device, params, sample_mode='single', start=-1)

# total test size is test_size * repeat_test
def test_weights(model: NeRF, model_fine: NeRF, test_size, repeat_test, weights_path, test_loader: BatchedRayLoader):
    weights = []
    ret = {}
    epochs = []

    p = Path(weights_path)

    for x in p.rglob('*.tar'):
        weights.append(x)

    loss_lst_f = []
    psnr_lst_f = []
    psnr0_lst_f = []

    for w in weights:
        psnr0 = None
        epoch = int(w.stem())
        epochs.append(epoch)

        # ckpt = torch.load(w)
        # start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #
        # # Load model
        # model.load_state_dict(ckpt['network_fn_state_dict'])
        # if model_fine is not None:
        #     model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        loss_lst = []
        psnr_lst = []
        psnr0_lst = []

        for n in range(repeat_test):

            rays, target_rgb = test_loader.get_sample(test_size)
            render_outputs, extras = render(
                H, W, K, params.ray_chunk_sz, rays, device, **render_kwargs_train
            )
            rgb, disp, acc = render_outputs

            loss = img2mse(rgb, target_rgb)  # * mean squared error as tensor
            psnr = mse2psnr(loss)

            if "rgb0" in extras:
                loss0 = img2mse(extras["rgb0"], target_rgb)
                loss = loss + loss0
                psnr0 = mse2psnr(loss0)

            loss_lst.append(loss.item())
            psnr_lst.append(psnr.item())

            if psnr0:
                psnr0_lst.append(psnr0.item())

        loss_lst_f.append(np.mean(loss_lst))
        psnr_lst_f.append(np.mean(psnr.avg))
        psnr0_lst_f.append(np.mean(psnr.avg))

    ret['loss'] = loss_lst_f
    ret['psnr'] = psnr_lst_f
    ret['psnr0'] = psnr0_lst_f
    ret['epoch'] = epochs
    return ret













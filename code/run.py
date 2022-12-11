import os
from pathlib import Path
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
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
    ########## drums ##########
    params.use_batching = False
    params.white_bkgd = True
    params.lrate_decay = 500
    params.N_samples = 64
    params.N_importance = 128
    params.ray_batch_sz = 1024
    params.precrop_iters = 500
    params.precrop_frac = 0.5
    params.half_res = True

    params.i_weights = 1000
    params.epochs = 150000
    ###########################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(Path.cwd().parent, *params.datadir, params.object)
    #TODO: right now the images outputted are 3 channel: RGB. However blender
    #TODO: images actually have a opacity layer. We could improve the model by
    #TODO: comparing density output with opacity from source image
    images, poses, render_poses, [H, W, F, K], near, far, i_split = \
        load_blender_data(data_dir, params.half_res, params.white_bkgd)
    print("Loaded blender", images.shape, render_poses.shape, H, W, F, K, data_dir)
    i_train, i_val, i_test = i_split

    render_poses = torch.Tensor(render_poses).to(device)

    # TODO: can we attach optimizer to render_kwargs or smth damn

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        params,
        device=device,
    )

    for kwargs in [render_kwargs_train, render_kwargs_test]:
        kwargs["near"] = near
        kwargs["far"] = far

    # Short circuit if only rendering out from trained model
    if params.render_only:
        test_imgs = images[i_test]
        Rs = torch.Tensor(poses[i_test]).to(device)
        generate_output(H, W, K, Rs, test_imgs, start, params, **render_kwargs_test)
        return

    # ------------------------------------------------------------------------ #
    #                          Main Training Loop                              #
    # ------------------------------------------------------------------------ #
    
    # batches and creates rays from poses
    sample_mode = 'all' if params.use_batching else 'single'
    dataloader = BatchedRayLoader(images, poses, i_train, H, W, K, device, params, sample_mode, start = start)
    coarse_fine = "coarse" if params.N_importance<=0 else "fine"

    time = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.now())
    tb_path = Path.cwd().parent / 'logs' / 'tensorboard' / time
    writer = SummaryWriter(log_dir=f'{tb_path}')

    for global_step in trange(start + 1, params.epochs + 1):
        # ---- Forward Pass (Sampling, MLP, Volumetric Rendering) ------------ #
        
        rays, target_rgb = dataloader.get_sample()
        
        #TODO: could probably clean up the function call parameters
        render_outputs, extras = render(
            H, W, K, params.ray_chunk_sz, rays, device, **render_kwargs_train
        )
        rgb, disp, acc = render_outputs
        optimizer.zero_grad()
        #rgb: (4096,3), target_rgb (4096, 4)

        # print(rgb.is_cuda)
        # print(target_rgb.is_cuda)

        # --- loss calculation normally done on cpu ---

        loss = torch.mean((rgb - target_rgb) ** 2)  # * mean squared error as tensor
        psnr = mse2psnr(loss)  # * peak signal-to-noise ratio as tensor
        if "rgb0" in extras:
            loss0 = torch.mean((extras["rgb0"] - target_rgb) ** 2)
            loss = loss + loss0
            psnr0 = mse2psnr(loss0)

        loss.backward()
        optimizer.step()

        update_lr(params, optimizer, global_step)

        # --------------- Saving Model Output / Weights ---------------------- #
        #TODO
        if global_step % params.i_weights == 0:
            folder_path = os.path.join("..", *params.savedir, "weights", coarse_fine, sample_mode, params.object)
            #  check if dir exists, if not create it
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, '{:06d}.tar'.format(global_step))
            if params.i_embed==1:
                #TODO: why should this ever be 1?
                pass
            else:
                if render_kwargs_train['network_fine'] != None:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, file_path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, file_path)
                print(f'Saved checkpoints for step {global_step} at', file_path)
        '''
        if global_step % params.i_video == 0 and global_step > 0:
            #TODO: unfinished, testing.
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
        '''
        if global_step % params.i_testset == 0 and global_step > 0:
            pass

        if global_step % params.i_print == 0:
            if params.N_importance>0:
                tqdm.write(
                    f"[TRAIN] Iter: {global_step} Loss: {loss.item()}  Fine PSNR: {psnr.item()} Coase PSNR: {psnr0.item()}" 
                )
            else:
                tqdm.write(
                    f"[TRAIN] Iter: {global_step} Loss: {loss.item()}  Coase PSNR: {psnr0.item()}" 
                )


        # --- DRAW ---

        if params.tensorboard:
            writer.add_scalar('Loss/train', loss, global_step)
            # writer.add_scalar('Loss/test', np.random.random(), global_step)
            writer.add_scalar('PSNR/train', psnr, global_step)
            # writer.add_scalar('PSNR/test', np.random.random(), global_step)
            if params.N_importance>0:
                writer.add_scalar("PSNR0/train", psnr0, global_step)


if __name__ == "__main__":
    main()

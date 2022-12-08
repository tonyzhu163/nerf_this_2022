import imageio
import torch

#TODO: This entire class needs to be refactored

def generate_output(params):
    print('SET TO RENDER ONLY')
    print('Rendering now...')
    with torch.no_grad():
        if params.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if params.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _ = render_path(render_poses, hwf, K, params.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=params.render_factor)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

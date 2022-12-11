import torch

from collections import defaultdict
from rays import generate_rays, sample_coarse, sample_fine
from raw2outputs import raw2outputs
from model import run_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render(H, W, K, ray_chunk_sz, rays, device, near, far, **kwargs):

    # section 4 in NERF

    # --- Prepare Input ---
    # when in testing, rays will represent every pixel of the test image set.
    # when in training, rays will be a small randomly sampled subset of the training data.
    rays_d, rays_o = rays

    sh = rays_d.shape 
    rays_o = rays_o.float()
    rays_d = rays_d.float()

    # near and far from ** kwargs?
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # --- Input Rays ---
    # rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # --- Render Ray in Chunks ---
    records = defaultdict(list)
    for i in range(0, rays.shape[0], ray_chunk_sz):
        ret = render_ray(rays[i:i + ray_chunk_sz, :], device=device, **kwargs)
        for k in ret:
            records[k].append(ret[k])

    records = {k: torch.cat(records[k], 0) for k in records}
    # why do we need to reshape here?

    for k in records:
        k_sh = list(sh[:-1]) + list(records[k].shape[1:])
        records[k] = torch.reshape(records[k], k_sh)

    inf = ['rgb_map', 'disp_map', 'acc_map']
    output = [records[i] for i in inf]
    extra = {i: records[i] for i in records if i not in inf}
    return output, extra


#TODO: remove ray_batch_sz if not used
def render_ray(rays, N_samples, device,
                network_fn, network_fine=None,
               N_importance=0, perturb=0, raw_noise_std =0., white_bkgd=False, **kwargs):
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]

    z_steps = torch.linspace(0, 1, N_samples)

    z_steps = z_steps.to(device)

    z_vals = near * (1 - z_steps) + far * z_steps
    z_vals = z_vals.expand(n_rays, N_samples)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])

    pts_coarse, perturbed_values = sample_coarse(z_vals, z_vals_mid, rays_o, rays_d, perturb, device)

    raw = run_network(pts_coarse, rays_d, network_fn, **kwargs)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, perturbed_values, rays_d, device,
                                                                 raw_noise_std, white_bkgd, False)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        pts_fine, z_samples, hiearchical_values = sample_fine(z_vals, z_vals_mid, rays_o, rays_d, weights, N_importance, perturb, device)

        run_fn = network_fn if network_fine is None else network_fine

        raw = run_network(pts_fine, rays_d, run_fn, **kwargs)
        rgb_map, disp_map, acc_map, _, _ = raw2outputs(raw, hiearchical_values, rays_d, device,
                                                       raw_noise_std, white_bkgd, False)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret

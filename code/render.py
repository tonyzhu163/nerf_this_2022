import torch

from collections import defaultdict
from rays import generate_rays, sample_coarse, sample_fine
from raw2outputs import raw2outputs
from model import run_network


def render(H, W, K, ray_chunk_sz, rays, near, far, pose=None, **kwargs):
    # section 4 in NERF

    # --- Prepare Input ---
    # test pose is provided for render_path (testing)
    if pose is not None:
        rays_d, rays_o = generate_rays(H, W, K, pose)
    else:
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
        ret = render_ray(rays[i:i + ray_chunk_sz, :], **kwargs)
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
def render_ray(rays, N_samples, n_importance=0, perturb=0, ray_batch_sz=1024*32, **kwargs):
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]

    z_steps = torch.linspace(0, 1, N_samples)
    z_vals = near * (1 - z_steps) + far * z_steps
    z_vals = z_vals.expand(n_rays, N_samples)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])

    pts_coarse = sample_coarse(z_vals, z_vals_mid, rays_o, rays_d, perturb)

    # --- NOT DONE ---
    raw = run_network(pts_coarse, rays_d, **kwargs)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, )

    if n_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        pts_fine, z_samples = sample_fine(z_vals, z_vals_mid, rays_o, rays_d, weights, n_importance, perturb)

        # --- NOT DONE ---
        raw = run_network(pts_fine, rays_d, **kwargs)
        rgb_map, disp_map, acc_map, _, _ = raw2outputs(raw, )

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if n_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    return ret

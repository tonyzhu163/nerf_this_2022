import torch

from rays import generate_rays, sample_coarse, sample_fine


def render(H, W, K, max_rays, rays, near, far, pose=None, **kwargs):
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

    # [H*W, 3 + 3 + 1 + 1]
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # --- Input Rays ---
    # rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
    # feel like we are repeating data here so this isn't neccesary
    # viewdirs = rays_d
    # rays = torch.cat([rays, viewdirs], -1)

    # --- OUTPUT DEPENDS ON BATCHIFY RAY ---

    return


def batchify_ray(rays_flat, chunk=1024*32, **kwargs):
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_ray(rays_flat[i:i + chunk], **kwargs)

    # --- OUTPUT DEPENDS ON RENDER_RAY ---

    return


def render_ray(rays, n_samples, n_importance=0, perturb=0, chunk=1024*32):
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]

    z_steps = torch.linspace(0, 1, n_samples)
    z_vals = near * (1 - z_steps) + far * z_steps
    z_vals = z_vals.expand(n_rays, n_samples)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])

    pts_coarse = sample_coarse(z_vals, z_vals_mid, rays_o, rays_d, perturb)

    if n_importance > 0:
        pts_fine = sample_fine(z_vals, z_vals_mid, rays_o, rays_d, weights, n_importance, perturb)


    return

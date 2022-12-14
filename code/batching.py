"""
Batching wrapper for rays
"""
import numpy as np
import torch
from params import ModelParameters

from rays import generate_rays
from get_device import device

# ---------------------------------------------------------------------------- #
#                             Ray Helper Functions                             #
# ---------------------------------------------------------------------------- #

#TODO: REFACTOR
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

# ---------------------------------------------------------------------------- #
#                                Main DataLoader                               #
# ---------------------------------------------------------------------------- #
class BatchedRayLoader():
    def __init__(self, images, poses, i_train, H, W, K, device, params: ModelParameters, sample_mode, start) -> None:
        """
        Arguments
        - sample_mode str: either 'all' or 'single', determines if each iteration samples from across all images or a single image
        """
        self.device = device
        self.sample_mode = sample_mode
        self.i = start+1
        self.images = images
        self.poses = poses
        self.i_train = i_train
        self.H = H
        self.W = W
        self.K = K
        self.params = params
        assert sample_mode in ['all', 'single']
        if sample_mode == 'all':
            self.all_rays = self.preload_all_rays()
            self.current_batch_i = 0
            # self.get_rays_fn = self.rays_from_all
        else:
            self.all_rays = None # not used if single sampling
            self.current_batch_i = None # not used if single sampling
            self.get_rays_fn = self.rays_from_single

    def get_sample(self, sample_size=None, img_i=None):
        """
        samples a random set of rays. sample source depends on if set to
        'all' or 'single'. 
        :return: a tuple batch_rays, target_s of shape (ray_batch_sz, 6) and
        (ray_batch_sz, 3)
        """

        if not sample_size:
            sample_size = self.params.ray_batch_sz

        batch_rays, target_s = self.get_rays_fn(sample_size, img_i)
        # --- TEMP FIX ---
        target_s = target_s.cpu()
        self.i +=1
        return batch_rays, target_s
        
    #TODO: implement
    
    def preload_all_rays(self):
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, self.images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in self.i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        
        # Move training data to GPU
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        return rays_rgb

    # not useable
    # #TODO: refactor further
    # def rays_from_all(self):
    #     ray_batch_sz = self.params.ray_batch_sz
    #     # Random over all images
    #     batch = self.all_rays[self.current_batch_i:self.current_batch_i+ray_batch_sz] # [B, 2+1, 3*?]
    #     batch = torch.transpose(batch, 0, 1)
    #     batch_rays, target_s = batch[:2], batch[2]
    #
    #     self.current_batch_i += ray_batch_sz
    #     if self.current_batch_i >= self.all_rays.shape[0]:
    #         print("Shuffle data after an epoch!")
    #         rand_idx = torch.randperm(self.all_rays.shape[0])
    #         self.all_rays = self.all_rays[rand_idx]
    #         self.current_batch_i = 0
    #     return batch_rays, target_s
    
    
    def get_coords(self, use_precrop=False):
        H, W = self.H, self.W
        if self.i < self.params.precrop_iters:
            dH = int(H//2 * self.params.precrop_frac)
            dW = int(W//2 * self.params.precrop_frac)
            grid = torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                )
            #* they used self.i == start, which is the "start" index for models
            #* restored from a checkpoint. For now we'll just use 0
            #* edit: now self.i = start implemented
            if self.i == 0:
                print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.params.precrop_iters}")
        else:
            #* basically meshgrid creates a cartesian product resulting in the coords of every discrete location in a grid. (0,1), (1,1), (0,2), (1,2) etc.
            grid = torch.meshgrid(
                torch.linspace(0, H-1, H),
                torch.linspace(0, W-1, W)
                )
        coords = torch.stack(grid, -1) # (H, W, 2)
        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        return coords

    
    def rays_from_single(self, sample_size, img_i):
        # randomly select one image from training set
        if not img_i:
            img_i = np.random.choice(self.i_train)
        target_image = self.images[img_i]
        target_image = torch.Tensor(target_image).to(self.device)
        cam_geo = [self.H, self.W]
        cam_geo += [torch.Tensor(self.K).to(self.device)]
        cam_geo += [torch.Tensor(self.poses[img_i, :3, :4]).to(self.device)]
        
        # if self.params.ray_batch_sz: #? why
        # get rays origin and direction from images
        # rays_o, rays_d = get_rays(*cam_geo) # (H,W,3), (H,W,3)
        rays_d, rays_o = generate_rays(*cam_geo)

        use_precrop = self.i < self.params.precrop_iters or self.i == -1
        ## 200 * 200 = 40000
        # TODO: REFACTOR
        coords = self.get_coords(use_precrop)
        # choose a random selection of rays of size ray_batch_sz
        select_inds = np.random.choice(coords.shape[0], size=[sample_size], replace=False)  # (ray_batch_sz,)
        # get those coords
        # select_coords = coords[select_inds].long()  # (ray_batch_sz, 2)
        # select from origin and direction via coords

        sel_c = coords[select_inds, :].long()
        sel_c_flat = torch.Tensor([x*self.W + y for x, y in sel_c]).long()

        # --- RAYS ARE FLAT, IMAGES ARE 2D + Channel ---
        rays_o = rays_o[sel_c_flat, :]  # (ray_batch_sz, 3)
        rays_d = rays_d[sel_c_flat, :]  # (ray_batch_sz, 3)

        # stack origin and direction together

        # --- IMPORTANT, D BEFORE O ---
        batch_rays = torch.stack([rays_d, rays_o], 0)
        # target rgb color
        target_s = target_image[sel_c[:, 0], sel_c[:, 1]]  # (ray_batch_sz, 3)
        return batch_rays, target_s
        
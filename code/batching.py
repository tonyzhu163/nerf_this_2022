"""
Batching wrapper for rays
"""
import numpy as np
import torch
from params import ModelParameters
# ---------------------------------------------------------------------------- #
#                             Ray Helper Functions                             #
# ---------------------------------------------------------------------------- #
#TODO: REFACTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    i = i.to(device)
    j = j.to(device)

    # print(i.is_cuda)
    # print(K.is_cuda)
    # print(c2w.is_cuda)

    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
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
    def __init__(self, images, poses, i_train, H, W, K, device, params: ModelParameters, sample_mode) -> None:
        """
        Arguments
        - sample_mode str: either 'all' or 'single', determines if each iteration samples from across all images or a single image
        """
        self.device = device
        self.sample_mode = sample_mode
        self.i = 0
        self.images = images
        self.poses = poses
        self.i_train = i_train
        self.H = H
        self.W = W
        self.K = K
        self.params = params
        assert sample_mode in ['all', 'single']
        if sample_mode == 'all':
            self.get_rays_fn = self.rays_from_all
        else:
            self.get_rays_fn = self.rays_from_single

    def get_sample(self):
        """samples a random set of rays. sample source depends on if set to
        'all' or 'single'. 
        :return: a tuple batch_rays, target_s of shape (ray_batch_sz, 6) and
        (ray_batch_sz, 3)
        """
        batch_rays, target_s = self.get_rays_fn()
        # --- TEMP FIX ---
        target_s = target_s.cpu()
        self.i +=1
        return batch_rays, target_s
        
    #TODO: implement
    def rays_from_all(self):
        return None, None
    
    
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

    
    def rays_from_single(self):
        # randomly select one image from training set
        img_i = np.random.choice(self.i_train)
        target_image = self.images[img_i]
        target_image = torch.Tensor(target_image).to(self.device)
        cam_geo = [self.H, self.W]
        cam_geo += [torch.Tensor(self.K).to(self.device)]
        cam_geo += [torch.Tensor(self.poses[img_i, :3, :4]).to(self.device)]
        
        # if self.params.ray_batch_sz: #? why
        # get rays origin and direction from images
        rays_o, rays_d = get_rays(*cam_geo) # (H,W,3), (H,W,3)
        
        use_precrop = self.i < self.params.precrop_iters
        # TODO: REFACTOR
        coords = self.get_coords(use_precrop)
        # choose a random selection of rays of size ray_batch_sz
        select_inds = np.random.choice(coords.shape[0], size=[self.params.ray_batch_sz], replace=False)  # (ray_batch_sz,)
        # get those coords
        select_coords = coords[select_inds].long()  # (ray_batch_sz, 2)
        # select from origin and direction via coords
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_sz, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_sz, 3)
        # stack origin and direction together
        # TODO: do this first lmfao
        batch_rays = torch.stack([rays_o, rays_d], 0) # (ray_batch_sz, 6)
        # target rgb color
        target_s = target_image[select_coords[:, 0], select_coords[:, 1]]  # (ray_batch_sz, 3)
        return batch_rays, target_s
        
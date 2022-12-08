from dataclasses import dataclass, field

@dataclass
class ModelParameters:
    """
    Class for keep track of all training parameters and hyperparameters.
    """

    # expname:
    # basedir: './logs/'
    datadir: list[str] #* value defined below to avoid defaultfactory hassle
    epochs: int = 10000
    # pixel_nerf 500 -> 5 min aim for 10000 for now
    
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    N_rand: int = 32*32*4 #* N_rand is the number of rays in the batch, This should be ray_batch_size
    lrate: float = 5e-4
    lrate_decay: int = 250
    chunk: int = 1024*32 # This should be max_ray
    netchunk: int = 1024*64 # This should be max_points
    use_batching: bool = False
    # no_reload:
    # ft_path:
    
    # ----------------------------- rendering options ---------------------------- #
    N_samples: int = 64 #* N_samples is number of coarse samples per ray
    N_importance: int = 0 #* N_importance if >= 1, increases the number of bins in the fine sample (only called on the fine NERF once)
    perturb: int = 1.
    # use_viewdirs: I think we assume this is always positive
    # i_embed:
    multires: int = 10
    multires_views: int = 4
    raw_noise_std: float = 0.
    render_only: bool = True
    render_test: bool = True
    # render_factor:
    # precrop_iters:
    # precrop_frac:
    # dataset_type:
    # testskip:
    # shape:
    white_bkgd: bool = True #? True?
    half_res: bool = False #? True?
    ## llff flags
    # factor:
    # no_ndc:
    # lindisp:
    # spherify:
    # llffhold:
    ## logging/saving options
    # i_print: number of epochs per console printout
    # i_img: number of epochs per tensorboard output
    # i_weights: number of epochs per checkpoint
    # i_testset:
    # i_video: number of epochs per render

def get_params():
    params = ModelParameters(datadir=['data', 'nerf_synthetic'])
    return params
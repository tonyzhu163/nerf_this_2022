from dataclasses import dataclass

@dataclass
class ModelParameters:
    """Class for keep track of all training parameters and hyperparameters."""

    # expname:
    # basedir: './logs/'
    datadir: str = 'data/nerf_synthetic'
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    N_rand: int = 32*32*4
    lrate: float = 5e-4
    lrate_decay: int = 250
    chunk: int = 1024*32
    netchunk: int = 1024*64
    use_batching: bool = True
    # no_reload:
    # ft_path:
    N_samples: int = 64
    N_importance: int = 0
    perturb: int = 1.
    # use_viewdirs:
    # i_embed:
    multires: int = 10
    multires_views: int = 4
    raw_noise_std: float = 0.
    # render_only:
    # render_test:
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
    # i_print:
    # i_img:
    # i_weights:
    # i_testset:
    # i_video:

def get_params():
    params = ModelParameters()
    return params
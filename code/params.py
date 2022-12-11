import argparse
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """
    Class for keep track of all training parameters and hyperparameters.
    """

    expname: str
    # basedir: './logs/'
    datadir: list[str]
    object: str
    savedir: list[str]
    epochs: int = 10000
    # pixel_nerf 500 -> 5 min aim for 10000 for now

    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    # * ray_batch_sz is the number of rays in the batch
    # ray_batch_sz: int = 32 * 32 * 4 #* previously N_rand
    ray_batch_sz: int = 1024 #* previously N_rand
    lrate: float = 5e-4
    lrate_decay: int = 250
    # render_batch further into mini-batches to avoid OOM during rendering
    # render_batch_sz
    #* ray_chunk_sz might only used during rendering, since chunks are larger than batch size
    ray_chunk_sz: int = 1024 * 8 #* previously chunk
    point_chunk_sz: int = 1024 * 16  #* previously netchunk
    use_batching: bool = False
    no_reload: bool = True
    ft_path: str = None

    # ----------------------------- rendering options ---------------------------- #
    # * N_samples is number of coarse samples per ray
    N_samples: int = 64
    # * N_importance if >= 1, increases the number of bins in the fine sample (only called on the fine NERF once)
    N_importance: int = 128 #TEMP, change this to 1 when coarse is working
    perturb: int = 1.0
    # use_viewdirs: I think we assume this is always positive
    #* set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical
    i_embed: int = 0
    #* set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical
    i_embed_views: int = 0
    multires: int = 10
    multires_views: int = 4
    raw_noise_std: float = 0.0
    #* do not optimize, reload weights and render out render_poses path
    render_only: bool = False
    #* render the test set instead of render_poses path
    render_test: bool = True
    render_factor: int = 0.5
    precrop_iters: int = 500 #* customize this for each model
    precrop_frac: float = 0.5 #* customize this for each model
    # dataset_type:
    # testskip:
    # shape:
    white_bkgd: bool = True  # ? True?
    half_res: bool = False  # ? True?
    ## llff flags
    # factor:
    # no_ndc:
    # lindisp:
    # spherify:
    # llffhold:
    ## logging/saving options
    i_print: int = 100 # number of epochs per console printout
    i_img: int = 500 # number of epochs per tensorboard output
    i_weights: int = 1000 # number of epochs per checkpoint, previously 10000
    i_testset: int = 1000 #
    i_video: int = 5000 # number of epochs per render
    tensorboard: bool = True


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        type=str,
        default="chair",
        help="chair, drums, ficus, hotdog, lego, materials, mic, or ship",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default="untitled",
        help="handy way to remember the experiment",
    )
    args = parser.parse_args()
    params = ModelParameters(datadir=["data", "nerf_synthetic"], savedir=["logs"], **vars(args))
    return params

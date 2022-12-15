# nerf_this_2022

This is our PyTorch implementation of the [Neural Radiance Fields Paper](https://arxiv.org/pdf/2003.08934.pdf)  

## Training from Scratch
```bash
cd code
python run.py --object [objectname] --no_reload
```

## Rendering from Weights
```bash
cd code
python run.py --object [objectname] --render_only
```

Training on a single image may take upwards of 20 hours depending on hardware.

For reference:
- M1 Ultra with 64gbs of RAM (lego): 16 hrs
- RTX3060 (ship): 14 hrs

## Results
- [Google Drive](https://drive.google.com/drive/folders/19VngwjdyA_Q5l2s-D8mha2ZMw6w8XV_c?usp=share_link)
- includes render output videos, weight checkpoints for selected objects.

## Based On 
1. https://github.com/bmild/nerf
2. https://github.com/yenchenlin/nerf-pytorch
3. https://github.com/yashbhalgat/HashNeRF-pytorch
4. https://github.com/kwea123/nerf_pl

## GIT Etiquette
- always fetch+pull before making a commit
- commit & push often to make sure your code is seen by everyone
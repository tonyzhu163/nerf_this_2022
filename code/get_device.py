from packaging import version
import torch
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'DEVICE SETTING: running on CUDA GPU')
else:
    mps_is_available = (
        version.parse(torch.__version__) >= version.parse("1.13.0")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if mps_is_available:
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
        print(f'DEVICE SETTING: running on Mac M1 MPS')
    else:
        device = torch.device("cpu")
        print(f'DEVICE SETTING: running on CPU')
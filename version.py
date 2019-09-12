__version__ = '0.3.0'
git_version = 'be376084d84dedd99284625d5b12a3643cfbe3d8'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION

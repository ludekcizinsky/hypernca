from torch.utils.data import Dataset
from pathlib import Path,PosixPath
from torch import Tensor
import numpy as np
from PIL import Image
import torch


class TextureDataset:
    def __init__(self, data_dir: str) -> None:
        self.images = list(Path(data_dir).glob(pattern='*.jpg'))
    
    def filter_images(self, pattern:str=None) -> None:
        """
        Filter images based on pattern

        Args:
            pattern (str) : Pattern like 'banded*' or 'banded_0002'
        """
        if pattern is not None:
            self.images = [img for img in self.images if pattern in str(img)]

    def random_sample(self, n:int) -> None:
        """
        Randomly sample n images from the dataset

        Args:
            n (int) : Number of images to sample
        """
        if len(self.images) > n:
            self.images = np.random.choice(self.images, size=n, replace=False).tolist()
        else:
            raise ValueError(f"Cannot sample {n} images from a dataset with only {len(self.images)} images.")

        
class TextureImageDataset(Dataset):
    def __init__(self, img_path:PosixPath) -> None:
        self.img_path = img_path
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx:int) -> tuple[Tensor,str]:
        img = Image.open(self.img_path).convert('RGB')
        img.thumbnail(size=(128, 128),resample=Image.LANCZOS)
        img = np.float32(img) / 255.0
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img,str(self.img_path)

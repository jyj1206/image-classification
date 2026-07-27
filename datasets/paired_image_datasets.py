import random
import cv2

from torch.utils.data import Dataset
from pathlib import Path


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=None, transform=None):
        super().__init__()
        self.root_path = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        sharp_path, blur_path = self.image_paths[idx]
        sharp_image = self._load_image(sharp_path)
        blur_image = self._load_image(blur_path)

        if self.patch_size:
            sharp_image, blur_image = self._paired_random_crop(sharp_image, blur_image)

        if self.transform:
            sharp_image = self.transform(sharp_image)
            blur_image = self.transform(blur_image)

        return sharp_image, blur_image

    def _get_image_paths(self):
        sharp_dir = self.root_path / self.split / "GT"
        blur_dir = self.root_path / self.split / "LQ"

        sharp_paths = {path.stem: path for path in sorted(sharp_dir.glob("*.png"))}
        blur_paths = sorted(blur_dir.glob("*.png"))

        image_paths = []
        for blur_path in blur_paths:
            sharp_name = blur_path.stem.split("-", 1)[0]
            sharp_path = sharp_paths[sharp_name]
            image_paths.append((sharp_path, blur_path))

        return image_paths

    def _paired_random_crop(self, sharp_image, blur_image):
        sharp_h, sharp_w = sharp_image.shape[:2]
        blur_h, blur_w = blur_image.shape[:2]

        if (sharp_h, sharp_w) != (blur_h, blur_w):
            raise ValueError(
                f"GT/LQ image size mismatch: GT {(sharp_h, sharp_w)}, LQ {(blur_h, blur_w)}"
            )

        patch_size = int(self.patch_size)
        if patch_size > sharp_h or patch_size > sharp_w:
            raise ValueError(
                f"Patch size {patch_size} is larger than image size {(sharp_h, sharp_w)}"
            )

        top = random.randint(0, sharp_h - patch_size)
        left = random.randint(0, sharp_w - patch_size)
        bottom = top + patch_size
        right = left + patch_size

        sharp_patch = sharp_image[top:bottom, left:right, :]
        blur_patch = blur_image[top:bottom, left:right, :]
        return sharp_patch, blur_patch
    
    def _load_image(self, path):
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
        
    

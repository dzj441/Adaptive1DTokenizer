from typing import List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, default_collate
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True # tolerate truncated images
Image.MAX_IMAGE_PIXELS = None  # keep consistent behavior


def is_distributed_initialized() -> bool:
    """Return True only if torch.distributed is initialized."""
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


class ImageTransform:
    """
    Match the transform knobs of the WebDataset (class-label) branch:
      - Train: Resize(shorter=resize_shorter_edge, BICUBIC, antialias=True)
               -> RandomCrop(crop_size) or CenterCrop(crop_size) (if random_crop=False)
               -> RandomHorizontalFlip() (if random_flip=True)
               -> ToTensor
               -> Normalize(mean, std)
      - Eval : Resize(crop_size) -> CenterCrop(crop_size) -> ToTensor -> Normalize(mean, std)
    """

    def __init__(
        self,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
    ):
        interp = transforms.InterpolationMode.BICUBIC

        train_ops = [
            transforms.Resize(resize_shorter_edge, interpolation=interp, antialias=True),
            transforms.RandomCrop(crop_size) if random_crop else transforms.CenterCrop(crop_size),
        ]
        if random_flip:
            train_ops.append(transforms.RandomHorizontalFlip())
        train_ops += [
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
        self.train_transform = transforms.Compose(train_ops)

        # For eval we always resize to crop_size to keep comparisons fair.
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(crop_size, interpolation=interp, antialias=True),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        print(f"[ImageTransform] train: {self.train_transform}")
        print(f"[ImageTransform] eval : {self.eval_transform}")


class ImageFolderWithFilename(ImageFolder):
    """
    Extend ImageFolder to also return filename so we can expose both 'filename' and '__key__'
    to mimic WebDataset batches.
    Returned item: (image_tensor, class_idx, filename_str)
    """

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]  # (path, class_idx)
        filename = Path(path).name
        return image, target, filename


def _collate_to_dict(batch):
    """
    Collate a list of (image, class_id, filename) into a dict (WebDataset-like):
      {
        "image": FloatTensor[B, 3, H, W],
        "class_id": LongTensor[B],
        "filename": List[str],   # filename with extension
        "__key__": List[str],    # key-like id; here we use filename stem to mimic WDS '__key__'
      }
    """
    images, targets, filenames = zip(*batch)
    images = default_collate(images)
    class_ids = torch.as_tensor(targets, dtype=torch.long)
    filenames = list(filenames)
    keys = [Path(fn).stem for fn in filenames]  # mimic WebDataset '__key__'
    return {
        "image": images,
        "class_id": class_ids,
        "filename": filenames,
        "__key__": keys,
    }


class SimpleImageFolderDataset:
    """
    Fusion dataset/dataloaders:
      * Transform knobs & batch field names aligned with WebDataset class-label branch.
      * Implemented with torchvision ImageFolder (via ImageFolderWithFilename wrapper).

    Expected ImageFolder layout:
      train_dir/
        <wnid_a>/*.jpg|*.jpeg|*.png|...
        <wnid_b>/*.jpg|*.jpeg|*.png|...
        ...
      eval_dir/
        <wnid_a>/*.jpg|*.jpeg|*.png|...
        <wnid_b>/*.jpg|*.jpeg|*.png|...
        ...

    Each batch is a dict with: "image", "class_id", "filename", "__key__".
    """

    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        per_gpu_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last_train: bool = True,
    ):
        self.train_dir = Path(train_dir)
        self.eval_dir = Path(eval_dir)
        assert self.train_dir.is_dir(), f"train_dir does not exist or is not a directory: {self.train_dir}"
        assert self.eval_dir.is_dir(), f"eval_dir does not exist or is not a directory: {self.eval_dir}"

        # ====== Transforms ======
        self.transforms = ImageTransform(
            resize_shorter_edge=resize_shorter_edge,
            crop_size=crop_size,
            random_crop=random_crop,
            random_flip=random_flip,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        # ====== Datasets ======
        self._train_dataset = ImageFolderWithFilename(
            root=str(self.train_dir),
            transform=self.transforms.train_transform,
        )
        self._eval_dataset = ImageFolderWithFilename(
            root=str(self.eval_dir),
            transform=self.transforms.eval_transform,
        )

        # ====== DataLoaders ======
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=per_gpu_batch_size,
            shuffle=True,   
            num_workers=num_workers_per_gpu,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers_per_gpu > 0),
            drop_last=drop_last_train,
            collate_fn=_collate_to_dict,
        )
        self._eval_dataloader = DataLoader(
            self._eval_dataset,
            batch_size=per_gpu_batch_size,
            shuffle=False,  
            num_workers=num_workers_per_gpu,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers_per_gpu > 0),
            drop_last=False,
            collate_fn=_collate_to_dict,
        )

    # ====== Public properties ======
    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader

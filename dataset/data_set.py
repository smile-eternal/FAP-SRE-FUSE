import os
import queue
import secrets
import threading

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import imgproc
from utils import  utils
class PairedImageDataset(Dataset):
    """Define Test dataset loading methods."""

    def __init__(
            self,
            paired_gt_images_dir: str,
    ) -> None:
        """

        Args:
            paired_gt_images_dir: The address of the ground-truth image after registration
            paired_lr_images_dir: The address of the low-resolution image after registration
        """

        super(PairedImageDataset, self).__init__()
        if not os.path.exists(paired_gt_images_dir):
            raise FileNotFoundError(f"Registered high-resolution image address does not exist: {paired_gt_images_dir}")

        # Get a list of all image filenames
        image_files = natsorted(os.listdir(paired_gt_images_dir))
        self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in image_files]


    def __getitem__(self, batch_index: int) -> [Tensor, Tensor, str]:
        # Read a batch of image data
        gt_image = cv2.imread(self.paired_gt_image_file_names[batch_index]).astype(np.float32) / 255.
        h,w,_=gt_image.shape
        lr_image = cv2.resize(gt_image, (int(w/2), int(h/2)), interpolation=cv2.INTER_CUBIC)
        # BGR convert RGB
        # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = utils.image_to_tensor(gt_image, False, False)
        lr_tensor = utils.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor,
                "lr": lr_tensor,
                "image_name": self.paired_gt_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.paired_gt_image_file_names)

class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)  # 此处next为内置函数不为下面的属性函数self.next，,调用data中的getitem函数，
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
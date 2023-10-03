from .datasets import MetaLensPair, LMDBMetaLensPair
from torch.utils.data import DataLoader
import torch
import cv2
import random
import numpy as np
def get_iter_flag(iter_sections, now_iter):
    for i, it in enumerate(iter_sections) :
        if now_iter < it :
            return i
    return len(iter_sections)-1 


def define_data_instances(config, index, normalization):

    batch_size = config.mini_batch_sizes[index]
    ps = config.gt_sizes[index]

    with open("dataset_instance_info.txt", "w") as file:
        file.write(f"batchsize : {batch_size} \n")
        file.write(f"Patchsize : {ps}")

    
    if config.data_mode == "lmdb":
        dataset = LMDBMetaLensPair(patch_size=ps, normalization=normalization)
    else:
        dataset = MetaLensPair(patch_size=ps, normalization=normalization)
    num_workers = 4 if index >= 3 else 8
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return dataset, dataloader

class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, cuda_num):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = torch.device(f'cuda:{cuda_num}')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)
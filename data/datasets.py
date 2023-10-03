import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2

from basicsr.data.data_util import paired_paths_from_lmdb
from basicsr.utils import FileClient

def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale=1, gt_path="a"):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def get_patch(imgs, patch_size):

    # [img1, img2, meshgrid] 
    # img1 : H, W, C , img2 : H, W, C, meshgrid : H, W, 2
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.transpose(imgs[i], (1, 0, 2))
    return imgs

def bit_to_minus_one_one(img):
    img /= 255.
    return img * 2 - 1

def open_images(path_pair):
    x, y = path_pair
    gt_img = Image.open(x).convert("RGB")
    n_img = Image.open(y).convert("RGB")
    return gt_img, n_img

# same image file type, same name
def get_paired_file_path(root, high_suffix="ground_truth", low_suffix="meta_image"):
    high_path = os.path.join(root, high_suffix)
    low_path = os.path.join(root, low_suffix)
    
    path_pairs = []
    file_names = os.listdir(high_path)
    for fn in file_names:
        hq_path = os.path.join(high_path, fn)
        lq_path = os.path.join(low_path, fn)
        path_pairs.append([hq_path, lq_path])
    return path_pairs
    
    
class LMDBMetaLensPair(Dataset):
    def __init__(self, path="/home/joon/Datasets/Metalens_0622", train=True, patch_size=256, image_size=(800, 1280), test_scale=1, coord_info=False, normalization=False):
        super().__init__()
        self.train = train
        self.patch_size = patch_size
        self.file_client = None
        self.test_scale = test_scale
        self.coord_info = coord_info
        self.normalization = normalization
        self.io_backend_opt = {}
        self.io_backend_opt['type'] = "lmdb"
        
        self.H, self.W = image_size
        
        if self.coord_info:
            # entire mesh grid.
            x_axis = np.arange(0, self.W) / self.W
            y_axis = np.arange(0, self.H) / self.H
            if self.normalization:
                x_axis, y_axis = x_axis*2-1, y_axis*2-1
            grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing='xy') # x : W, y : H -> 가로세로 인덱싱 고려 -> xy
            self.mesh_grid = np.stack([grid_x, grid_y], axis=-1) # H, W, 2 & normalizeds
            print(self.mesh_grid.shape) # 800, 1280
        else:
            self.mesh_grid = None
        
        
        if train :
            path = os.path.join(path, 'train')
        else :
            path = os.path.join(path, "test")
        self.gt_folder = os.path.join(path, "ground_truth.lmdb")
        self.meta_folder = os.path.join(path, "meta.lmdb")

        self.io_backend_opt['db_paths'] = [self.meta_folder, self.gt_folder]
        self.io_backend_opt['client_keys'] = ['meta', 'ground_truth']
        
        
        self.paths = paired_paths_from_lmdb([self.gt_folder, self.meta_folder], ['ground_truth', 'meta'])
        
    def __len__(self):
        return len(self.paths)

        
    def __getitem__(self, index):
        if self.file_client is None :
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        index = index % len(self.paths)
        
        gt_path = self.paths[index]["ground_truth_path"]
        img_bytes = self.file_client.get(gt_path, 'ground_truth') 
        try:
            img_gt = imfrombytes(img_bytes, float32=True) # [0-1]
        except:
            raise Exception("gt path {} not working".format(gt_path))

        meta_path = self.paths[index]["meta_path"]
        img_bytes = self.file_client.get(meta_path, 'meta') 
        try:
            img_meta = imfrombytes(img_bytes, float32=True) # [0-1]
        except:
            raise Exception("lq path {} not working".format(meta_path))
        
        #### image load. ####
        
        if self.normalization:
            img_gt, img_meta = img_gt*2-1, img_meta*2-1

        if self.train:
            if self.coord_info:
                [x, y, position_coord]= get_patch([img_gt, img_meta, self.mesh_grid], self.patch_size)
                pe = torch.tensor(np.transpose(position_coord, axes=[2, 0, 1]).astype('float32'))
            else:
                [x, y] = get_patch([img_gt, img_meta], self.patch_size)
            x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32'))
            y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32'))
            
        else :
            if self.coord_info:
                pe = torch.tensor(np.transpose(self.mesh_grid, axes=[2, 0, 1]).astype('float32'))

            x = torch.tensor(np.transpose(img_gt, axes=[2, 0, 1]).astype('float32'))
            y = torch.tensor(np.transpose(img_meta, axes=[2, 0, 1]).astype('float32'))

            if self.test_scale < 1:
                C, H, W = x.shape
                new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((new_H, new_W))
                ])
                x,y = transform(x), transform(y)
                if self.coord_info:
                    pe = transform(pe)
        
        if self.coord_info:
            return x, y, pe
        else:
            return x, y


class MetaLensPair(Dataset):
    def __init__(self, path="/home/joon/Datasets/Metalens_0622", train=True, patch_size=256, test_scale=1, coord_info=False, normalization=False):
        super().__init__()

        self.patch_size = patch_size
        self.train = train
        self.test_scale = test_scale
        self.coord_info = coord_info
        self.normalization = normalization
                
        high_qual = "ground_truth"
        low_qual = "meta_image"
        
        self.path_pairs = get_paired_file_path(path, high_suffix=high_qual, low_suffix=low_qual)

    def __len__(self):
        return len(self.path_pairs)
    

    def __getitem__(self, idx):
        # Test 의 경우, 앞에서 언급했듯이, mat file에 데이터가 전체로 들어가있따.
        # 그러므로, path 단위로 꺼내오는게 아니라, 우선 mat file에서 데이터를 모두 불러오고 이 데이터를 indexing해서 배치화 하자.
        if not self.train : # full resolution으로 돌리고싶긴 하다만, 안돌아감.
            x, y = open_images(self.path_pairs[idx])
            x, y = np.array(x), np.array(y)
            x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32')) / 255.
            y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32')) / 255.
            
            if self.normalization: # 0-1 -> -1, 1
                x, y = x*2-1, y*2-1

            C, H, W = x.shape

            new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((new_H, new_W))
            ])

            return transform(x), transform(y)

        else :
            x, y = open_images(self.path_pairs[idx])
            if self.patch_size > 0 :
                x, y = np.array(x), np.array(y)
                if self.coord_info:
                    [x, y], xx, yy = get_patch([x, y], self.patch_size, self.coord_info)
                else:
                    [x, y] = get_patch([x, y], self.patch_size, self.coord_info)
                x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32')) / 255.
                y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32')) / 255.
                
                if self.normalization: # 0-1 -> -1, 1
                    x, y = x*2-1, y*2-1
                
                if self.coord_info:
                    return x, y, xx, yy

            return x, y

def np_augs(imgs):
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.transpose(imgs[i], (1, 0, 2))
    return imgs

class LMDBOriginMetaLensPair(Dataset):
    def __init__(self, path="/home/joon/Datasets/Metalens_0622", train=True, gt_size=384, image_size=(800, 1280), test_scale=1, coord_info=False, normalization=False):
        super().__init__()
        self.train = train
        self.file_client = None
        self.test_scale = test_scale
        self.coord_info = coord_info
        self.normalization = normalization
        self.gt_size = gt_size
        self.io_backend_opt = {}
        self.io_backend_opt['type'] = "lmdb"
        
        self.H, self.W = image_size
        
        if self.coord_info:
            # entire mesh grid.
            x_axis = np.arange(0, self.W) / self.W
            y_axis = np.arange(0, self.H) / self.H
            if self.normalization:
                x_axis, y_axis = x_axis*2-1, y_axis*2-1
            grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing='xy') # x : W, y : H -> 가로세로 인덱싱 고려 -> xy
            self.mesh_grid = np.stack([grid_x, grid_y], axis=-1) # H, W, 2 & normalizeds
            print(self.mesh_grid.shape) # 800, 1280
        else:
            self.mesh_grid = None
        
        
        if train :
            path = os.path.join(path, 'train')
        else :
            path = os.path.join(path, "test")
        self.gt_folder = os.path.join(path, "ground_truth.lmdb")
        self.meta_folder = os.path.join(path, "meta.lmdb")

        self.io_backend_opt['db_paths'] = [self.meta_folder, self.gt_folder]
        self.io_backend_opt['client_keys'] = ['meta', 'ground_truth']
        
        
        self.paths = paired_paths_from_lmdb([self.gt_folder, self.meta_folder], ['ground_truth', 'meta'])
        
    def __len__(self):
        return len(self.paths)

        
    def __getitem__(self, index):
        if self.file_client is None :
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        index = index % len(self.paths)
        
        gt_path = self.paths[index]["ground_truth_path"]
        img_bytes = self.file_client.get(gt_path, 'ground_truth') 
        try:
            img_gt = imfrombytes(img_bytes, float32=True) # [0-1]
        except:
            raise Exception("gt path {} not working".format(gt_path))

        meta_path = self.paths[index]["meta_path"]
        img_bytes = self.file_client.get(meta_path, 'meta') 
        try:
            img_meta = imfrombytes(img_bytes, float32=True) # [0-1]
        except:
            raise Exception("lq path {} not working".format(meta_path))
        
        #### image load. ####
        
        if self.normalization:
            img_gt, img_meta = img_gt*2-1, img_meta*2-1

        # 일단 coord info 버전 제외.
        if self.train:
            # random crop
            img_gt, img_meta = paired_random_crop(img_gt, img_meta, self.gt_size)
            
            [x,y] = np_augs([img_gt,img_meta])
            x = torch.tensor(np.transpose(x, axes=[2, 0, 1]).astype('float32'))
            y = torch.tensor(np.transpose(y, axes=[2, 0, 1]).astype('float32'))
            
        else :

            x = torch.tensor(np.transpose(img_gt, axes=[2, 0, 1]).astype('float32'))
            y = torch.tensor(np.transpose(img_meta, axes=[2, 0, 1]).astype('float32'))

            if self.test_scale < 1:
                C, H, W = x.shape
                new_H, new_W = list(map(int, [self.test_scale * H, self.test_scale * W]))
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((new_H, new_W))
                ])
                x,y = transform(x), transform(y)
        
        return {
            'lq': y,
            'gt': x,
        }
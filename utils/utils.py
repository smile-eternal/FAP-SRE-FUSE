# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import cv2
import os
from numpy import ndarray
import torch
from enum import Enum
from torch import distributed as dist
class MyDataset(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        with open(path,'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open('./data/images/'+fn).convert('RGB')
        # print('img.size',img.size)
        label = fn.split('.')[0].split('_')[-1] #groundtruth name
        gt = cv2.imread('./data/template2/' + str(label) + '.bmp') #read groundtruth
        gt = np.fabs(np.float32((cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY) // 255.0)[np.newaxis,:,:])) #convert to binaray gray

        if self.transform is not None:
            img = self.transform(img)

        return img, gt, label

    def __len__(self):
        return len(self.imgs)
class MyDataset_Raw_Blur(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        with open(path,'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        p1 = self.imgs[index][0]
        p2 = self.imgs[index][1]
        pgt = '_'.join(p2.split('_')[1:5]) + '.jpg'
        img1 = Image.open('./data/images/' + p1).convert('RGB')
        img2 = Image.open('./data/images/' + p2).convert('RGB')
        gtimg = Image.open('./data/raw/' + pgt).convert('RGB')

        label1 = p1.split('.')[0].split('_')[-1] #groundtruth name
        label2 = p2.split('.')[0].split('_')[-1] #groundtruth name
        gt1 = Image.open('./data/template2/' + str(label1) + '.bmp')
        gt2 = Image.open('./data/template2/' + str(label2) + '.bmp')

        if self.transform is not None:
            #print('self.transform',self.transform)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            gt1 = self.transform(gt1)[0,:,:].unsqueeze(0).clamp_(0,1)
            gt2 = self.transform(gt2)[0,:,:].unsqueeze(0).clamp_(0,1)
            gtimg = self.transform(gtimg)

        return img1,img2,gt1,gt2,label1,label2,gtimg

    def __len__(self):
        return len(self.imgs)

class MyDataset_Gradient(Dataset):
    def __init__(self, path, transform = None, target_transform = None):
        with open(path,'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        p1 = self.imgs[index][0]
        p2 = self.imgs[index][1]
        pgt = '_'.join(p1.split('_')[1:5])+'.jpg'

        img1 = Image.open('./data/image/' + p1).convert('RGB')
        img2 = Image.open('./data/image/' + p2).convert('RGB')
        gtimg = Image.open('./data/raw/' + pgt).convert('RGB')

        label1 = p1.split('.')[0].split('_')[-1] #groundtruth name
        label2 = p2.split('.')[0].split('_')[-1] #groundtruth name
        gt1 = Image.open('./data/template2/' + str(label1) + '.bmp')
        gt2 = Image.open('./data/template2/' + str(label2) + '.bmp')

        if self.transform is not None:
            #print('self.transform',self.transform)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            gt1 = self.transform(gt1)[0,:,:].unsqueeze(0).clamp_(0,1)
            gt2 = self.transform(gt2)[0,:,:].unsqueeze(0).clamp_(0,1)
            gtimg = self.transform(gtimg)

        return img1,img2,gt1,gt2,label1,label2,gtimg

    def __len__(self):
        return len(self.imgs)

class MyTestDataset(Dataset):
    def __init__(self, path, pkfile, transform = None, target_transform = None):
        imgs = []  #for each element represent a pair tuple

        with open(pkfile,'rb') as file:
            imgs = pkl.load(file)
        print(imgs)
        print(len(imgs))
        self.imgs = imgs
        # print(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.path = path

    def __getitem__(self, index):
        fn = self.imgs[index]

        img1 = Image.open(self.path +'/' + fn[0]).convert('RGB')
        img2 = Image.open(self.path +'/' + fn[1]).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1,img2,fn[0],fn[1]

    def __len__(self):
        return len(self.imgs)
class test_s_p_dataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        with open(path, 'rb') as file:
            imgs = pkl.load(file)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        p1 = self.imgs[index][0]
        p2 = self.imgs[index][1]
        pgt = '_'.join(p1.split('_')[1:5]) + '.jpg'
        img1 = cv2.imread('./data/sampleval100/' + p1)
        img2 = cv2.imread('./data/sampleval100/' + p2)
        l_img1=cv2.resize(img1,(64,64),interpolation=cv2.INTER_CUBIC)

        l_img2=cv2.resize(img2,(64,64),interpolation=cv2.INTER_CUBIC)
        gtimg = cv2.imread('./data/raw/' + pgt)



        if self.transform is not None:
            # print('self.transform',self.transform)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            l_img1=self.transform(l_img1)
            l_img2=self.transform(l_img2)
            gtimg = self.transform(gtimg)

        return img1, img2, l_img1,l_img2,gtimg

    def __len__(self):
        return len(self.imgs)


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor
def transTensor2Numpy(image:torch.tensor):
    image=image.cpu().squeeze(0)
    image=image.detach().numpy()*255
    image=np.transpose(image,(1,2,0))
    return image
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n#加和
        self.count += n#计数
        self.avg = self.sum / self.count#均值

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.4f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.4f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.4f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    #输出当前batch\总batch、loss及用时等信息
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):#num_batches:172
        num_digits = len(str(num_batches // 1))#num_digits:3
        fmt = "{:" + str(num_digits) + "d}"#fmt:'{:3d}'
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"#[{:3d}/172]
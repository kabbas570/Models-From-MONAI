import glob
import os
import datetime
import glob
import os
import random

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    SpatialPadd,
    SqueezeDimd,
    ToTensord,
    NormalizeIntensityd,
)
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter


def get_data_list(data_folder):
    
    images2d_path = glob.glob(
        os.path.join(data_folder, "*", f"*LA_{'ES'}.nii.gz")
    ) + glob.glob(
        os.path.join(data_folder, "*", f"*LA_{'ED'}.nii.gz")
    )
      
    gt2d_path = [
        x.replace(".nii.gz", "_gt.nii.gz") for x in images2d_path
    ]
    
    
    gt3d_path = [
        x.replace("LA", "SA") for x in gt2d_path
    ]
    
    images3d_path = [
        x.replace("LA", "SA") for x in images2d_path
    ]
    
    return images2d_path , gt2d_path , images3d_path , gt3d_path 

data_folder = r'C:\My_Data\M2M Data\data\data_2\val'

images2d_path , gt2d_path , images3d_path, gt3d_path = get_data_list(data_folder)

def all_files_exist(files):
    return all([os.path.exists(x) for x in files])

assert all_files_exist(images2d_path), "Some 2D images are missing."
assert all_files_exist(images3d_path), "Some 3D images are missing."

assert all_files_exist(gt2d_path), "Some 2D images are missing."
assert all_files_exist(gt3d_path), "Some 3D images are missing."

def get_transforms(mode="small"):
    if mode == "full":
        pixdim = [2.0, 2.0, 5.0]
        pad3d = [256, 256, 32]
        crop3d = [256, 256, 32]
    elif mode == "small":
        pixdim = [8.0, 8.0, 8.0]

        pad3d = [64, 64, 32]
        crop3d = [64, 64, 32]

    return Compose(
        [
            LoadImaged(keys=["image2d", "label2d","image3d","label3d",]),
            SqueezeDimd(keys=["image2d","label2d"], dim=-1),
            
            AddChanneld(keys=["image2d", "label2d","image3d","label3d",]),
            Spacingd(keys=["image3d","label3d"], pixdim=pixdim, mode=("bilinear")),
            # Orientationd(keys=["image3d"], axcodes="RAS"),
            SpatialPadd(keys=["image2d", "label2d"], spatial_size=(256, 256)),
            SpatialPadd(keys=["image3d","label3d"], spatial_size=pad3d),
            CenterSpatialCropd(keys=["image2d","label2d"], roi_size=(256, 256)),
            CenterSpatialCropd(keys=["image3d","label3d"], roi_size=crop3d),
            # mean and std normalize -- offset input
            # NormalizeIntensityd(keys=["image2d", "image3d"], subtrahend=10, divisor=1),
            ToTensord(keys=["image2d", "label2d","image3d","label3d"]),
        ]
    )

def get_data_loader(
    images2d_path , 
    gt2d_path , 
    images3d_path, 
    gt3d_path ,
    batch_size=1,
    num_workers=0,
    transforms=get_transforms(),
):
    images_dict = []
    for i2d,gt2d, i3d , gt3d in zip(images2d_path , gt2d_path , images3d_path, gt3d_path):
        images_dict.append({"image2d": i2d,"label2d": gt2d, "image3d": i3d, "label3d": gt3d})

    random.shuffle(images_dict)

    dataset = monai.data.Dataset(data=images_dict, transform=transforms)

    loader = monai.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        collate_fn=monai.data.list_data_collate,
    )

    return loader

lr = 1e-4
train_val_split = 0.8
batch_size = 1
num_workers = 4
exp_mode = "full"
    
train_loader = get_data_loader(
        images2d_path , 
        gt2d_path , 
        images3d_path, 
        gt3d_path ,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=get_transforms(mode=exp_mode),
    )

a  = iter(train_loader)
a1 = next(a)

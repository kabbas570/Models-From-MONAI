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

data_folder = r'C:\My_Data\M2M Data\data\data_2\train'

images2d_path , gt2d_path , images3d_path, gt3d_path = get_data_list(data_folder)

def all_files_exist(files):
    return all([os.path.exists(x) for x in files])

assert all_files_exist(images2d_path), "Some 2D images are missing."
assert all_files_exist(images3d_path), "Some 3D images are missing."

assert all_files_exist(gt2d_path), "Some 2D images are missing."
assert all_files_exist(gt3d_path), "Some 3D images are missing."

def get_transforms():
    return Compose(
        [
            LoadImaged(keys=["image2d", "label2d","image3d","label3d",]),
            AddChanneld(keys=["image3d","label3d","image2d", "label2d"]), ## does it always add to  axis = 0 ?
            SqueezeDimd(keys=["image2d", "label2d"], dim=-1),   ## wil sequeeze last dimenssion , if we expand first and then sequees
                                                                ## it will affect the 2D images pixeims as well, and change spatial dims.
            ## Adjust Spacing for 2D Data  ###
            Spacingd(keys=["image2d"], pixdim=[1.25,1.25], mode=("bilinear")),
            Spacingd(keys=["label2d"], pixdim=[1.25,1.25], mode=("nearest")),
                        
            ## Adjust Spacing for 3D Data  ###

            Spacingd(keys=["image3d"], pixdim=[1.25,1.25,5], mode=("bilinear")),
            Spacingd(keys=["label3d"], pixdim=[1.25, 1.25,5], mode=("nearest")),
            
 
            ## Cropping 2D Data  ###
            SpatialPadd(keys=["image2d","label2d"], spatial_size=(256, 256)), # for this I understand why we use this, i.e. if the spatial
                                                                              # dims is smaller than 256 it will fist padd to make it 256 
                                                                              # followed by center crop of (256 x 256)
            CenterSpatialCropd(keys=["image2d","label2d"], roi_size=(256, 256)),
            
            ## Cropping 3D Data  ###
            SpatialPadd(keys=["image3d","label3d"], spatial_size=(256,256,32)),  
            CenterSpatialCropd(keys=["image3d","label3d"], roi_size=(256,256,32)),
            
            Orientationd(keys=["image3d","label3d"], axcodes="RAS"), ## this moves the depth to second last dimenssions not the first [B,C,D,H,W]
            
            # ToTensord(keys=["image2d", "image3d"]),  I think this is not necessary as it already returns Tensors
            
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

batch_size = 1
num_workers = 0
    
train_loader = get_data_loader(
        images2d_path , 
        gt2d_path , 
        images3d_path, 
        gt3d_path ,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=get_transforms(),
    )

a  = iter(train_loader)

for i in range(1):
    a1 = next(a)
    
    img2d = a1["image2d"][0,0,:].numpy()
    gt2d = a1["label2d"][0,0,:].numpy()
    
    img3d = a1["image3d"].numpy()
    gt3d = a1["label3d"].numpy()
    
    print(img3d.shape)
    print(gt3d.shape)

    
    # tf = a1["image2d_meta_dict"]
    # b =tf["pixdim"].numpy()
    
    

 

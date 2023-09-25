import glob
import os

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
    
    return images2d_path , gt2d_path , gt3d_path , images3d_path

data_folder = r'C:\My_Data\M2M Data\data\data_2\val'

images2d_path , gt2d_path , gt3d_path , images3d_path = get_data_list(data_folder)

def all_files_exist(files):
    return all([os.path.exists(x) for x in files])

assert all_files_exist(images2d_path), "Some 2D images are missing."
assert all_files_exist(images3d_path), "Some 3D images are missing."

assert all_files_exist(gt2d_path), "Some 2D images are missing."
assert all_files_exist(gt3d_path), "Some 3D images are missing."

def get_data_loader(
    images2d_path,
    images3d_path,
    train_val_split=0.8,
    batch_size=1,
    num_workers=0,
    transforms=get_transforms(),
):
    images_dict = []
    for i2d, i3d in zip(images2d_path, images3d_path):
        images_dict.append({"image2d": i2d, "image3d": i3d})

    random.shuffle(images_dict)

    # split into train and val
    train_len = int(len(images_dict) * train_val_split)
    train_images = images_dict[:train_len]
    val_images = images_dict[train_len:]

    train_dataset = monai.data.Dataset(data=train_images, transform=transforms)
    val_dataset = monai.data.Dataset(data=val_images, transform=transforms)

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        collate_fn=monai.data.list_data_collate,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        collate_fn=monai.data.list_data_collate,
    )

    return train_loader, val_loader

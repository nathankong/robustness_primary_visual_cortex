import torch
import torchvision

import numpy as np

from torch.utils import data
from torchvision import transforms, datasets


#=======================================================
# Main function to get dataloader from dataset
#=======================================================

def _acquire_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    assert isinstance(dataset, data.Dataset)
    loader = data.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return loader


#=======================================================
# ImageNet data sets and loaders (from image files)
#=======================================================

def get_imagenet_loaders(train_params, my_transforms=None, shuffle=True):
    # Assumes image_dir organization is /PATH/TO/IMAGENET/{train, val}/{synsets}/*.JPEG
    assert "image_dir" in train_params.keys()
    assert "batch_size" in train_params.keys()
    assert "num_workers" in train_params.keys()

    if my_transforms is None:
        my_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        }
    else:
        assert "train" in my_transforms.keys()
        assert "val" in my_transforms.keys()

    print(f"Using transforms: {my_transforms}")

    # Train set
    train_set = datasets.ImageFolder(train_params["image_dir"] + "/train/", transform=my_transforms["train"])
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=train_params["batch_size"], 
        shuffle=shuffle,
        pin_memory=True,
        num_workers=train_params["num_workers"]
    )

    # Validation set
    val_set = datasets.ImageFolder(train_params["image_dir"] + "/val/", transform=my_transforms["val"])
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=train_params["batch_size"], 
        shuffle=shuffle,
        pin_memory=True,
        num_workers=train_params["num_workers"]
    )

    return train_loader, val_loader


#=======================================================
# Wrapper for getting dataloaders
#=======================================================

def get_dataloaders(train_params, my_transforms=None, shuffle=True):
    assert "dataset" in train_params.keys()

    if train_params["dataset"] == "imagenet_jpeg":
        dataloaders = get_imagenet_loaders(
            train_params,
            my_transforms=my_transforms,
            shuffle=shuffle
        )
    else:
        assert 0, "No data set loaded."

    return dataloaders


#===================================================
# Array dataset for images in array format
#===================================================

class ArrayDataset(data.Dataset):
    """
    General dataset constructor using an array of images and labels.
    
    Arguments:
        image_array : numpy array of shape (N, H, W, 3)
        labels      : numpy array of shape (N,)
        t           : torchvision.transforms instance
    """
    def __init__(self, image_array, labels, t=None):
        assert image_array.shape[0] == labels.shape[0]
        assert t is not None

        self.transforms = t
        self.image_array = image_array
        self.labels = labels
        self.n_images = image_array.shape[0]

    def __getitem__(self, index):
        inputs = self.transforms(self.image_array[index,:,:,:])
        labels = self.labels[index]
        return inputs, labels

    def __len__(self):
        return self.n_images

def get_image_array_dataloader(
    image_array,
    labels,
    batch_size=256,
    transform=None,
    shuffle=False,
    num_workers=8,
    pin_memory=True
):
    """
    Inputs: 
        image_array : numpy array (N, H, W, 3)
        labels      : numpy array (N,)
        t           : torchvision.transforms instance

    Outputs:
        torch.utils.data.DataLoader for the image array
    """

    assert image_array.shape[0] == labels.shape[0]

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = ArrayDataset(image_array, labels, t=transform)
    dataloader = _acquire_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory=pin_memory)
    return dataloader


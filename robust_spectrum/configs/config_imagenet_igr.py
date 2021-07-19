"""
This is the training configuration file for training a ResNet-50 to minimize the
input gradient regularization loss.

Modify "image_dir" to the location of ImageNet.
Modify "exp_id" to a desired experiment name.
Modify "save_dir" to a desired checkpoint save directory.
"""

import torch.nn as nn
import torch.optim as optim

from robust_spectrum.models.imagenet_models.resnet import resnet50
from robust_spectrum.loss_functions.InputGradientLoss import InputGradientLoss

def load_config():
    config = dict()
    config["image_dir"] = "/mnt/fs5/nclkong/imagenet_raw/imagenet_raw/"
    config["exp_id"] = "igr_res50_reg0.3_test"
    config["save_dir"] = "/mnt/fs5/nclkong/trained_models/imagenet/robust_spectrum/"

    config["dataset"] = "imagenet_jpeg"
    config["num_workers"] = 8
    config["resume"] = ""
    config["batch_size"] = 128
    config["num_epochs"] = 100
    config["learning_rate"] = 0.1
    config["lr_stepsize"] = 30
    config["lr_gamma"] = 0.1
    config["scheduler"] = "multistep"
    config["milestones"] = [35,70,90]
    config["weight_decay"] = 0.0001
    config["momentum"] = 0.9
    config["model"] = resnet50()
    config["pretrained"] = False
    config["save_freq"] = 10
    config["pretrained_dir"] = ""

    # Image transforms
    config["image_transforms"] = None

    # Initialize optimizer object
    optimizer = optim.SGD(
        config["model"].parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    config["optimizer"] = optimizer

    # Initialize loss function
    loss_func = InputGradientLoss(
        lam=0.3,
        h=0.01,
        norm="L2",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    config["loss_func"] = loss_func

    # Initialize learning rate scheduler object
    if config["scheduler"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif config["scheduler"] == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["milestones"],
            gamma=config["lr_gamma"]
        )
    elif config["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_stepsize"],
            gamma=config["lr_gamma"]
        )
    elif config["scheduler"] == "constant": # gamma = 1 => no update to LR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    else:
        assert 0, "Scheduler {} is not implemented.".format(config["scheduler"])
    config["lr_scheduler"] = scheduler

    return config

if __name__ == "__main__":
    cfg = load_config()
    print(cfg)



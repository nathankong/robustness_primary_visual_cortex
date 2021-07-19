import os
import shutil
import random

import numpy as np

import torch
import torchvision.models
import torch.optim as optim


#======================================================
# Logging metrics for each epoch in training
#======================================================

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


#======================================================
# Reproducibility (random seed)
#======================================================

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#======================================================
# Metrics and other utility funcs
#======================================================

def compute_accuracy(output, target, topk=(1,)):
    """
    Adapted from PyTorch tutorial.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def save_checkpoint(state, save_dir, is_best, save_epoch):
    fname = save_dir + "/checkpoint.pt"

    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, save_dir+"/model_best.pt")

    if save_epoch is not None:
        shutil.copyfile(fname, save_dir+"/checkpoint_epoch_{}.pt".format(save_epoch+1))

def _load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        cpt = torch.load(checkpoint_path)
        return cpt
    else:
        print("No checkpoint at '{}'".format(checkpoint_path))
        assert 0

def load_checkpoint(checkpoint_path, model, optimizer):
    # Wrapper for _load_checkpoint()
    cpt = _load_checkpoint(checkpoint_path)

    train_params = cpt["train_params"]
    if train_params["scheduler"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    elif train_params["scheduler"] == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params["milestones"],
            gamma=train_params["lr_gamma"]
        )
    elif train_params["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_params["lr_stepsize"],
            gamma=train_params["lr_gamma"]
        )
    elif train_params["scheduler"] == "constant": # gamma = 1 => no update to LR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    else:
        assert 0, "Scheduler {} is not implemented.".format(train_params["scheduler"])

    model.load_state_dict(cpt["state_dict"]) # Load model
    optimizer.load_state_dict(cpt["optimizer"]) # Load optimizer
    scheduler.load_state_dict(cpt["lr_scheduler"]) # Load LR scheduler

    start_epoch_idx = cpt["epoch"] # Load epoch idx saved and start there
    best_acc = cpt["best_acc"] # Load best accuracy achieved so far
    results = cpt["results"] # Load dictionary of losses/accs
    save_dir = train_params["save_dir"] # Load dictionary where model is saved

    return (
        model, 
        optimizer, 
        train_params, 
        scheduler, 
        start_epoch_idx,
        best_acc,
        results,
        save_dir
    )

def check_best_accuracy(current_acc, best_acc):
    if current_acc > best_acc:
        return current_acc, True
    return best_acc, False


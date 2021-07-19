import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_spectrum.loss_functions.LossFunctionBase import LossFunctionBase

class InputGradientLoss(LossFunctionBase):
    """
    Loss function from "Scalable Input Gradient Regularization for Adversarial Robustness"
    Adapted from https://github.com/cfinlay/tulip/blob/master/imagenet/train.py
    https://arxiv.org/pdf/1905.11468.pdf

    Inputs:
        lam  : (float) regularization parameter for input gradient regularization
        h    : (float) step size for the finite difference approximation of the gradient
        norm : (string) which norm to use for normalizing the gradient
        mean : (list) data set specific mean (per channel) for normalization
        std  : (list) data set specific std (per channel) for normalization
    """
    def __init__(self, lam=1.0, h=0.01, norm="L2", mean=None, std=None):
        super(InputGradientLoss, self).__init__()
        assert mean is not None and std is not None
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none").cuda()

        mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
        self.mean = mean.cuda()
        self.std = std.cuda()

        self.lam = lam
        self.norm = norm
        self.h = h

    def loss_func(self, model, x, y):
        x.requires_grad_(True)

        # Compute \nabla_x \ell(x)
        preds = model((x - self.mean) / self.std)
        loss = self.cross_entropy_loss(preds, y)
        dx = torch.autograd.grad(loss.mean(), [x], retain_graph=True)[0]

        x.requires_grad_(False)

        # v is finite difference direction and then normalize
        sh = dx.shape
        v = dx.view(sh[0], -1)
        _, Nd = v.shape

        if self.norm == "L2":
            nv = v.norm(2, dim=-1, keepdim=True)
            nz = nv.view(-1) > 0 # find where norm is non-zero to avoid divide-by-zero
            v[nz] = v[nz].div(nv[nz])
        elif self.norm == "L1":
            v = v.sign()
            v = v / np.sqrt(Nd)
        else:
            raise ValueError(f"{self.norm} not implemented.")

        # Now get the input moved in the direction of v
        v = v.view(sh)
        xf = x + self.h * v

        # Calculate loss of moved input
        loss_xf = self.cross_entropy_loss(model((xf - self.mean) / self.std), y)

        # Calculate squared loss difference
        dl = ((loss_xf - loss) / self.h).pow(2)
        dl_mean = dl.mean() / 2.0

        # Penalized loss = loss + \lambda*gradient_loss
        penalized_loss = loss.mean() + self.lam * dl_mean

        return penalized_loss, preds

    def forward(self, model, inp, target, epoch=None):
        assert epoch is not None
        if model.training:
            if epoch < 0: # warm start without regularization
                print("Warm start...")
                preds = model((inp - self.mean) / self.std)
                loss = self.cross_entropy_loss(preds, target).mean()
            else:
                print("Train with regularization...")
                loss, preds = self.loss_func(model, inp, target)
        else:
            preds = model((inp - self.mean) / self.std)
            loss = self.cross_entropy_loss(preds, target).mean()

        return loss, preds


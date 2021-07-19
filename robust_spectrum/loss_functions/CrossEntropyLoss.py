import torch
import torch.nn as nn
import numpy as np

from robust_spectrum.loss_functions.LossFunctionBase import LossFunctionBase

class CrossEntropyLoss(LossFunctionBase):
    def __init__(self, mean=None, std=None):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        assert mean is not None and std is not None
        mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
        self.mean = mean.cuda()
        self.std = std.cuda()

    def forward(self, model, inp, target, **kwargs):
        preds = model((inp - self.mean) / self.std)
        loss = self.loss(preds, target)
        return loss, preds


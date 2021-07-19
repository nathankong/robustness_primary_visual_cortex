import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from robust_spectrum.loss_functions.LossFunctionBase import LossFunctionBase

class TradesLoss(LossFunctionBase):
    """
    Loss function from "Theoretically Principled Trade-off between Robustness and Accuracy"
    Adapted from https://github.com/yaodongyu/TRADES/
    http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf

    Inputs:
        optimizer     : (torch.optim optimizer) optimizer used in main training script
        mean          : (list) data set specific mean (per channel) for normalization
        std           : (list) data set specific std (per channel) for normalization
        step_size     : (float) step size for optimizing an adversary
        epsilon       : (float) maximum perturbation size of adversary
        perturb_steps : (int) number of steps for generating adversary
        beta          : (float) regularization coefficient for TRADES loss
        distance      : (string) L_p norm for the set of adversaries
    """
    def __init__(
        self,
        optimizer,
        mean=None,
        std=None,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        beta=1.0,
        distance="L_inf"
    ):
        super(TradesLoss, self).__init__()
        assert mean is not None and std is not None
        self.optimizer = optimizer

        mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
        std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
        self.mean = mean.cuda()
        self.std = std.cuda()

        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance

    def loss_func(self, model, x_natural, y):
        prev_training = bool(model.training)

        # Don't want batch norm / dropout here
        model.eval()

        kl_loss = nn.KLDivLoss(size_average=False)
        batch_size = len(x_natural)

        # Generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if self.distance == "L_inf":
            # Iterated FGSM (a.k.a. PGD)
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = kl_loss(
                        F.log_softmax(model((x_adv - self.mean) / self.std), dim=1),
                        F.softmax(model((x_natural - self.mean) / self.std), dim=1)
                    )

                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise NotImplementedError

        if prev_training:
            # Train mode (i.e. for updating batch norm and for drop out)
            model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # Zero gradient
        model.zero_grad()
        self.optimizer.zero_grad()

        # Calculate robust loss
        logits = model((x_natural - self.mean) / self.std)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * \
                        kl_loss(
                            F.log_softmax(model((x_adv - self.mean) / self.std), dim=1),
                            F.softmax(model((x_natural - self.mean) / self.std), dim=1)
                        )
        loss = loss_natural + self.beta * loss_robust

        return loss, logits

    def forward(self, model, inp, target, **kwargs):
        if model.training:
            loss, preds = self.loss_func(model, inp, target)
        else:
            print("Eval mode")
            preds = model((inp - self.mean) / self.std)
            loss = F.cross_entropy(preds, target)

        return loss, preds



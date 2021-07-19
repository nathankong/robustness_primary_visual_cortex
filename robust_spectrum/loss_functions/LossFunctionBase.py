import torch.nn as nn

class LossFunctionBase(nn.Module):
    def __init__(self):
        super(LossFunctionBase, self).__init__()

    def forward(self, model, inp, target, **kwargs):
        """
        Computes the loss function given a model, inputs and labels.

        Inputs:
            model  : (torch.nn.Module) the model being trained
            inp    : (torch.FloatTensor) (N, C, H, W); input images
            target : (torch.LongTensor) (N,); image labels

        Outputs:
            loss   : (torch.Tensor) scalar; loss value
            preds  : (torch.Tensor) (N, K); model predictions for K classes
        """
        raise NotImplementedError


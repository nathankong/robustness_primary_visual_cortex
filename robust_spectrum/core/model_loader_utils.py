import os

from robust_spectrum.core.default_dirs import TORCH_HOME
os.environ["TORCH_HOME"] = TORCH_HOME

import torch

import robust_spectrum.models.imagenet_models as imagenet_models

from robust_spectrum.models.model_layers import LAYERS

def get_model(arch_name, trained, dataparallel_wrap=True):
    """
    Inputs:
        arch_name : (string) Name of deep net architecture.
        trained   : (boolean) Whether or not to load a pretrained model.

    Outputs:
        model     : (torch.nn.DataParallel) model
    """
    try:
        print(f"Loading {arch_name}. Trained: {trained}.")
        if dataparallel_wrap:
            model = torch.nn.DataParallel(imagenet_models.__dict__[arch_name](pretrained=trained))
        else:
            model = imagenet_models.__dict__[arch_name](pretrained=trained)
    except:
        raise ValueError(f"{arch_name} not implemented yet.")

    return model

def load_model(arch_name, trained=False, model_path=None, dataparallel_wrap=True):
    """
    Inputs:
        arch_name  : (string) Name of architecture (e.g. "resnet18")
        trained    : (boolean) Whether to load a pretrained or trained model.
        model_path : (string) Path of model checkpoint from which to load
                     weights.
        dataparallel_wrap : (boolean) Whether to wrap the model in a DataParallel module

    Outputs:
        model      : (torch.nn.DataParallel) model
    """
    model = get_model(arch_name, trained=trained, dataparallel_wrap=dataparallel_wrap)

    # Load weights if params file is given.
    if model_path is not None:
        try:
            if torch.cuda.is_available():
                params = torch.load(model_path)
            else:
                params = torch.load(model_path, map_location="cpu")
        except:
            raise ValueError(f"Could not open file: {model_path}")

        assert "state_dict" in params.keys(), "'state_dict' not in params dictionary."
        sd = params["state_dict"]
        model.load_state_dict(sd)
        print(f"Loaded parameters from {model_path}")

    # Set model to eval mode
    model.eval()

    assert arch_name in LAYERS.keys(), f"Layers for {arch_name} not identified."
    layers = LAYERS[arch_name]

    return model, layers


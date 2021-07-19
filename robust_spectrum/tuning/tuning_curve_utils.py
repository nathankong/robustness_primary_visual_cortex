import torch

import numpy as np
import torchvision.transforms as transforms

from robust_spectrum.models.model_stats import MODEL_STATS
from robust_spectrum.core.dataloader_utils import get_image_array_dataloader
from robust_spectrum.neural_datasets.neural_dataset_utils import duplicate_channels
from robust_spectrum.core.feature_extractor import FeatureExtractor, get_layer_features

def get_stim_dataloader(stim, params, model_name):
    """
    Returns a dataloader for the stimuli.

    Inputs:
        stim       : (numpy.ndarray) of dimensions (num_stimuli, height, width)
                     for the Gabor or fullfield sine stimuli
        params     : (numpy.ndarray) where each entry contains an array of the
                     parameter values for the respective stimulus. The array contains
                     parameters in the order: (orientation, spatial frequency, phase)
        model_name : (string) identifier for the architecture

    Outputs:
        dataloader   : (torch.utils.data.Dataloader) for the stimuli
    """
    assert stim.shape[1] == stim.shape[2] == 224, \
        "Currently only support image size of 224 px."
    assert model_name in MODEL_STATS.keys(), f"{model_name} is non-existent."
    # Currently support three Gabor / fullfield sine parameters
    assert params.ndim == 2 and params.shape[1] == 3
    assert params.shape[0] == stim.shape[0], \
            "Number of stimuli does not match the number of metadata entries. " \
            f"Given: {stim.shape[0]} stimuli vs. {params.shape[0]} params"

    # Normalization for the model
    img_mean = MODEL_STATS[model_name]["mean"]
    img_std = MODEL_STATS[model_name]["std"]

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=img_mean, std=img_std
        )
    ])

    images = duplicate_channels(stim)
    labels = params

    dataloader = get_image_array_dataloader(
        images,
        labels,
        batch_size=256,
        transform=img_transforms,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    return dataloader

def get_model_features(dataloader, model, layer_name, vectorize=True):
    """
    This function obtains the model features for a particular layer  given
    the dataloader. Assumes that layer_name is a valid layer for the model.
    Cross-check with ../models/model_layers.py.

    Inputs:
        dataloader : (torch.utils.data.Dataloader) dataloader for the stimuli
        model      : (torch.nn.Module) model object
        layer_name : (string) layer of the model from which to extract
                     activations
        vectorize  : (boolean) whether to flatten activations. Default: True.

    Outputs:
        features   : (numpy.ndarray) array of flattened activations associated
                     with a particular layer of the model.
    """
    # Get feature extractor for stimuli
    feature_extractor= FeatureExtractor(
        dataloader,
        n_batches=None,
        vectorize=vectorize,
        debug=False
    )

    # Extract features for a particular layer
    features = get_layer_features(feature_extractor, layer_name, model)
    return features


if __name__ == "__main__":
    import numpy as np

    from robust_spectrum.core.model_loader_utils import load_model
    from robust_spectrum.tuning.gabor_utils import generate_gabors
    from robust_spectrum.tuning.constants import PARAM_DICT

    angles = PARAM_DICT["orientations"]
    sfs = PARAM_DICT["spatial_frequencies"]
    phases = PARAM_DICT["phases"]
    gbs, params = generate_gabors(angles, sfs, phases)

    model_name = "alexnet"
    model, _ = load_model(model_name, trained=True)
    dl = get_stim_dataloader(gbs, params.values, model_name)
    features = get_model_features(dl, model, "features.0")
    print(features.shape)



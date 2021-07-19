"""
This file contains the function that is used to obtain neural datasets given
a dataset name. It is standalone since I don't want to move it to neural_dataset_utils.py
and because this function could get long depending on the number of neural datasets.
"""

from robust_spectrum.neural_datasets.stringer_dataset import StringerDataset
from robust_spectrum.neural_datasets.cadena_dataset import CadenaDataset
from robust_spectrum.core.default_dirs import \
    STRINGER_DATASET_STIM, \
    STRINGER_DATASET_RESP, \
    CADENA_NATURAL_DATASET_STIM, \
    CADENA_NATURAL_DATASET_RESP, \
    CADENA_ALL_DATASET_STIM, \
    CADENA_ALL_DATASET_RESP

def get_neural_dataset(dataset_name, img_mean=None, img_std=None, model_input_size=None):
    """
    Inputs:
        img_mean         : (list) normalization mean for images for the three channels
        img_std          : (list) normalization std for images for the three channels
        model_input_size : (tuple) required input image size for the model. (height, width)
    """
    # Neural responses and stimulus dataloader
    if dataset_name.lower() == "stringer":
        print("Using Stringer et al. 2019 neural responses, mouse V1.")

        dataset = StringerDataset(
            STRINGER_DATASET_RESP,
            STRINGER_DATASET_STIM,
            img_mean=img_mean,
            img_std=img_std,
            model_input_size=model_input_size
        )

    elif dataset_name.lower() == "cadena_natural":
        print("Using Cadena et al. 2019 neural responses from natural scenes stimuli.")

        dataset = CadenaDataset(
            CADENA_NATURAL_DATASET_RESP,
            CADENA_NATURAL_DATASET_STIM,
            img_mean=img_mean,
            img_std=img_std,
            model_input_size=model_input_size
        )

    elif dataset_name.lower() == "cadena_all":
        print("Using Cadena et al. 2019 neural responses from all scenes stimuli, macaque V1.")

        dataset = CadenaDataset(
            CADENA_ALL_DATASET_RESP,
            CADENA_ALL_DATASET_STIM,
            img_mean=img_mean,
            img_std=img_std,
            model_input_size=model_input_size
        )

    else:
        raise ValueError(f"{dataset_name} does not exist.")

    return dataset


if __name__ == "__main__":
    dataset = get_neural_dataset("stringer", img_mean=[0,0,0], img_std=[1,1,1], model_input_size=None)
    print("Stringer")
    print(dataset.get_trial_averaged_neural_responses().shape)
    print(len(dataset.get_stim_dataloader()))

    dataset = get_neural_dataset("cadena_natural", img_mean=[0,0,0], img_std=[1,1,1], model_input_size=None)
    print("Cadena natural")
    print(dataset.get_trial_averaged_neural_responses().shape)
    print(len(dataset.get_stim_dataloader()))

    dataset = get_neural_dataset("cadena_all", img_mean=[0,0,0], img_std=[1,1,1], model_input_size=None)
    print("Cadena all")
    print(dataset.get_trial_averaged_neural_responses().shape)
    print(len(dataset.get_stim_dataloader()))


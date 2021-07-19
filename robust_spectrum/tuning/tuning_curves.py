import numpy as np
import torchvision.transforms as transforms

from robust_spectrum.tuning.gabor_utils import generate_gabors
from robust_spectrum.tuning.tuning_curve_utils import get_stim_dataloader
from robust_spectrum.tuning.tuning_curve_utils import get_model_features

class TuningCurves():
    """
    Class for computing tuning curves of artificial neurons.

    Arguments:
        param_dict : (dict) where key is parameter name and value is list of
                     parameter values. There should be three keys:
                     "orientations", "spatial_frequencies", "phases"
        model_name : (string) model architecture identifier
        model      : (torch.nn.Module) object for the model
        stim_type  : (string) stimulus type to probe model tuning properties.
                     Choices are: ["gabor"]
    """
    def __init__(
        self, 
        param_dict, 
        model_name, 
        model, 
        stim_type="gabor",
    ):
        assert stim_type in ["gabor"], f"{stim_type} is an invalid type."
        assert set(param_dict.keys()) == \
            set(["orientations", "spatial_frequencies", "phases"]), \
            f"Incorrect Gabor parameters. Given {param_dict.keys()}."

        self.activations = None

        self.stim_type = stim_type
        self.param_dict = param_dict
        self.model_name = model_name
        self.model = model

        self.dataloader, self.params = self._get_stim_dataloader()
        assert set(param_dict.keys()) == set(self.params.columns)

    def _get_stim_dataloader(self):
        """
        Generates Gabor stimuli given the orientation, spatial frequency and
        phase parameters. Then returns a dataloader for those stimuli.
        """
        # Generate stimuli and get their respective parameters (i.e., labels)
        angles = self.param_dict["orientations"]
        sfs = self.param_dict["spatial_frequencies"]
        phases = self.param_dict["phases"]

        print(f"Generating Gabors and getting dataloader...")
        if self.stim_type == "gabor":
            stim, params = generate_gabors(angles, sfs, phases)
        else:
            raise ValueError(f"{self.stim_type} is not supported.")

        # Get Gabor dataloader
        dataloader = get_stim_dataloader(stim, params.values, self.model_name)

        return dataloader, params

    def _extract_features(self, layer_name, vectorize=True):
        """
        Extracts model features for a layer called layer_name associated
        with the Gabor stimuli.
        """
        # Activations for Gabor patch stimuli
        stim_features = get_model_features(
            self.dataloader,
            self.model,
            layer_name,
            vectorize=vectorize
        )

        return stim_features

    def compute_tuning_curves(self, layer_name, unvectorize=True):
        """
        For a layer of the model, compute the tuning curves for each neuroid.

        Inputs:
            layer_name    : (string) identifier for the layer from which to analyze 
                            tuning
            unvectorize   : (boolean) whether to convert the vectorized activations back
                            to its original shape

        Outputs:
            tuning_curves : (dict) where the key is the parameter name and the value
                            is a numpy array of tuning of shape (num_param_values,
                            n_neuroids).
        """
        # Extract activations (num_stimuli, num_activations)
        activations = self._extract_features(layer_name, vectorize=False)
        original_shape = list(activations.shape)
        activations = np.reshape(activations, [activations.shape[0], -1])
        assert activations.ndim == 2
        feature_slice = activations

        n_neuroids = feature_slice.shape[1]
        print(f"Activations: {feature_slice.shape}")

        self.activations = activations

        # Compute tuning curves for each neuroid for each Gabor parameter
        tuning_curves = dict()
        for param, param_vals in self.param_dict.items():

            assert param_vals.ndim == 1
            n_params = param_vals.shape[0]

            # This data structure will contain tuning for each neuroid
            param_activations = np.zeros((n_params,n_neuroids))

            # For each parameter value, marginalize over the other parameters
            for i, param_val in enumerate(param_vals):
                curr_features = feature_slice[self.params[param].values == param_val,:]
                param_activations[i,:] = curr_features.mean(axis=0)

            if unvectorize:
                assert np.prod(original_shape[1:]) == param_activations.shape[1]
                original_shape[0] = param_activations.shape[0]
                tuning_curves[param] = np.reshape(param_activations, original_shape)
            else:
                tuning_curves[param] = param_activations

        return tuning_curves



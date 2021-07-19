import os

from robust_spectrum.core.dataloader_utils import get_image_array_dataloader

class NeuralDatasetBase():
    def __init__(self, responses_fname, stimuli_fname, img_mean=None, img_std=None):
        """
        Inputs:
            responses_fname : (string) file path to the neural responses
            stimuli_fname   : (string) file path to the stimuli (images)
            img_mean        : (list) three floats, the values should be the means that
                              were used when normalizing inputs for training the model
            img_std         : (list) three floats, the values should be the stds that
                              were used when normalizing inputs for training the model
        """
        assert os.path.isfile(responses_fname)
        assert os.path.isfile(stimuli_fname)
        assert img_mean is not None and img_std is not None

        self.responses_fname = responses_fname
        self.stimuli_fname = stimuli_fname
        self.img_mean = img_mean
        self.img_std = img_std
        self.trial_avg_neural_responses = None
        self.by_trial_neural_responses = None
        self.stim_dataloader = None

        # Parameters for fitting power law exponent after computing the neural
        # response eigenspectrum.
        self.spectrum_fit_range = None
        self.spectrum_n_comp = None # Basically, number of components for cvPCA

    def load_responses(self):
        """
        Outputs:
            responses : numpy array of dimensions (2 trials, n_stim, n_neurons)
            labels    : numpy array of stim labels / indices
        """
        raise NotImplementedError

    def load_stimuli(self):
        """
        Outputs:
            dataloader : torch.utils.data.DataLoader for the stimuli
        """
        raise NotImplementedError

    def get_spectrum_fit_range(self):
        assert self.spectrum_fit_range is not None
        return self.spectrum_fit_range

    def get_spectrum_n_comp(self):
        assert self.spectrum_n_comp is not None
        return self.spectrum_n_comp

    def get_trial_averaged_neural_responses(self):
        """
        Neural responses of this format are used for neural fitting

        Outputs:
            neural_responses : numpy array of dimensions (n_stim, n_neurons)
        """
        assert self.trial_avg_neural_responses is not None
        assert self.trial_avg_neural_responses.ndim == 2
        return self.trial_avg_neural_responses

    def get_by_trial_neural_responses(self):
        """
        Neural responses of this format are used for reliability calculations

        Outputs:
            neural_responses : numpy array of dimensions (n_trial, n_stim, n_neurons)
                               or list of (n_trial, n_stim, n_neurons)
        """
        assert self.by_trial_neural_responses is not None

        if not isinstance(self.by_trial_neural_responses, list):
            assert self.by_trial_neural_responses.ndim == 3
        else:
            for d in self.by_trial_neural_responses:
                assert d.ndim == 3

        return self.by_trial_neural_responses

    def get_stim_dataloader(self):
        """
        Outputs:
            stim_dataloader : torch.utils.data.DataLoader for the stimuli
        """
        assert self.stim_dataloader is not None
        assert len(self.stim_dataloader.dataset) == self.trial_avg_neural_responses.shape[0]
        return self.stim_dataloader

    def get_dataloader(self, images, labels, image_transforms):
        """
        Inputs:
            images           : numpy array of dimensions (n_stim, height, width, 3)
            labels           : numpy array of dimensions (n_stim,)
            image_transforms : torchvision.transforms composition

        Output:
            dataloader : torch.utils.data.DataLoader for stimuli
        """
        assert images.shape[0] == labels.shape[0]
        assert images.ndim == 4
        assert images.shape[3] == 3
        assert image_transforms is not None

        # Construct dataset and dataloader
        dataloader = get_image_array_dataloader(
            images,
            labels,
            batch_size=256,
            transform=image_transforms,
            num_workers=8,
            shuffle=False,
            pin_memory=True
        )
        return dataloader


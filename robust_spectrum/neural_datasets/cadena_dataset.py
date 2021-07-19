import numpy as np

from torchvision import transforms

from robust_spectrum.neural_datasets.neural_dataset_base import NeuralDatasetBase
from robust_spectrum.neural_datasets.neural_dataset_utils import duplicate_channels

class CadenaDataset(NeuralDatasetBase):
    def __init__(self, responses_fname, stimuli_fname, img_mean=None, img_std=None, model_input_size=None):
        super(CadenaDataset, self).__init__(
            responses_fname,
            stimuli_fname,
            img_mean=img_mean,
            img_std=img_std
        )
        if model_input_size is None:
            self.pad_amount = (0,0,0,0) # left, top, right, bottom
        else:
            # NOTE: here we assume that the stimulus is 40x40
            input_height = model_input_size[0]
            input_width = model_input_size[1]
            pad_left = int(np.floor((input_width-40)/2.))
            pad_top = int(np.floor((input_height-40)/2.))
            pad_right = int(np.ceil((input_width-40)/2.))
            pad_bottom = int(np.ceil((input_height-40)/2.))
            self.pad_amount = (pad_left, pad_top, pad_right, pad_bottom)

        self.by_trial_neural_responses, self.trial_avg_neural_responses, _ = self.load_responses()
        self.stim_dataloader = self.load_stimuli()

        # Parameters for fitting power law exponent after computing the neural
        # response eigenspectrum.
        self.spectrum_fit_range = np.arange(8,100)
        self.spectrum_n_comp = 150

    def load_responses(self):
        """
        Assumes data set is already processed into dimensions (2 trials, 1250 stim, 
        166). The stimuli are the natural scenes from the stimulus data set. Some of
        the natural scenes were removed since there were not enough trials for them.

        Output:
            responses      : numpy array of dimensions (2 trials, 1250 stim, 166 neurons)
            trial_avg_data : numpy array of dimensions (n_stim, n_neurons)
            labels         : numpy array of stim labels / indices
        """
        responses = np.load(self.responses_fname)
        assert responses.shape[0] == 2 and responses.shape[2] == 166
        n_stim = responses.shape[1]
        labels = np.arange(n_stim).astype(int)

        print(responses.shape)

        trial_avg_data = responses.mean(axis=0)
        return responses, trial_avg_data, labels

    def load_stimuli(self):
        """
        Cropping to images to 80 px (as done in Cadena et al. 2019), which is about 
        1.1 degrees in monkey "land".  Then resize the stimuli to 40 px (assuming 
        that 224 px in deep net "land" is 6.4 degrees. So 40 px is about 1.1 degrees.

        Output:
            dataloader : torch.utils.data.DataLoader for the stimuli
        """
        stim = np.load(self.stimuli_fname)
        assert stim.ndim == 3
        stim = duplicate_channels(stim)
        labels = np.arange(stim.shape[0]).astype(np.int)

        # Construct transforms necessary for neural fits
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(80),
            transforms.Resize(40),
            transforms.Pad(self.pad_amount, fill=0, padding_mode="constant"),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.img_mean, std=self.img_std
            )
        ])

        dataloader = self.get_dataloader(stim, labels, img_transforms)
        return dataloader


if __name__ == "__main__":
    from robust_spectrum.core.default_dirs import CADENA_ALL_DATASET_STIM, CADENA_ALL_DATASET_RESP

    c = CadenaDataset(
        CADENA_ALL_DATASET_RESP,
        CADENA_ALL_DATASET_STIM,
        img_mean=[0,0,0],
        img_std=[1,1,1]
    )

    r = c.get_trial_averaged_neural_responses()
    print(r.shape)

    b = c.get_by_trial_neural_responses()
    print(b.shape)

    dl = c.get_stim_dataloader()
    print(len(dl))


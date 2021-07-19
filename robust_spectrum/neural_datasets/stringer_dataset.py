import numpy as np
import scipy.io as io

from torchvision import transforms

from robust_spectrum.neural_datasets.neural_dataset_base import NeuralDatasetBase
from robust_spectrum.neural_datasets.neural_dataset_utils import \
    duplicate_channels, \
    subtract_32d_spont, \
    split_into_two_repeats

class StringerDataset(NeuralDatasetBase):
    def __init__(self, responses_fname, stimuli_fname, img_mean=None, img_std=None, model_input_size=None):
        super(StringerDataset, self).__init__(
            responses_fname,
            stimuli_fname,
            img_mean=img_mean,
            img_std=img_std
        )
        if model_input_size is None:
            self.pad_amount = (0,0,0,0) # left, top, right, bottom
        else:
            # NOTE: here we assume that the stimulus is 68x100
            input_height = model_input_size[0]
            input_width = model_input_size[1]
            pad_left = int(np.floor((input_width-100)/2.))
            pad_top = int(np.floor((input_height-68)/2.))
            pad_right = int(np.ceil((input_width-100)/2.))
            pad_bottom = int(np.ceil((input_height-68)/2.))
            self.pad_amount = (pad_left, pad_top, pad_right, pad_bottom)

        self.by_trial_neural_responses, self.trial_avg_neural_responses, _ = self.load_responses()
        self.stim_dataloader = self.load_stimuli()

        # Parameters for fitting power law exponent after computing the neural
        # response eigenspectrum.
        self.spectrum_fit_range = np.arange(11,500)
        self.spectrum_n_comp = 2000

    def load_responses(self):
        """
        Loads neural responses from Stringer et al. 2019. Subtracts 32D of
        spontaneous activity and constructs neural response data set where each
        stimuli has two trials.

        Outputs:
            responses      : numpy array of dimensions (2, n_stim, n_neurons)
            trial_avg_data : numpy array of dimensions (n_stim, n_neurons)
            labels         : numpy array of stim labels / indices
        """
        dat = io.loadmat(self.responses_fname)
        resp, nimg, istim = subtract_32d_spont(dat)
        responses = split_into_two_repeats(resp, nimg, istim)
        assert responses.shape[0] == 2
        labels = np.arange(responses.shape[1]).astype(np.int)

        print(responses.shape)

        trial_avg_data = responses.mean(axis=0)
        return responses, trial_avg_data, labels

    def load_stimuli(self):
        """
        Loads and crops the natural image data set used in Stringer et al. 2019.
        Assumes that the stimuli are of dimensions (68,270,2800) (height, width, 
        n_stimuli).

        Output:
            dataloader : torch.utils.data.DataLoader for Stringer data set stimuli.
        """
        imgs = io.loadmat(self.stimuli_fname)
        stim = imgs["imgs"] # (68,270,2800)
        assert stim.shape == (68,270,2800)
        stim = np.transpose(stim, (2,0,1))

        stim = self._crop_68x100_stim(stim)
        assert stim.shape == (2800,68,100)
        stim = duplicate_channels(stim)
        labels = np.arange(2800).astype(np.int)

        # Construct transforms necessary for neural fits
        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(self.pad_amount, fill=0, padding_mode="constant"),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.img_mean, std=self.img_std
            )
        ])

        dataloader = self.get_dataloader(stim, labels, img_transforms)
        return dataloader

    def _crop_68x100_stim(self, imgs):
        """
        Crops the images into the region where the V1 receptive fields were located.
        NOTE: This is not rigorously done--was eyeballed from the receptive field 
        locations visualized in the publication.

        Input:
            imgs : numpy array of dimensions (2800,68,270)

        Output:
            imgs : numpy array of dimensions (2800,68,100)
        """
        assert imgs.shape == (2800,68,270)
        return imgs[:,:,50:150]

if __name__ == "__main__":
    from robust_spectrum.core.default_dirs import STRINGER_DATASET_STIM, STRINGER_DATASET_RESP

    s = StringerDataset(
        STRINGER_DATASET_RESP,
        STRINGER_DATASET_STIM,
        img_mean=[0,0,0],
        img_std=[1,1,1]
    )

    r  = s.get_trial_averaged_neural_responses()
    print(r.shape)

    d  = s.get_by_trial_neural_responses()
    print(d.shape)

    dl = s.get_stim_dataloader()
    print(len(dl))


import pickle

import numpy as np

from robust_spectrum.core.default_dirs import \
    CADENA_ALL_DATASET_STIM, \
    CADENA_ALL_DATASET_RESP, \
    CADENA_NATURAL_DATASET_RESP, \
    CADENA_RAW_DATA

def split_into_two_trials_cadena(responses):
    """
    Since some neurons only have two trials, we find the images that have at least
    two trials for all the neurons (the trials where the neuron does not have a 
    response is coded as `NaN' in the data set. We then organize the responses 
    into a numpy array of dimensions (2 trials, n_stim, n_neurons).

    Input:
        responses     : numpy array of dimensions (4 trials, n_stim, n_neurons)
        valid_img_idx : numpy array of booleans indicating which stimuli were
                        included and which were removed (removed if there are less
                        than two trials for the image)
    """
    valid_img_idx = np.ones((responses.shape[1],)).astype(bool)
    responses_by_trial = list()
    for i in range(responses.shape[1]):
        by_trial = list()
        for j in range(responses.shape[0]):
            if np.sum(np.isnan(responses[j,i,:])) == 0:
                by_trial.append(responses[j,i,:])
        # If the number of trials for stim is less than 2, remove the stim
        if (len(by_trial) < 2):
            print(i)
            valid_img_idx[i] = 0
        # Otherwise, average each half of all the trials and create two
        # `pseudo-trials' for the stim
        else:
            by_trial = np.array(by_trial)
            half_trials = int(len(by_trial)/2)
            r1 = np.mean(by_trial[:half_trials,:], axis=0)[np.newaxis,:]
            r2 = np.mean(by_trial[half_trials:,:], axis=0)[np.newaxis,:]
            responses_by_trial.append(np.vstack((r1,r2)))

    responses_by_trial = np.transpose(np.array(responses_by_trial), (1,0,2))
    return responses_by_trial, valid_img_idx

img_type = "original"
cadena_data = pickle.load(open(CADENA_RAW_DATA, "rb"))

img_idx = np.array(cadena_data["image_types"] == img_type).flatten()
responses = cadena_data["responses"][:,img_idx,:]
responses, valid_img_idx = split_into_two_trials_cadena(responses)
print(responses.shape)
_r = np.load(CADENA_NATURAL_DATASET_RESP)
assert np.array_equal(_r, responses)

all_responses = []
all_stims = []
img_types = ["conv1", "conv2", "conv3", "conv4", "original"]
for img_type in img_types:
    img_idx = np.array(cadena_data["image_types"] == img_type).flatten()
    responses = cadena_data["responses"][:,img_idx,:]
    responses, valid_img_idx = split_into_two_trials_cadena(responses)
    all_responses.append(responses)
    imgs = cadena_data["images"][img_idx,:,:][valid_img_idx]
    all_stims.append(imgs)

all_responses = np.concatenate(all_responses, axis=1)
all_stims = np.concatenate(all_stims, axis=0)
print(all_responses.shape)
print(all_stims.shape)
_r = np.load(CADENA_ALL_DATASET_RESP)
_s = np.load(CADENA_ALL_DATASET_STIM)
assert np.array_equal(_r, all_responses)
assert np.array_equal(_s, all_stims)



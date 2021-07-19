import numpy as np

from scipy.sparse.linalg import eigsh

#==================================================
# Code adapted from:
# https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/powerlaws.ipynb
#==================================================

def subtract_32d_spont(dat):
    """
    Given datafile from loaded from mat file, removes 32-dimensional subspace
    of spontaneous activity.

    Returns:
        resp : numpy array of dimensions (n_stim, n_neurons)
    """

    resp = dat['stim'][0]['resp'][0] # stim x neurons
    spont = dat['stim'][0]['spont'][0] # timepts x neurons
    istim = (dat['stim'][0]['istim'][0]).astype(np.int32) # stim ids 
    istim -= 1 # get out of MATLAB convention
    istim = istim[:,0]
    nimg = istim.max() # these are blank stims (exclude them)
    resp = resp[istim<nimg, :]
    istim = istim[istim<nimg]

    # subtract spont (32D)
    mu = spont.mean(axis=0)
    sd = spont.std(axis=0) + 1e-6
    resp = (resp - mu) / sd
    spont = (spont - mu) / sd
    sv,u = eigsh(spont.T @ spont, k=32)
    resp = resp - (resp @ u) @ u.T

    return resp, nimg, istim

def split_into_two_repeats(resp, nimg, istim):
    """
    Splits responses into to image repeats.

    Returns:
        sresp : split responses of dimensions (2, n_stim, n_neurons)
    """

    # split stimuli into two repeats
    NN = resp.shape[1]
    sresp = np.zeros((2, nimg, NN), np.float64)
    inan = np.zeros((nimg,), np.bool)
    for n in range(nimg):
        ist = (istim==n).nonzero()[0]
        i1 = ist[:int(ist.size/2)]
        i2 = ist[int(ist.size/2):]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, i1.size < 1):
            inan[n] = 1
        else:
            sresp[0, n, :] = resp[i1, :].mean(axis=0)
            sresp[1, n, :] = resp[i2, :].mean(axis=0)
            
    # remove image responses without two repeats
    sresp = sresp[:,~inan,:]

    return sresp

def generate_train_test_split(n, ratio, seed=0):
    """
    Generates a random train-test split for neural response predictions.
    """

    np.random.seed(seed)
    sh = np.random.permutation(range(n))
    return sh[:int(len(sh)*(ratio))], sh[int(len(sh)*(ratio)):]

def duplicate_channels(gray_images):
    """
    Converts single channel grayscale images into rgb channel images

    Input:
        gray_images : (N,H,W)

    Output:
        rgb : (N,H,W,3)
    """

    n, dim0, dim1 = gray_images.shape[:3]
    rgb = np.empty((n, dim0, dim1, 3), dtype=np.uint8)
    rgb[:, :, :, 2] = rgb[:, :, :, 1] = rgb[:, :, :, 0] = gray_images
    return rgb



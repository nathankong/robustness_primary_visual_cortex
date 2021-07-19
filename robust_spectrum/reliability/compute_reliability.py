import numpy as np

from scipy import stats

def compute_split_half_reliability(neural_data):
    """
    Computes the split-half reliability for each neuron.

    Input:
        neural_data : (numpy.ndarray) of dimensions (n_trial, n_stim, n_neurons)
                      or (list of numpy.ndarray) [(n_trial, n_stim, n_neurons), ...]

    Output:
        ccs : (numpy.ndarray) of Pearson correlations for each neuron of dimensions
              (n_neurons,)
    """
    if isinstance(neural_data, list):
        ccs = _compute_reliability_list(neural_data)
    else:
        ccs = _compute_reliability_array(neural_data)

    return ccs

def _compute_reliability_list(neural_data, n_bootstrap_iter=200):
    """
    Inputs:
        neural_data      : (list) of numpy arrays of dimensions
                           [(n_trial, n_stim, n_neurons), ...]
        n_bootstrap_iter : (int) number of bootstrap iterations for split-half
                           reliability

    Outputs:
        ccs : (numpy.ndarray) of length n_neurons of reliabilities
    """
    assert isinstance(neural_data, list)

    NN = neural_data[0].shape[2]
    # Make sure same number of neurons in all arrays (since we extend along stim dimension)
    for d in neural_data:
        assert d.shape[2] == NN, f"Number of neurons is {NN}, found {d.shape[2]}."
    
    ccs = []
    for n in range(NN):
        print(f"Neuron {n}...")
        neuron_data = [X[:,:,n] for X in neural_data]
        cc = _compute_reliability_per_neuron(neuron_data, num_bootstrap=n_bootstrap_iter)
        assert cc.shape[0] == n_bootstrap_iter and cc.ndim == 1
        cc = np.mean(cc)
        ccs.append(cc)

    return np.array(ccs)

def _compute_reliability_array(neural_data, n_bootstrap_iter=100):
    """
    Inputs:
        neural_data      : (numpy.ndarray) of dimensions (n_trial, n_stim, n_neurons)
        n_bootstrap_iter : (int) number of bootstrap iterations for split-half reliability

    Outputs:
        ccs : (numpy.ndarray) of length n_neurons of reliabilities
    """
    assert neural_data.ndim == 3

    n_trials = neural_data.shape[0]
    NN = neural_data.shape[2]
    ccs = []
    for n in range(NN):
        if n_trials == 2:
            s1 = neural_data[0,:,n]
            s2 = neural_data[1,:,n]
            cc, _ = stats.pearsonr(s1, s2)
        else:
            assert n_trials > 2
            cc = _compute_reliability_per_neuron([neural_data[:,:,n]], num_bootstrap=n_bootstrap_iter)
            assert cc.shape[0] == n_bootstrap_iter and cc.ndim == 1
            cc = np.mean(cc)
        ccs.append(cc)

    return np.array(ccs)

def _compute_reliability_per_neuron(neural_data, num_bootstrap=200):
    """
    Inputs:
        neural_data : (list of numpy.ndarray) of data for one neuron of dimensions
                      [(n_trials, n_stim), ...]

    Outputs:
        ccs : (list) of split-half correlations for each bootstrap iter
    """
    assert isinstance(neural_data, list)

    def split_half(X, seed=0):
        # X dimensions: (n_trials, n_stim)
        assert X.ndim == 2
        n_trials = X.shape[0]
        idxs = np.arange(n_trials)
        np.random.RandomState(seed).shuffle(idxs)
        half = int(n_trials / 2.0)
        half1 = idxs[:half]
        half2 = idxs[half:]
        return X[half1,:].mean(axis=0), X[half2,:].mean(axis=0)

    # First gather randomly shuffled trials per bootstrap iteration
    Xsmp, Ysmp = [], []
    for i in range(num_bootstrap):
        xs, ys = [], []
        for X in neural_data:
            x, y = split_half(X, i)
            xs.extend(x)
            ys.extend(y)
        xs, ys = np.array(xs), np.array(ys)

        Xsmp.append(xs)
        Ysmp.append(ys)

    # Do correlations
    ccs = list()
    for x, y in zip(Xsmp, Ysmp):
        cc, _ = stats.pearsonr(x, y)
        ccs.append(cc)

    return np.array(ccs)


if __name__ == "__main__":
    from robust_spectrum.core.default_dirs import CADENA_ALL_DATASET_RESP

    neural_data = np.load(CADENA_ALL_DATASET_RESP)
    ccs = compute_split_half_reliability(neural_data)
    med_cc = np.median(ccs)

    print(f"Reliability dimensions: {ccs.shape}")
    print(f"Median reliability: {med_cc:.3f}")
    print(f"Spearman-Brown corrected median reliability: {2*med_cc / (1 + med_cc):.3f}")


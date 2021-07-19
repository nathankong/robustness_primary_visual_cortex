import os
import pickle

import numpy as np

from robust_spectrum.tuning.constants import PARAM_DICT
from robust_spectrum.models.model_layers import LAYERS


#=================================================================
# Functions to help obtain model neural fit and tuning curve data
#=================================================================

def aggregate_neural_results(layers, model_name, fname_descriptor):
    """
    Obtains neural fit results that have been saved previously.
    """

    all_corrs = list()
    for i, l in enumerate(layers):
        fname = (fname_descriptor + "/{}/{}.pkl").format(model_name, l)
        results = pickle.load(open(fname, "rb"), encoding="latin1")
        
        # Indices of the selected neurons for regression
        select_neuron_ids = results["select_neuron_ids"]
        
        # Spearman-Brown correct the reliabilities of the chosen neurons for analyses first
        reliabilities = results["reliability"][select_neuron_ids]
        reliabilities = np.divide((2*reliabilities), (1+reliabilities))
        
        # Normalized correlation is correlation divide by sqrt of reliability of neuron
        assert results["correlations"][:,1,:].shape[1] == reliabilities.size
        assert reliabilities.ndim == 1
        normed_corrs = results["correlations"][:,1,:] / np.sqrt(reliabilities)
        
        # Record the median normalized correlations for each split and compute the mean
        # and std of those correlation values across splits.
        assert results["correlations"].shape[2] == reliabilities.shape[0] == select_neuron_ids.sum()
        
        if len(all_corrs) != 0:
            assert normed_corrs.shape == all_corrs[-1].shape
        all_corrs.append(normed_corrs)

    # Shape: (num_layers, num_splits, num_neurons)
    return np.array(all_corrs)

def get_best_neural_fit_layer(model_name, neural_fit_results_dir):
    """
    Obtains the model layer that provided the best neural fits.
    """

    layers = LAYERS[model_name]
    corrs = aggregate_neural_results(layers, model_name, neural_fit_results_dir)
    
    assert corrs.ndim == 3
    median_corrs = np.median(corrs, axis=2)
    avg_median_corrs = np.mean(median_corrs, axis=1)
    
    best_fit_idx = np.argmax(avg_median_corrs)
    best_fit_layer_name = layers[best_fit_idx]
    
    return best_fit_layer_name

def get_preferred_sf_distribution(
    model_name,
    stim_type,
    neural_fit_results_dir,
    tuning_fit_results_dir,
    n_samples=None,
    n_experiments=None
):
    """
    Returns preferred frequency distribution. If n_experiments is None, returns 
    distribution across all channels and returns a matrix of shape: (n_channels,).
    Otherwise, samples n_samples neurons across all channels and returns a 
    two-dimensional matrix of shape: (n_experiments, n_samples)
    """
    
    # Get most V1-like model layer
    best_layer = get_best_neural_fit_layer(model_name, neural_fit_results_dir)

    # Get distributions of preferred spatial frequency
    sf_peaks_distribution = get_tuning_of_center_neuron(
        tuning_fit_results_dir,
        model_name,
        best_layer,
        stim_type,
        n_samples=n_samples,
        n_experiments=n_experiments
    )
    
    return sf_peaks_distribution

def get_tuning_of_center_neuron(
    tuning_results_dir,
    model_name,
    layer_name,
    stim_type,
    n_samples=None,
    n_experiments=None
):
    """
    Returns preferred frequency distribution for the neuron in the center of the
    activations matrix. If n_experiments is None, returns distribution across all 
    channels and returns a matrix of shape: (n_channels,). Otherwise, samples 
    n_samples neurons across all channels and returns a two-dimensional matrix of 
    shape: (n_experiments, n_samples)
    """
    
    _sf_tuning = get_center_neuron_sf_tuning_curves(
        tuning_results_dir,
        model_name,
        layer_name,
        stim_type
    )
    
    # Only use channels where peak-to-peak is larger than 0.0
    pass_idx = (np.ptp(_sf_tuning, axis=0) > 0.0)
    
    sf_peaks = np.argmax(_sf_tuning[:,pass_idx], axis=0)
    sf_peaks = PARAM_DICT["spatial_frequencies"][sf_peaks] # in units cycles per image
    assert sf_peaks.shape[0] == pass_idx.sum()
    
    if n_experiments is not None:
        assert n_samples is not None
        _sf_peaks = list()
        for i in range(n_experiments):
            _idxs = np.random.randint(sf_peaks.shape[0], size=n_samples)
            _sf_peaks.append(sf_peaks[_idxs])
        sf_peaks = np.array(_sf_peaks)
    
    return sf_peaks

def get_center_neuron_sf_tuning_curves(tuning_results_dir, model_name, layer_name, stim_type):
    """
    Gets tuning curve of center neuron. Returns a matrix of shape: (n_frequencies, n_channels)
    """
    
    assert layer_name in LAYERS[model_name]
    
    results_dir = os.path.join(tuning_results_dir, f"{stim_type}", f"{model_name}")
    results_fname = os.path.join(results_dir, f"{layer_name}.pkl")
    assert os.path.isfile(results_fname)
    tuning_curves = pickle.load(open(results_fname, "rb"), encoding="latin1")
    
    assert set(tuning_curves.keys()) == set(["spatial_frequencies", "orientations", "phases"])
    assert tuning_curves["spatial_frequencies"].shape[0] == PARAM_DICT["spatial_frequencies"].shape[0]
    
    # tuning_curves is of shape: (n_frequencies, C, H, W)
    # Extract central neuron location
    assert tuning_curves["spatial_frequencies"].ndim == 4
    mid_i = int(np.floor(tuning_curves["spatial_frequencies"].shape[2] / 2))
    mid_j = int(np.floor(tuning_curves["spatial_frequencies"].shape[3] / 2))
    
    _sf_tuning = tuning_curves["spatial_frequencies"][:,:,mid_i,mid_j]
    
    return _sf_tuning

#=================================================================
# Compare to V1 preferred spatial frequency data
#=================================================================

def get_v1_data(fov=6.4):
    # From De Valois et al., (1982).

    sf_dist = np.array([   0,   3,   3,   6,  17,  22,  18,  19,   9,   4])
    sf_bins = np.array([0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.2])
    
    return sf_dist, sf_bins * fov

def comparison_to_v1_metric(model_sfs, fov):
    # 1 if no experiments were done, 2 if experiments were done. If 2: shape is (n_experiments, n_samples)
    assert model_sfs.ndim in [1,2]
    
    # First, convert histograms to CDFs
    v1_counts, v1_bins = get_v1_data(fov=fov)
    v1_counts_cdf = v1_counts.cumsum() / v1_counts.sum()
    
    if model_sfs.ndim == 2:
        scores = list()
        for i in range(model_sfs.shape[0]):
            model_sf_hist = np.histogram(model_sfs[i,:], bins=v1_bins, density=False)[0]
            model_sf_cdf = model_sf_hist.cumsum() / model_sf_hist.sum()
            
            cdf_diffs = v1_counts_cdf - model_sf_cdf
            d = 1 - max(np.abs(cdf_diffs))
            scores.append(d)

        return scores
    else:
        model_sf_hist = np.histogram(model_sfs, bins=v1_bins, density=False)[0]
        model_sf_cdf = model_sf_hist.cumsum() / model_sf_hist.sum()

        cdf_diffs = v1_counts_cdf - model_sf_cdf
        score = 1 - max(np.abs(cdf_diffs))

        return score


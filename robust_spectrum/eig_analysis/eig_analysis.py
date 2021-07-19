import os
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from robust_spectrum.eig_analysis.eigenspectrum_utils import get_powerlaw, shuff_cvPCA
from robust_spectrum.core.feature_extractor import FeatureExtractor, get_layer_features

class Eigenspectrum():
    def __init__(self):
        self.data = None
        self.ss = None
        self.alpha = None
        self.ypred = None

    def compute_eigenspectrum(self):
        """
        Output:
            array of variance explained of size n_components
        """
        assert self.data is not None
        self.ss = self.pca_func(self.data)
        return self.ss

    def estimate_power_law_coefficient(self, fit_range=None):
        assert self.ss is not None
        assert fit_range is not None

        fit_range = np.arange(fit_range.min(), min(fit_range.max()+1, self.ss.size))

        # Fit power law coefficient
        self.alpha, self.ypred = get_powerlaw(self.ss/self.ss.sum(), fit_range.astype(int)) 
        return self.alpha, self.ypred

    def set_results(self, results):
        assert "ss" in results.keys()
        assert "ypred" in results.keys()
        assert "alpha" in results.keys()

        self.ss = results["ss"]
        self.alpha = results["alpha"]
        self.ypred = results["ypred"]

    def save_results(self, fname=None, results_dir=None):
        assert self.ss is not None
        assert self.alpha is not None
        assert self.ypred is not None

        if fname is None or results_dir is None:
            print("Did not save results. Filename is None.")
            return

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        fname = results_dir + f"/{fname}"

        results = dict()
        results["explained_variance_proportion"] = self.ss
        results["alpha"] = self.alpha
        results["y_pred"] = self.ypred
        pickle.dump(results, open(fname, "wb"))

    def pca_func(self, X):
        """
        Input:
            X : some data matrix (dimensions depends on pca_func implementation

        Output:
            array of variance explained of size n_components
        """
        raise NotImplementedError

class ArtificialNeuralResponseSpectrum(Eigenspectrum):
    def __init__(self, model, layer_name, dataloader, n_comp, n_batches=None):
        """
        Inputs:
            model      : (torch.nn.Module) PyTorch model.
            layer_name : (string) Layer from which to obtain features (activations).
            dataloader : (torch.utils.data.Dataloader) Dataloader for images.
            n_comp     : (int) Number of components for PCA.
            n_batches  : (int or None) Number of batches to obtain images from the 
                         dataloader. If the value is None, then use all the images.
        """
        super(Eigenspectrum ,self).__init__()
        fe = FeatureExtractor(
            dataloader,
            n_batches=n_batches,
            vectorize=True,
            debug=False
        )
        self.data = get_layer_features(fe, layer_name, model)
        self.n_components = n_comp

    def pca_func(self, X):
        """
        Input:
            X  : numpy array of dimensions (n_samples, n_features)

        Output:
            ss : numpy array of variance explained by each component
                 dimensions of (n_components,)
        """
        num_components = min(self.n_components, X.shape[0]-1, X.shape[1]-1)

        # Scale data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)

        pca = PCA(n_components=num_components, svd_solver="full", whiten=False).fit(X)
        return pca.explained_variance_

class BiologicalNeuralResponseSpectrum(Eigenspectrum):
    def __init__(self, neural_data, n_comp=2000, n_shuffles=5):
        """
        Inputs:
            neural_data : NeuralDataset object. Contains information about the neural
                          responses and stimulus set used.
            n_comp      : int. Number of components for cross-validated PCA.
            n_shuffles  : int. Number of shuffles for cross-validated PCA.
        """
        super(Eigenspectrum, self).__init__()
        self.data = neural_data
        self.n_shuffles = n_shuffles
        self.n_components = n_comp

    def pca_func(self, X):
        """
        Input:
            X  : numpy array of dimensions (2, n_stimuli, n_neurons)

        Output:
            ss : numpy array of variance explained by each component
                 dimensions of (n_components,)
        """
        assert X.ndim == 3
        assert X.shape[0] == 2
        ss = shuff_cvPCA(X, self.n_components, nshuff=self.n_shuffles)
        ss = ss.mean(axis=0)
        return ss



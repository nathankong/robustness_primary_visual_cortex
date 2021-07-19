import os
import pickle

from joblib import Parallel, delayed, dump, load

import numpy as np
import torchvision.transforms as transforms

from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from robust_spectrum.models.model_stats import MODEL_STATS
from robust_spectrum.core.dataloader_utils import get_imagenet_loaders
from robust_spectrum.core.feature_extractor import FeatureExtractor, get_layer_features
from robust_spectrum.neural_datasets.neural_dataset_utils import generate_train_test_split
from robust_spectrum.reliability.compute_reliability import compute_split_half_reliability

class NeuralPredictions():
    def __init__(
        self,
        model,
        model_arch,
        imagenet_dir,
        layer_name,
        neural_dataset,
        num_splits,
        results_dir,
        results_fname,
        memmap_dir,
        n_jobs
    ):
        """
        Inputs:
            model          : torch.nn.Module object. PyTorch model.
            model_arch     : string. model name.
            imagenet_dir   : string. location of imagenet images.
            layer_name     : string. Layer name of model to perform neural fits.
            neural_dataset : NeuralDataset object for the neural dataset used to
                             evaluate the model.
            num_splits     : int. Number of random splits for the linear mapping
                             procedure.
            results_dir    : string. Directory where fit results will be saved.
            results_fname  : string. Results file indicator.
            memmap_dir     : string. Directory to temporarily save model features
                             to help with parallelizing the neural fitting splits
            n_jobs         : int. Number of processes to create for parallelizing
                             the neural fitting splits.
        """
        print(f"Neural fits for {layer_name}.")
        self.layer_name = layer_name
        self.num_splits = num_splits

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        self.memmap_fname = memmap_dir + f"/{layer_name}"
        self.n_jobs = n_jobs

        self.results_fname = "{}/{}".format(results_dir, results_fname)

        self.model = model
        self.neural_dataset = neural_dataset
        self.stim_dataloader = neural_dataset.get_stim_dataloader()

        self.layer_pca = self._do_layer_pca(imagenet_dir, model_arch)

        self.feature_extractor = FeatureExtractor(
            self.stim_dataloader,
            vectorize=True,
            debug=False
        )

        self.neuron_ccs = self.compute_neuron_reliability()
        self.neuron_ids = None
        self.corrs = None

    def _do_layer_pca(self, imagenet_dir, model_arch):
        print(f"Performing layer PCA using ImageNet validation images...")

        imagenet_dataloader = _get_imagenet_val_dataloader(imagenet_dir, model_arch)
        imagenet_feature_extractor = FeatureExtractor(
            imagenet_dataloader,
            n_batches=6,
            vectorize=True,
            debug=False
        )
        features = get_layer_features(
            imagenet_feature_extractor,
            self.layer_name,
            self.model
        )

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(features)
        features = scaler.transform(features)

        n_components = 1000
        if features.shape[1] <= 1000:
            n_components = None

        layer_pca = PCA(n_components=n_components).fit(features)
        return layer_pca

    def save_results(self):
        assert self.neuron_ids is not None
        assert self.corrs is not None
        assert self.neuron_ccs is not None

        results = dict()
        results["correlations"] = self.corrs
        results["reliability"] = self.neuron_ccs
        results["select_neuron_ids"] = self.neuron_ids
        pickle.dump(results, open(self.results_fname, "wb"))

    def compute_neuron_reliability(self):
        print(f"Computing split-half reliability of neurons...")
        neural_data = self.neural_dataset.get_by_trial_neural_responses()
        neuron_ccs = compute_split_half_reliability(neural_data)
        return neuron_ccs

    def subsample_neural_data(self):
        """
        Since some data sets have a large number of neurons, we decide to
        subsample the neurons for the analysis. The decision to keep a
        neuron is based on its trial-trial correlation. This only affects
        data sets which *large* numbers of neurons.
        """
        trial_averaged_data = self.neural_dataset.get_trial_averaged_neural_responses()
        assert trial_averaged_data.ndim == 2

        if self.neuron_ccs.shape[0] > 5000:
            self.neuron_ids = (self.neuron_ccs >= np.percentile(self.neuron_ccs, 100-1))
        else:
            self.neuron_ids = (self.neuron_ccs >= np.percentile(self.neuron_ccs, 0))

        assert self.neuron_ids.size == trial_averaged_data.shape[1]
        neural_data_subset = trial_averaged_data[:,self.neuron_ids]

        print(f"Median correlation for 'good' neurons: {np.median(self.neuron_ccs[self.neuron_ids])}")
        print(f"Neural data subset dimensions: {neural_data_subset.shape}")

        return neural_data_subset

    def fit_responses(self, regr_func, **regr_kwargs):
        # Extract layer features
        features = get_layer_features(self.feature_extractor, self.layer_name, self.model)
        features = self.layer_pca.transform(features)

        dump(features, self.memmap_fname)
        features = load(self.memmap_fname, mmap_mode='r')

        print(f"Layer features dimensions: {features.shape}")

        neural_data_subset = self.subsample_neural_data()
        n_neurons = neural_data_subset.shape[1]

        corrs = Parallel(n_jobs=self.n_jobs)(delayed(_neural_fit_worker)(
            features, neural_data_subset, s, regr_func(**regr_kwargs)
        ) for s in range(self.num_splits))
        self.corrs = np.array(corrs)
        assert self.corrs.shape == (self.num_splits, 2, n_neurons)

        # Save fit results
        self.save_results()

def _neural_fit_worker(features, neural_data_subset, seed, reg):
    print(f"Seed {seed}")

    # Generate train and test split indices
    train_idx, test_idx = generate_train_test_split(features.shape[0], 0.75, seed=seed)

    # Do fit on all neurons
    n_neurons = neural_data_subset.shape[1]
    train_r, test_r = _fit_responses_routine(
        features,
        neural_data_subset,
        train_idx,
        test_idx,
        reg
    )
    assert train_r.size == test_r.size == n_neurons

    corrs = np.zeros((2, n_neurons))
    corrs[0,:] = train_r
    corrs[1,:] = test_r

    return corrs

def _per_neuron_correlations(pred, target):
    assert pred.shape == target.shape
    n_targets = pred.shape[1]
    corrs = np.zeros((n_targets,))
    for i in range(n_targets):
        corrs[i] = pearsonr(pred[:,i], target[:,i])[0]
    return corrs

def _fit_responses_routine(X, Y, train_idx, test_idx, reg):
    """
    Performs linear regression to predict Y from X using model defined by the
    regr_func class.

    Inputs:
        X             : numpy array of dimensions (n_samples, n_features)
        Y             : numpy array of dimensions (n_samples, n_targets)
        train_idx     : numpy array for indices for the train set
        test_idx      : numpy array for indices for the test set
        reg           : sklearn model class object (e.g. PLSRegression(n_components=2))

    Outputs:
        train_r : train set prediction correlation with actual values
        test_r  : test set prediction correlation with actual values
    """
    assert X.shape[0] == Y.shape[0]
    Y_train = Y[train_idx,:]
    Y_test = Y[test_idx,:]

    X_train = X[train_idx,:]
    X_test = X[test_idx,:]

    # Do fitting
    reg.fit(X_train, Y_train)

    # Do predictions
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    train_r = _per_neuron_correlations(pred_train, Y_train)
    test_r = _per_neuron_correlations(pred_test, Y_test)

    return train_r, test_r

def _get_imagenet_val_dataloader(imagenet_dir, model_arch):
    assert imagenet_dir is not None
    params = dict()
    params["image_dir"] = imagenet_dir
    params["batch_size"] = 256
    params["num_workers"] = 8

    if model_arch not in MODEL_STATS.keys():
        raise ValueError(f"Information not available for {model_arch}.")

    img_mean = MODEL_STATS[model_arch]["mean"]
    img_std = MODEL_STATS[model_arch]["std"]
    model_input_size = MODEL_STATS[model_arch]["input_size"]

    resize = 256
    crop = 224
    if model_input_size is not None:
        assert model_input_size[0] == model_input_size[1]
        if model_input_size[0] != 224: # e.g. inception_v3 is resize 299, crop 299
            resize = model_input_size[0]
            crop = resize

    my_transforms = {
        "train": None,
        "val": transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
    }

    _, loader = get_imagenet_loaders(params, my_transforms=my_transforms, shuffle=True)
    return loader


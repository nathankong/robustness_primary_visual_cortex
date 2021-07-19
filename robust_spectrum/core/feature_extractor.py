import numpy as np
import torch

class FeatureExtractor():
    """
    Extracts activations from a layer of a model.

    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
        debug      : (boolean) whether or not to test with two batches
    """
    def __init__(self, dataloader, n_batches=None, vectorize=False, debug=False):
        self.dataloader = dataloader
        if n_batches is None:
            self.n_batches = len(self.dataloader)
        else:
            self.n_batches = n_batches
        self.vectorize = vectorize
        self.debug = debug

    def _store_features(self, layer, inp, out):
        out = out.cpu().numpy()

        if self.vectorize:
            self.layer_feats.append(np.reshape(out, (len(out), -1)))
        else:
            self.layer_feats.append(out)

    def extract_features(self, model, model_layer):
        if torch.cuda.is_available():
            model.cuda().eval()
        else:
            model.cpu().eval()
        self.layer_feats = list()

        # Set up forward hook to extract features
        handle = model_layer.register_forward_hook(self._store_features)

        with torch.no_grad():
            for i, (x,_) in enumerate(self.dataloader):
                if i == self.n_batches:
                    break

                # Break when you go through 2 batches for faster testing
                if self.debug and i == 2:
                    break

                print(f"Step {i+1}/{self.n_batches}")
                if torch.cuda.is_available():
                    x = x.cuda()

                model(x)

        self.layer_feats = np.concatenate(self.layer_feats) 

        # Reset forward hook so next time function runs, previous hooks
        # are removed
        handle.remove()

        return self.layer_feats

def get_layer_features(feature_extractor, layer_name, model):
    """
    Helper function to extract stimuli features from a layer in a model.

    Inputs:
        feature_extractor : object used to extract features from a layer
        layer_name        : name of layer from which to extract features
        model             : torch.nn.Module object

    Output:
        features          : numpy array of dimensions (num_images, num_features)
    """
    if isinstance(model, torch.nn.DataParallel):
        layer_module = model.module
    else:
        layer_module = model

    for part in layer_name.split('.'):
        layer_module = layer_module._modules.get(part)
        assert layer_module is not None, \
                f"No submodule found for layer {layer_name}, at part {part}."

    features = feature_extractor.extract_features(model, layer_module)
    return features

if __name__ == "__main__":
    from robust_spectrum.core.dataloader_utils import get_image_array_dataloader
    from robust_spectrum.models.model_layers import LAYERS
    from robust_spectrum.core.model_loader_utils import load_model

    model, layers = load_model("mnasnet1_0", trained=False, model_path=None)

    N = 10
    array = torch.rand(N, 224, 224, 3).numpy()
    dataloader = get_image_array_dataloader(array, torch.ones(N))

    layer_name = "layers.16"
    fe = FeatureExtractor(dataloader, n_batches=3, vectorize=False, debug=True)
    features = get_layer_features(fe, layer_name, model)
    print(features.shape)



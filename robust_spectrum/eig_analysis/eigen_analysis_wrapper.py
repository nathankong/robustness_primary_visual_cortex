from torchvision import transforms

from robust_spectrum.models.model_stats import MODEL_STATS
from robust_spectrum.models.model_paths import MODEL_PATHS
from robust_spectrum.core.dataloader_utils import get_imagenet_loaders
from robust_spectrum.neural_datasets.neural_datasets import get_neural_dataset
from robust_spectrum.core.model_loader_utils import load_model
from robust_spectrum.eig_analysis.eig_analysis import \
    BiologicalNeuralResponseSpectrum, \
    ArtificialNeuralResponseSpectrum

def _get_imagenet_dataloader(imagenet_dir, model_arch):
    assert imagenet_dir is not None
    params = dict()
    params["image_dir"] = imagenet_dir
    params["batch_size"] = 256
    params["num_workers"] = 8

    if model_arch not in MODEL_STATS.keys():
        raise ValueError(f"Information not available for {model_arch}.")

    img_mean = MODEL_STATS[model_arch]["mean"]
    img_std = MODEL_STATS[model_arch]["std"]

    assert "input_size" in MODEL_STATS[model_arch].keys()
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

def main(args):
    results_dir = args.results_dir

    if args.neural_dataset is not None:
        neural_dataset = get_neural_dataset(args.neural_dataset.lower(), img_mean=[0,0,0], img_std=[1,1,1])
        print("Using neural response stimuli.")
    else:
        assert args.neural_dataset is None
        assert args.model_arch is not None
        assert args.stim_dataset.lower() == "imagenet"

    # If no model architecture is provided, it means that we want to compute
    # the eigenspectrum of the neural responses.
    if args.model_arch is None:
        neural_data = neural_dataset.get_by_trial_neural_responses()
        fit_range = neural_dataset.get_spectrum_fit_range()
        n_comp = neural_dataset.get_spectrum_n_comp()

        print("Computing biological neural response eigenspectrum.")

        # Neural responses eigenspectrum
        spectrum = BiologicalNeuralResponseSpectrum(
            neural_data,
            n_comp=n_comp,
            n_shuffles=20
        )

        _ = spectrum.compute_eigenspectrum()
        _, _ = spectrum.estimate_power_law_coefficient(fit_range=fit_range)
        spectrum.save_results(f"/{args.neural_dataset}.pkl", results_dir)

    # Compute the eigenspectrum of the model layers
    else:
        trained = args.trained

        # Eigenspectrum fit range for neural network features
        fit_range = np.arange(9,999)

        print("Computing artificial neural response eigenspectrum.")

        # Grab custom model path if applicable
        model_path = None
        if args.model_arch in MODEL_PATHS.keys():
            model_path = MODEL_PATHS[args.model_arch]

        # Load model
        model, layers = load_model(
            args.model_arch,
            trained=trained,
            model_path=model_path
        )

        for layer_name in layers:
            print(f"Layer {layer_name}")
            results = {"alpha": [], "ypred": [], "ss": []}
            for i in range(args.num_subsets):
                print(f"Subset {i+1}/{args.num_subsets}")
                assert args.neural_dataset is None

                # Get new stimulus dataloader every subset
                print("Using ImageNet stimuli.")
                stim_dataloader = _get_imagenet_dataloader(args.imagenet_dir, args.model_arch)

                # Deep net eigenspectrum
                spectrum = ArtificialNeuralResponseSpectrum(
                    model,
                    layer_name,
                    stim_dataloader,
                    2000,
                    n_batches=11
                )

                ss = spectrum.compute_eigenspectrum()
                alpha, ypred = spectrum.estimate_power_law_coefficient(fit_range=fit_range)
                results["alpha"].append(alpha)
                results["ypred"].append(ypred)
                results["ss"].append(ss)

            spectrum.set_results(results)
            spectrum.save_results(f"/{layer_name}.pkl", results_dir + f"/{args.model_arch}/")

if __name__ == "__main__":
    import argparse
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", type=str, default=None)
    parser.add_argument("--neural-dataset", type=str, default=None, choices=["cadena_all", "cadena_natural", "stringer"])
    parser.add_argument("--stim-dataset", type=str, default=None, choices=["imagenet"])
    parser.add_argument("--imagenet-dir", type=str, default=None)
    parser.add_argument("--num-subsets", type=int, default=3)
    parser.add_argument("--results-dir", type=str, default="./")
    parser.add_argument('--trained', dest="trained", action="store_true")
    parser.set_defaults(trained=False)
    args = parser.parse_args()

    main(args)



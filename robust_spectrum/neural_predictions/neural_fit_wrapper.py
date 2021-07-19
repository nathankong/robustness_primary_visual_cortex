from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV

from robust_spectrum.core.model_loader_utils import load_model
from robust_spectrum.neural_datasets.neural_datasets import get_neural_dataset
from robust_spectrum.neural_predictions.neural_predictions import NeuralPredictions
from robust_spectrum.models.model_stats import MODEL_STATS
from robust_spectrum.models.model_paths import MODEL_PATHS
from robust_spectrum.models.model_layers import LAYERS

def main(args):
    assert args.model_arch in MODEL_STATS.keys()
    assert args.model_arch in LAYERS.keys()

    # Grab custom model path if applicable
    model_path = None
    if args.model_arch in MODEL_PATHS.keys():
        model_path = MODEL_PATHS[args.model_arch]

    # Load model
    model, layers = load_model(args.model_arch, trained=args.trained, model_path=model_path)

    # Get neural response dataset
    assert "input_size" in MODEL_STATS[args.model_arch].keys()
    model_input_size = MODEL_STATS[args.model_arch]["input_size"]
    neural_dataset = get_neural_dataset(
        args.neural_dataset,
        img_mean=MODEL_STATS[args.model_arch]["mean"],
        img_std=MODEL_STATS[args.model_arch]["std"],
        model_input_size=model_input_size
    )

    # Set up results directory
    results_dir = args.results_dir + f"/{args.neural_dataset + '_' + args.map_type}/{args.model_arch}/"

    import time
    start = time.time()
    for layer_name in layers:
        results_fname = layer_name + ".pkl"
        memmap_dir = args.memmap_dir + f"/{args.neural_dataset + '_' + args.map_type}/{args.model_arch}/"

        neural_pred = NeuralPredictions(
            model,
            args.model_arch,
            args.imagenet_dir,
            layer_name,
            neural_dataset,
            int(args.num_splits),
            results_dir,
            results_fname,
            memmap_dir,
            args.njobs
        )

        # Determine regression function and do fits
        if args.map_type == "pls":
            regr_func = PLSRegression
            regr_kwargs = {"n_components": 25, "scale": False}
        elif args.map_type == "ridge":
            regr_func = RidgeCV
            regr_kwargs = {"alphas":(0.01,0.1,1.0,10.0), "cv":5}
        else:
            assert 0, f"{args.map_type} is not supported yet."
        neural_pred.fit_responses(regr_func, **regr_kwargs)

    print("Time elapsed (sec):", time.time() - start)

if __name__ == "__main__":
    import argparse
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-dataset", type=str, default="stringer")
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="./")
    parser.add_argument("--imagenet-dir", type=str, default="")
    parser.add_argument("--model-arch", type=str, default="resnet18")
    parser.add_argument("--map-type", type=str, default="pls")
    parser.add_argument("--memmap-dir", type=str, default="./")
    parser.add_argument("--njobs", type=int, default=10)
    parser.add_argument('--trained', dest="trained", action="store_true")
    parser.set_defaults(trained=False)
    args = parser.parse_args()

    from robust_spectrum.models.model_layers import LAYERS
    print(f"Fitting for {args.model_arch}.")
    if args.model_arch not in LAYERS.keys():
        raise ValueError(f"Layer information not available for {args.model_arch}.")

    if LAYERS[args.model_arch] != []:
        main(args)



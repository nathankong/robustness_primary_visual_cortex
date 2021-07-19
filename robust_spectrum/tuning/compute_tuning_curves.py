import os
import pickle

import numpy as np

from robust_spectrum.models.model_paths import MODEL_PATHS
from robust_spectrum.core.model_loader_utils import load_model
from robust_spectrum.tuning.constants import PARAM_DICT
from robust_spectrum.tuning.tuning_curves import TuningCurves

def main(args):

    # Model / Architecture
    trained = args.trained
    model_name = args.model_arch

    # Grab custom model path if applicable
    model_path = None
    if model_name in MODEL_PATHS.keys():
        model_path = MODEL_PATHS[model_name]

    model, layers = load_model(model_name, trained=trained, model_path=model_path)

    # Tuning curves
    tc = TuningCurves(PARAM_DICT, model_name, model, stim_type=args.stim_type)

    # Compute tuning curves for each layer
    for layer in layers:

        # Results save directory
        results_dir = os.path.join(args.results_dir, f"{args.stim_type}", f"{model_name}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print(f"Computing tuning curves for {layer}...")
        tuning_curves = tc.compute_tuning_curves(layer)

        # Save results
        results_fname = os.path.join(results_dir, f"{layer}.pkl")
        pickle.dump(tuning_curves, open(results_fname, "wb"))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", type=str, default="resnet18")
    parser.add_argument("--trained", dest="trained", action="store_true")
    parser.add_argument("--stim-type", type=str, default="gabor")
    parser.add_argument("--results-dir", type=str, default="./")
    parser.set_defaults(trained=False)
    args = parser.parse_args()

    main(args)



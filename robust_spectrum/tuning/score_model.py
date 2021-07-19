import numpy as np

from scoring_utils import get_preferred_sf_distribution, comparison_to_v1_metric

def main(args):
    arch = args.model_arch
    neural_results_dir = args.neural_fit_results_dir
    tuning_curves_dir = args.tuning_curves_dir
    n_samples = args.num_samples
    n_experiments = args.num_experiments

    # First obtain n_experiments to obtain preferred spatial frequencies
    preferred_sf_distribution = get_preferred_sf_distribution(
        arch,
        "gabor",
        neural_results_dir,
        tuning_curves_dir,
        n_samples=n_samples,
        n_experiments=n_experiments
    )

    # Score the model with respect to macaque V1 foveal data
    score = comparison_to_v1_metric(preferred_sf_distribution, args.fov)
    print(f"{arch}: {np.mean(score)} +/- {np.std(score)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", type=str, default="resnet18")
    parser.add_argument("--neural-fit-results-dir", type=str, default="./")
    parser.add_argument("--tuning-curves-dir", type=str, default="./")
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--fov", type=float, default=6.4)
    args = parser.parse_args()

    main(args)


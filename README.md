## Robustness and Macaque Primary Visual Cortex

Our preprint is located here: https://www.biorxiv.org/content/10.1101/2021.06.29.450334v1

To install the repository, follow these steps:
1. Create a virtual environment for this project (for example): `python3.6 -m venv robust_v1_env`
2. Activate the environment: `source robust_v1_env/bin/activate`
3. Clone the repository: `git clone https://github.com/nathankong/robustness_primary_visual_cortex.git`
4. Change your working directory: `cd robustness_primary_visual_cortex`
5. Install the required packages: `pip install -e .`
6. Navigate to `robust_spectrum/core/` and modify the directories in `default_dirs.py` to the location 
of your data.
7. Navigate to `robust_spectrum/configs/` and modify the directories found in each configuration file 
if training models is desired (read the description at the top of the configuration file first).

### Neural Predictions on Macaque V1 data

1. `cd robust_spectrum/neural_predictions`
2. Run (for example): `CUDA_VISIBLE_DEVICES=0 python neural_fit_wrapper.py --neural-dataset cadena_all 
--num-splits 20 --results-dir ./ --imagenet-dir /PATH/TO/IMAGENET/VAL/ --model-arch alexnet --map-type pls 
--memmap-dir ./ --njobs 5 --trained`

The above command would use GPU 0 to extract features from a trained AlexNet.  Twenty train-test splits would 
be constructed and PLS regression would be the linear map used. The results would be saved into 
`robust_spectrum/neural_predictions` and five jobs would be used to parallelize the neural fits across 
train-test splits. `--memmap-dir` is only to help with parallelization (but remember to delete files located
there because a lot of memory could be used!).  The path to the ImageNet validation set images is needed for
performing PCA on the model features.

### Computing Power Law Exponents of Biological Neural Responses

1. `cd robust_spectrum/eig_analysis`
2. Run this command to compute the power law exponent of the macaque V1 responses to natural scenes: 
`python eigen_analysis_wrapper.py --neural-dataset cadena_natural --results-dir ./`
3. Run this command to compute the power law exponent of the mouse V1 responses to natural scenes: 
`python eigen_analysis_wrapper.py --neural-dataset stringer --results-dir ./`

The commands in 2. and 3. compute the power law exponents to the macaque V1 neural responses and mouse V1
responses to natural scenes.  The results are saved in `robust_spectrum/eig_analysis`.

### Computing Power Law Exponents of Artificial Neural Responses

1. `cd robust_spectrum/eig_analysis`
2. Run (for example): `CUDA_VISIBLE_DEVICES=0 python eigen_analysis_wrapper.py --model-arch alexnet 
--stim-dataset imagenet --imagenet-dir /PATH/TO/IMAGENET/VAL/ --num-subsets 3 --results-dir ./ --trained`

The above command would use GPU 0 to extract features from a trained AlexNet in response to a set of ImageNet 
images. Three random subsets from the ImageNet validation set would be used to obtain three different power 
law exponents for each layer of AlexNet. The results would be saved into `robust_spectrum/eig_analysis`.
You can change the `--model-arch` to whatever desired model you want.

### Computing Tuning Curves and Scoring Each Model With Respect to Cells in the Foveal Area of Macaque V1

1. `cd robust_spectrum/tuning`
2. To compute tuning curves, run (for example): `CUDA_VISIBLE_DEVICES=0 python compute_tuning_curves.py 
--model-arch alexnet --stim-type gabor --results-dir ./ --trained`
3. To score the model, run the command (this assumes that neural fits have been performed for the model already):
`python score_model.py --model-arch alexnet --neural-fit-results-dir /PATH/TO/NEURAL/FIT/RESULTS/ --tuning-curves-dir ./
--num-experiments 1000 --num-samples 150 --fov 6.4`

The command in 2. computes the tuning curves for each artificial neuron in each layer of a trained AlexNet
using GPU 0 and saves the tuning curves in `robust_spectrum/tuning`. The command in 3. computes how well AlexNet's
preferred spatial frequency distribution matches that of cells in the foveal area of macaque V1 (using data from
De Valois et al., 1982). Assuming that you used the command in the previous sections to perform neural predictions,
`/PATH/TO/NEURAL/FIT/RESULTS/` would be replaced with `../neural_predictions/cadena_all_pls/`. 1000 in-silico
experiments would be performed on the model and in each experiment, a random sample of 150 artificial neurons (where
each is one of the "center neurons" in the activations matrix) is obtained to create the model layer's distribution
of preferred spatial frequencies.

### Computing the Adversarial Robustness of Models

1. `cd robust_spectrum/robustness_eval`
2. Run the command: `CUDA_VISIBLE_DEVICES=0 python imagenet_robustness.py --type Linf --model-arch alexnet
--imagenet-dir /PATH/TO/IMAGENET/VAL/ --trained --results-dir ./`

This would compute the adversarial robustness of a trained AlexNet using all 50000 ImageNet validation set
images against L-inf perturbations of norms: `[0., 1./1020, 1./255, 4./255]`. If `--type` is set to `L2`,
then the perturbations would be of norms: `[0., 0.15, 0.6, 2.4]`. If `--type` is set to `L1`, then the 
perturbations would be of norms: `[0., 40., 160., 640.]`.

### Training ResNet-50 on TRADES or Input Gradient Regularization (IGR)

1. Look at the configuration files in `robust_spectrum/configs/` and make sure the values in the config
dictionary are desired (e.g., where to save model checkpoints, the location of ImageNet, etc.)
2. `cd robust_spectrum`
3. To train the model on one GPU to minimize the TRADES loss, run: `CUDA_VISIBLE_DEVICES=0 python train.py 
--config configs/config_imagenet_trades.py`
4. To train the model on one GPU to minimize the IGR loss, run: `CUDA_VISIBLE_DEVICES=0 python train.py 
--config configs/config_imagenet_igr.py`

Note that you may need to use more GPUs, so you would set `CUDA_VISIBLE_DEVICES` as desired.



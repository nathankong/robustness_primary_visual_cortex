import os

from robust_spectrum.core.default_dirs import MODEL_SAVE_DIR
from robust_spectrum.models.model_layers import LAYERS
from robust_spectrum.models.model_stats import MODEL_STATS

MODEL_PATHS = {

    "robust_resnet50_linf_2":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_resnet50_linf_2.pt"),

    "robust_resnet50_linf_4":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_resnet50_linf_4.pt"),

    "robust_resnet50_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_resnet50_l2_3.pt"),

    "robust_wide_resnet50_2_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_wide_resnet50_2_l2_3.pt"),

    "robust_wide_resnet50_2_linf_4":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_wide_resnet50_2_linf_4.pt"),

    "robust_resnet18_linf_1":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_resnet18_linf_1.pt"),

    "robust_densenet161_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_densenet161_l2_3.pt"),

    "robust_resnet18_linf_0_5":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_resnet18_linf_0_5.pt"),

    "robust_shufflenet_v2_x1_0_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_shufflenet_v2_x1_0_l2_3.pt"),

    "robust_mnasnet1_0_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_mnasnet1_0_l2_3.pt"),

    "robust_mobilenet_v2_l2_3":
        os.path.join(MODEL_SAVE_DIR, "pgd_robust_mobilenet_v2_l2_3.pt"),

    "resnet50_simclr":
        os.path.join(MODEL_SAVE_DIR, "resnet50_simclr.pt"),

    "free_robust_resnet50_linf_4":
        os.path.join(MODEL_SAVE_DIR, "free_robust_resnet50_linf_4.pt"),

    "fast_robust_resnet50_linf_4":
        os.path.join(MODEL_SAVE_DIR, "fast_robust_resnet50_linf_4.pt"),

    "free_trades_robust_resnet50_linf_4":
        os.path.join(MODEL_SAVE_DIR, "free_trades_robust_resnet50_linf_4.pt"),

    "trades_robust_resnet50_linf_4":
        os.path.join(MODEL_SAVE_DIR, "trades_robust_resnet50_imagenet_linf_4.pt"),

    "igr_robust_resnet50":
        os.path.join(MODEL_SAVE_DIR, "igr_robust_resnet50.pt"),

}

for model in MODEL_PATHS.keys():
    assert model in LAYERS.keys(), f"{model} not in model_layers.py"
    assert model in MODEL_STATS.keys(), f"{model} not in model_stats.py"
    assert os.path.isfile(MODEL_PATHS[model])



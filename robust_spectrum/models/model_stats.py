"""
Contains information on image mean and standard deviation for RGB channels
that models were trained on. Will also contain necessary padding for doing 
neural prediction. For example, inception_v3 needs to be padded up to 
299x299 input.

The format is as follows: each model name is associated with a dictionary
that has "mean", "std", "input_size" as keys.  "input_size" is the image size
the model was trained on.  Padding comes into play when doing neural fits 
(e.g. a stimulus size of 40x40 may need to be padded up to 224x224).

Robust models should always contain the "robust" as a substring. For example,
you can have pgd_robust_resnet50 or robust_resnet50.
"""

MODEL_STATS = {

    "alexnet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "vgg11": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "vgg13": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "vgg16": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "vgg19": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet18": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet34": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet50": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet101": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet152": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "wide_resnet50_2": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "wide_resnet101_2": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_resnet18_linf_0_5": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_resnet18_linf_1": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "resnet50_simclr": { # SIMCLR has no normalization
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "input_size": (224,224)
    },

    "robust_resnet50_linf_2": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_resnet50_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "fast_robust_resnet50_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "free_robust_resnet50_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "trades_robust_resnet50_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "igr_robust_resnet50": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "free_trades_robust_resnet50_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_resnet50_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_wide_resnet50_2_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_wide_resnet50_2_linf_4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "squeezenet1_0": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "squeezenet1_1": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "densenet121": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "densenet161": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "densenet169": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "densenet201": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_densenet161_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "googlenet": { # Inception v1
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "inception_v3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (299,299)
    },

    "shufflenet_v2_x0_5": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "shufflenet_v2_x1_0": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_shufflenet_v2_x1_0_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "mobilenet_v2": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_mobilenet_v2_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "mnasnet0_5": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "mnasnet1_0": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "robust_mnasnet1_0_l2_3": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (224,224)
    },

    "xception": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "input_size": (299,299)
    },

    "nasnetamobile": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "input_size": (224,224)
    },

}



"""
PyTorch modules for each model architecture. These are the layers that we will
extract activations from for neural prediction and for eigenspectrum analysis.

Robust models should always contain the "robust" as a substring. For example,
you can have pgd_robust_resnet50 or robust_resnet50.
"""

LAYERS = {
    # AlexNet
    "alexnet":
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"] +
        ["features.12"] +
        ["classifier.2"] +
        ["classifier.5"],

    # VGGs
    "vgg11":
        ["features.2"] +
        ["features.5"] +
        ["features.10"] +
        ["features.15"] +
        ["features.20"] +
        ["classifier.1"] +
        ["classifier.4"],

    "vgg13":
        ["features.4"] +
        ["features.9"] +
        ["features.14"] +
        ["features.19"] +
        ["features.24"] +
        ["classifier.1"] +
        ["classifier.4"],
        
    "vgg16":
        ["features.4"] +
        ["features.9"] +
        ["features.16"] +
        ["features.23"] +
        ["features.30"] +
        ["classifier.1"] +
        ["classifier.4"],

    "vgg19":
        ["features.4"] +
        ["features.9"] +
        ["features.18"] +
        ["features.27"] +
        ["features.36"] +
        ["classifier.1"] +
        ["classifier.4"],

    # ResNets
    "resnet18":
        ["relu", "maxpool"] +
        ["layer1.0", "layer1.1"] +
        ["layer2.0", "layer2.1"] +
        ["layer3.0", "layer3.1"] +
        ["layer4.0", "layer4.1"] +
        ["avgpool"],

    "resnet34":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet50":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet101":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(23)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet152":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(8)] +
        [f"layer3.{i}" for i in range(36)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "wide_resnet50_2":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "wide_resnet101_2":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(23)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    # Robust ResNet18s
    "robust_resnet18_linf_0_5":
        ["relu", "maxpool"] +
        ["layer1.0", "layer1.1"] +
        ["layer2.0", "layer2.1"] +
        ["layer3.0", "layer3.1"] +
        ["layer4.0", "layer4.1"] +
        ["avgpool"],

    "robust_resnet18_linf_1":
        ["relu", "maxpool"] +
        ["layer1.0", "layer1.1"] +
        ["layer2.0", "layer2.1"] +
        ["layer3.0", "layer3.1"] +
        ["layer4.0", "layer4.1"] +
        ["avgpool"],

    # Unsupervised ResNet50s
    "resnet50_simclr":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    # Robust ResNet50s
    "robust_resnet50_linf_2":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "robust_resnet50_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "fast_robust_resnet50_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "free_robust_resnet50_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "trades_robust_resnet50_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "igr_robust_resnet50":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "free_trades_robust_resnet50_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "robust_resnet50_l2_3":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "robust_wide_resnet50_2_l2_3":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "robust_wide_resnet50_2_linf_4":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    # SqueezeNets
    "squeezenet1_0": [
        "features." + layer for layer in
        ["2"] + [f"{i}.expand3x3_activation" for i in [3, 4, 5, 7, 8, 9, 10, 12]]
    ],
    "squeezenet1_1": [
        "features." + layer for layer in
        ["2"] + [f"{i}.expand3x3_activation" for i in [3, 4, 6, 7, 9, 10, 11, 12]]
    ],

    # DenseNets
    "densenet121":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(24)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(16)],

    "densenet161":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(36)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(24)],

    "densenet169":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(32)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(32)],

    "densenet201":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(48)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(32)],

    # Robust DenseNets
    "robust_densenet161_l2_3":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(36)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(24)],

    # Inceptions
    "googlenet": # Inception v1
        ["maxpool1", "maxpool2"] +
        [f"inception3{i}" for i in ['a', 'b']] +
        ["maxpool3"] +
        [f"inception4{i}" for i in ['a', 'b', 'c', 'd', 'e']] +
        ["maxpool4"] +
        [f"inception5{i}" for i in ['a', 'b']] +
        ["avgpool"],

    "inception_v3":
        #["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"] +
        ["maxpool1", "maxpool2"] +
        [f"Mixed_5{i}" for i in ['b', 'c', 'd']] +
        [f"Mixed_6{i}" for i in ['a', 'b', 'c', 'd', 'e']] +
        [f"Mixed_7{i}" for i in ['a', 'b', 'c']],

    # ShuffleNet
    "shufflenet_v2_x0_5":
        ["maxpool"] +
        [f"stage2.{i}" for i in range(4)] +
        [f"stage3.{i}" for i in range(8)] +
        [f"stage4.{i}" for i in range(4)] +
        ["conv5.2"],

    "shufflenet_v2_x1_0":
        ["maxpool"] +
        [f"stage2.{i}" for i in range(4)] +
        [f"stage3.{i}" for i in range(8)] +
        [f"stage4.{i}" for i in range(4)] +
        ["conv5.2"],

    "robust_shufflenet_v2_x1_0_l2_3":
        ["maxpool"] +
        [f"stage2.{i}" for i in range(4)] +
        [f"stage3.{i}" for i in range(8)] +
        [f"stage4.{i}" for i in range(4)] +
        ["conv5.2"],

    # MobileNet
    "mobilenet_v2":
        ["features.0.2"] +
        ["features.1.conv.1"] +
        [f"features.{i}.conv.2" for i in range(2,18)] + # output conv of each inverted res block
        ["features.18.2"],

    "robust_mobilenet_v2_l2_3":
        ["features.0.2"] +
        ["features.1.conv.1"] +
        [f"features.{i}.conv.2" for i in range(2,18)] + # output conv of each inverted res block
        ["features.18.2"],

    # MNASNets
    "mnasnet0_5":
        ["layers.2", "layers.5"] +
        [f"layers.8.{i}.layers.6" for i in range(3)] +
        [f"layers.9.{i}.layers.6" for i in range(3)] +
        [f"layers.10.{i}.layers.6" for i in range(3)] +
        [f"layers.11.{i}.layers.6" for i in range(2)] +
        [f"layers.12.{i}.layers.6" for i in range(4)] +
        [f"layers.13.{i}.layers.6" for i in range(1)] +
        ["layers.16"],

    "mnasnet1_0":
        ["layers.2", "layers.5"] +
        [f"layers.8.{i}.layers.6" for i in range(3)] +
        [f"layers.9.{i}.layers.6" for i in range(3)] +
        [f"layers.10.{i}.layers.6" for i in range(3)] +
        [f"layers.11.{i}.layers.6" for i in range(2)] +
        [f"layers.12.{i}.layers.6" for i in range(4)] +
        [f"layers.13.{i}.layers.6" for i in range(1)] +
        ["layers.16"],

    "robust_mnasnet1_0_l2_3":
        ["layers.2", "layers.5"] +
        [f"layers.8.{i}.layers.6" for i in range(3)] +
        [f"layers.9.{i}.layers.6" for i in range(3)] +
        [f"layers.10.{i}.layers.6" for i in range(3)] +
        [f"layers.11.{i}.layers.6" for i in range(2)] +
        [f"layers.12.{i}.layers.6" for i in range(4)] +
        [f"layers.13.{i}.layers.6" for i in range(1)] +
        ["layers.16"],

    "xception":
        [],

    "nasnetamobile":
        [],

}


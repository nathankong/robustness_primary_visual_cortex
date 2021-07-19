import os
import numpy as np

import torch
from torchvision import transforms, datasets

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD, L1PGD

from robust_spectrum.core.train_utils import AverageMeter
from robust_spectrum.core.dataloader_utils import get_imagenet_loaders
from robust_spectrum.core.model_loader_utils import load_model
from robust_spectrum.models.model_stats import MODEL_STATS
from robust_spectrum.models.model_paths import MODEL_PATHS

def _get_imagenet_dataloader(imagenet_dir, model_input_size=None):
    params = dict()
    params["image_dir"] = imagenet_dir
    params["batch_size"] = 256
    params["num_workers"] = 8

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
        ])
    }

    _, loader = get_imagenet_loaders(params, my_transforms=my_transforms, shuffle=False)
    return loader

if __name__ == "__main__":
    import pickle
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="Linf")
    parser.add_argument('--model-arch', type=str, default="")
    parser.add_argument('--imagenet-dir', type=str, default="")
    parser.add_argument('--trained', dest="trained", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.set_defaults(trained=False)
    args = parser.parse_args()

    # Grab custom model path if applicable
    model_path = None
    if args.model_arch in MODEL_PATHS.keys():
        model_path = MODEL_PATHS[args.model_arch]

    # Instantiate a model
    model, _ = load_model(
        args.model_arch,
        trained=args.trained,
        model_path=model_path
    )

    if args.model_arch not in MODEL_STATS.keys():
        raise ValueError(f"Information not available for {args.model_arch}.")

    # Model meta data
    img_mean = MODEL_STATS[args.model_arch]["mean"]
    img_std = MODEL_STATS[args.model_arch]["std"]

    assert "input_size" in MODEL_STATS[args.model_arch].keys()
    model_input_size = MODEL_STATS[args.model_arch]["input_size"]

    preprocessing = dict(mean=img_mean, std=img_std, axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    val_loader = _get_imagenet_dataloader(args.imagenet_dir, model_input_size=model_input_size)

    if args.type == "Linf":
        epsilons = [0., 1./1020, 1./255, 4./255]
        attack_type = LinfPGD
    elif args.type == "L2":
        epsilons = [0., 0.15, 0.6, 2.4]
        attack_type = L2PGD
    elif args.type == "L1":
        epsilons = [0., 40., 160., 640.]
        attack_type = L1PGD
    else:
        assert 0

    rob_meters = dict()
    for eps in epsilons:
        rob_meters[eps] = AverageMeter(eps)

    avg_meter = AverageMeter("val")
    for i, (images, labels) in enumerate(val_loader):
        print(f"Iteration {i+1}/{len(val_loader)}")

        N = images.shape[0]
        images, labels = images.cuda(), labels.cuda()
        images, labels = ep.astensor(images), ep.astensor(labels)

        for eps in epsilons:
            if eps == 0:
                robust_accuracy = [accuracy(fmodel, images, labels)]
            else:
                attack = attack_type(abs_stepsize=eps*2.0/20, steps=20, random_start=False)
                advs, _, success = attack(fmodel, images, labels, epsilons=[eps])
                robust_accuracy = 1 - ep.astensor(success).float32().mean(axis=-1)

            for e, rob_acc in zip([eps], robust_accuracy):
                # Update robust acc for each eps meter
                if e == 0:
                    rob_meters[e].update(rob_acc, N)
                else:
                    rob_meters[e].update(rob_acc.item(), N)

    # Print results
    results_dict = dict()
    for eps in epsilons:
        print(f"eps: {eps} || rob acc: {rob_meters[eps].avg}")
        results_dict[eps] = rob_meters[eps].avg

    # Save results
    if args.results_dir is not None:
        save_dir = args.results_dir + f"/{args.model_arch}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = save_dir + f"/{args.type}.pkl"
        pickle.dump(results_dict, open(fname, "wb"))




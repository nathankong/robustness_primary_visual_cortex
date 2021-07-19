import os
import sys
import pickle

import torch
import torch.nn as nn

import numpy as np

from collections import defaultdict

from core.dataloader_utils import get_dataloaders
from core.train_utils import \
    AverageMeter, \
    save_checkpoint, \
    load_checkpoint, \
    check_best_accuracy, \
    compute_accuracy, \
    seed_torch

def train_step(model, train_loader, optimizer, loss_func, epoch, results):
    losses = AverageMeter("Loss", ':.4e')
    top1 = AverageMeter("Acc@1", ':6.2f')
    top5 = AverageMeter("Acc@5", ':6.2f')
    num_steps = len(train_loader)

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        loss, predictions = loss_func(model, data, labels, epoch=epoch)

        # Backward propagation
        loss.backward()
        optimizer.step()

        # Metrics
        acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        print(
            "[Epoch {}; Step {}/{}] Train Loss: {:.6f}; Train Accuracy: {:.6f}"\
                .format(epoch+1, i+1, num_steps, loss, acc1)
        )
        sys.stdout.flush()

        # Save per iteration train losses/accuracies
        results["train"]["iter_losses"].append(loss.item())
        results["train"]["iter_top1_accs"].append(acc1.item())
        results["train"]["iter_top5_accs"].append(acc5.item())

    results["train"]["losses"].append(losses.avg)
    results["train"]["top1_accs"].append(top1.avg)
    results["train"]["top5_accs"].append(top5.avg)

    # Save losses and accuracies every epoch so we can plot loss and accuracy
    pickle.dump(results, open(results["fname"], "wb"))

    return losses.avg, top1.avg, top5.avg

def test_step(model, test_loader, loss_func, epoch, results):
    losses = AverageMeter("Loss", ':.4e')
    top1 = AverageMeter("Acc@1", ':6.2f')
    top5 = AverageMeter("Acc@5", ':6.2f')
    num_steps = len(test_loader)

    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.cuda(), labels.cuda()

            # Compute loss and predictions
            loss, predictions = loss_func(model, data, labels, epoch=epoch)

            # Metrics
            acc1, acc5 = compute_accuracy(predictions, labels, topk=(1, 5))

            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

            print(
                "[Epoch {}; Step {}/{}] Val Loss: {:.6f}; Val Accuracy: {:.6f}"\
                    .format(epoch+1, i+1, num_steps, loss, acc1)
            )
            sys.stdout.flush()

    results["val"]["losses"].append(losses.avg)
    results["val"]["top1_accs"].append(top1.avg)
    results["val"]["top5_accs"].append(top5.avg)

    # Save losses and accuracies every epoch so we can plot loss and accuracy
    pickle.dump(results, open(results["fname"], "wb"))

    return losses.avg, top1.avg, top5.avg

def train(train_params, save_dir):
    # Load model
    m = train_params["model"]
    m = nn.DataParallel(m).cuda()

    if train_params["pretrained"]:
        assert train_params["pretrained_dir"] != ""
        params = torch.load(train_params["pretrained_dir"])
        m.load_state_dict(params["state_dict"])
        del params
        for name, x in m.named_parameters():
            assert x.requires_grad
            print(name)

    # Load optimizer
    optimizer = train_params["optimizer"]

    # Load LR scheduler
    scheduler = train_params["lr_scheduler"]

    # Load number of epochs
    num_epochs = train_params["num_epochs"]

    # Load loss function
    loss_func = train_params["loss_func"]

    # Load from checkpoint if needed
    if train_params["resume"] != '':
        m, optimizer, _, scheduler, start_epoch_idx, best_acc, results, save_dir = \
                load_checkpoint(train_params["resume"], m, optimizer) # Load ckpt
    else:
        start_epoch_idx = 0
        best_acc = 0.0
        results = dict() # Initialize the dictionary that records losses and accuracies
        results["train"] = defaultdict(list)
        results["val"] = defaultdict(list)
        results["fname"] = save_dir + "/results.pkl"

    # Load data loaders
    train_loader, val_loader = get_dataloaders(train_params, my_transforms=train_params["image_transforms"])

    # Loop through epochs
    for epoch_idx in range(start_epoch_idx, num_epochs):
        print("Epoch {}; LR ".format(epoch_idx+1))
        for param_group in optimizer.param_groups:
            print("  ", param_group['lr'])

        # Do a train step
        _, _, _ = train_step(
            m,
            train_loader,
            optimizer,
            loss_func,
            epoch_idx,
            results
        )

        # Do a validation step
        test_loss, test_top1, _ = test_step(
            m,
            val_loader,
            loss_func,
            epoch_idx,
            results
        )

        # Step learning rate scheduler
        if train_params["scheduler"] == "plateau":
            scheduler.step(test_loss)
        elif train_params["scheduler"] == "step" or train_params["scheduler"] == "constant" \
            or train_params["scheduler"] == "multistep":
            scheduler.step()
        else:
            assert 0, "Condition should not be reached."

        # Set flag if current test set accuracy is the best so far
        best_acc, is_best = check_best_accuracy(test_top1, best_acc)

        # Every 10 epochs, save into new checkpoint file. Overwrite existing otherwise.
        if ((epoch_idx+1) % train_params["save_freq"] == 0) or (epoch_idx == 0):
            save_epoch = epoch_idx
        else:
            save_epoch = None

        save_checkpoint(
            {
                'epoch': epoch_idx+1,
                'state_dict': m.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'results': results,
                'train_params': train_params,
                'best_acc': best_acc
            }, 
            save_dir,
            is_best,
            save_epoch
        )


if __name__ == "__main__":
    import imp, argparse, shutil, copy
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/config_imagenet.py")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ### Set random seed
    seed_torch(seed=int(args.seed))

    #### Load config file
    cfg_fname = args.config.split('/')[-1]
    cfg = imp.load_source("configs", args.config)
    cfg = cfg.load_config()

    #### Save model and stats directory. Copy config file to save directory.
    save_dir = cfg["save_dir"] + '/' + cfg["exp_id"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg["save_dir"] = save_dir
    shutil.copyfile(args.config, save_dir+"/{}".format(cfg_fname))

    #### Start training
    print("Training parameters:", cfg)
    train(cfg, save_dir)



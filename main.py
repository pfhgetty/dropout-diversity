import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CyclicLR, StepLR

import advertorch.attacks as attacks
from torch.utils.tensorboard import SummaryWriter

from net import Net
import numpy as np

from loops import train, test, test_adversarial

import matplotlib.pyplot as plt
import sys
import os


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Dropout-Diversity testing script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to train/test on. Options: 'mnist', 'cifar10'. (default: 'mnist')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        metavar="N",
        help="number of epochs to train (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=False,
        help="Save models after training.",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        default=False,
        help="Load models instead of training them.",
    )
    parser.add_argument(
        "--run-adversarial",
        action="store_true",
        default=False,
        help="Run the adversarial example testing.",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if args.dataset == "mnist":
        dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    elif args.dataset == "cifar10":
        dataset1 = datasets.CIFAR10(
            "../data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.CIFAR10("../data", train=False, transform=transform)
    else:
        print("--dataset must be one of either 'mnist' or 'cifar10'.")
        sys.exit(0)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    summary = SummaryWriter("runs/baseline")
    nc_summary = SummaryWriter("runs/nc")

    # Baseline Model
    model = Net(args.dataset).to(device)
    # Model to be trained with non-correlation loss.
    nc_model = Net(args.dataset).to(device)

    if not args.load_models:
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        nc_optimizer = torch.optim.Adadelta(nc_model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}:")
            print("Baseline:")
            train(
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                nc=False,
                log_interval=10,
                summary=summary,
                name="baseline",
            )
            test(
                model,
                device,
                test_loader,
                summary=summary,
                name="baseline",
                epoch=epoch,
            )
            print("Non-Correlated:")
            train(
                nc_model,
                device,
                train_loader,
                nc_optimizer,
                epoch,
                nc=True,
                log_interval=10,
                summary=nc_summary,
                name="non-correlated",
            )
            test(
                nc_model,
                device,
                test_loader,
                summary=nc_summary,
                name="non-correlated",
                epoch=epoch,
            )

        if args.save_models:
            if not os.path.exists("saved_models/"):
                os.mkdir("saved_models/")
            torch.save(model.state_dict(), "saved_models/model.pt")
            torch.save(nc_model.state_dict(), "saved_models/nc_model.pt")
    else:
        model.load_state_dict(torch.load("saved_models/model.pt"))
        nc_model.load_state_dict(torch.load("saved_models/nc_model.pt"))

    if args.run_adversarial:
        min_eps = 0.001
        max_eps = 0.01
        x, y = test_adversarial(model, device, test_loader, summary, name="baseline", min_eps=min_eps, max_eps=max_eps)
        nc_x, nc_y = test_adversarial(
            nc_model,
            device,
            test_loader,
            nc_summary,
            name="non-correlated",
            min_eps=min_eps,
            max_eps=max_eps,
        )

        fig = plt.figure(dpi=500)
        plt.plot(x, y, "-", color="orange")
        plt.plot(nc_x, nc_y, "-", color="blue")
        plt.axis([min_eps, max_eps, 0, 1])
        plt.xlabel("Max Per-Pixel Distortion")
        plt.ylabel("Accuracy")
        plt.title("Adversarial Testing " + args.dataset.upper())
        plt.grid(True)
        if summary:
            summary.add_figure("fgsm_acc_adv", fig)

    print("Finished successfully!")


if __name__ == "__main__":
    main()
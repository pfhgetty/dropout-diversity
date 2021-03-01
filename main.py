from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CyclicLR, StepLR

from net import Net
import numpy as np
import random
import math

# torch.autograd.set_detect_anomaly(True)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0

    def soft_nll(output, target, a=0.5):
        nll = F.nll_loss(output, target, reduction='none')
        losses = (math.log(a) + nll) * torch.relu(nll + math.log(a))
        return losses

    def soft_ce (input, target):
        target = torch.exp(target)
        return  -(target * input).sum(dim=1)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        model.train()
        output = model(data)
        softmax = torch.exp(output)
        model.eval()
        output2 = model(data).detach()

        loss = 0
        loss += torch.mean(F.nll_loss(output, target))
        loss -= torch.mean(soft_ce(output2, output))
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        # u_loss_sum += u_loss.item()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    sum_loss / (batch_idx + 1),
                ),
                end="\r",
            )
            if args.dry_run:
                break
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()


    # Average loss
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
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
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    # dataset1 = torch.utils.data.Subset(dataset1, range(0, 1000))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    model = Net().to(device)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    optimizer = optim.AdaBelief(model.parameters(), lr=1e-3, rectify=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    # optimizer = optim.Adahessian(model.parameters())
    # optimizer = optim.AdaMod(model.parameters())

    scheduler = StepLR(optimizer, 1, 1)
    # scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=1000, cycle_momentum=False)
    for epoch in range(1, 100000 + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
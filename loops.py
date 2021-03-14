import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import advertorch.attacks as attacks
from advertorch.context import ctx_noparamgrad_and_eval 

import matplotlib.pyplot as plt

# Cross entropy loss for soft targets
def soft_ce (input, target):
    target = torch.exp(target)
    return  torch.mean(-(target * input).sum(dim=1))

def train(args, model, device, train_loader, optimizer, nc=False, summary=None, log_interval=10, name=None):
    sum_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        model.train()
        output = model(data)
        loss = F.nll_loss(output, target)

        # Write train loss to tensorboard
        if summary:
            summary.add_scalar("train_loss/" + name, loss.item())

        # Non-correlation loss
        if nc:
            model.eval()
            output2 = model(data).detach()
            loss -= soft_ce(output2, output)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                "[{}/{} ({:.0f}%)]\tLoss: {:.5f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    sum_loss / (batch_idx + 1),
                ),
                end="\r",
            )


def test(model, device, test_loader, attack=None, summary=None, eps=None, name=None):
    model.eval()
    test_loss = 0
    correct = 0
    with ctx_noparamgrad_and_eval(model):
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if attack:
                data = attack.perturb(data, target)

                # print(target[0].cpu().numpy())
                # example = data[0]
                # plt.imshow(example.squeeze(0).cpu().numpy(), cmap='gray')
                # plt.show()
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
    acc = correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, {}/{} ({:.0f})\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            acc,
        )
    )

    if summary:
        summary.add_scalar("test_loss/" + name, test_loss)
        summary.add_scalar("test_acc/" + name, acc)

    # Returns loss and accuracy
    return test_loss, acc

def test_adversarial(model, device, test_loader, summary=None, min_eps=0.01, max_eps=0.3, num_eps=50, name=None):
    x = np.linspace(min_eps, max_eps, num=num_eps)
    y = []
    for eps in x:
        attack = attacks.FGSM(model, loss_fn=F.nll_loss, eps=eps)
        _, accuracy = test(model, device, test, None, attack, eps)
        y.append(accuracy)

    fig = plt.figure()
    plt.plot(x, y)
    if summary:
        summary.add_scalar("fgsm_acc/" + name , fig)

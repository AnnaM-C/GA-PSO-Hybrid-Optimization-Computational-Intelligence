import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from PSO import PSO

import torch.optim as optim

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import operator
import random

import math

import cnn


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # The dataset comes in data of the range 0-1, we want to convert this to -1 to 1 as part of normalisation

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Function to transform the dataset to the specified range

    batch_size = 4

    # Downloading the training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

    # Loading the training set into a variable
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    # etc.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    # Specifying the classes of the dataset that we want to train the classifier on
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    PATH = ('GD_Model.pth')
    PopModel = cnn.Net()
    PopModel.load_state_dict(torch.load(PATH))

    for param in PopModel.parameters():
        # By setting the requires grad for each parameter to false, we no longer track it with autograd
         # Consequently, backpropagation or whatever will not compute the gradients for these layers
        param.requires_grad = False

    # Getting the final layer of the network
    finalLayer = list(PopModel.children())[-1]

    # print(len(finalLayer.weight))

    # Set the final layer to require_grad

    finalLayer.requires_grad = True

    # Reset the final layer's weights
    nn.init.xavier_uniform(finalLayer.weight)

    def convert_np_function(self, intermediate_tensor):
        intermediate_value = intermediate_tensor.numpy()
        self.intermediate_values.append(intermediate_value)

    pso=PSO(population=20)

    running_loss=0
    for x, y in trainloader:

        loss=pso.search(x,y,finalLayer,PopModel)

        running_loss+=loss
    
    print("Total loss: ", running_loss/len(trainloader))



if __name__ == '__main__':
    main()
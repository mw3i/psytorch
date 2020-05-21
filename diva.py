'''
DIVergent Autoencoder (Kurtz 2007)
based on tutorial from pytorch website
'''
import numpy as np 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

## Define Model
class DIVA(nn.Module):
    
    def __init__(self, **hps):
        super(DIVA, self).__init__()
        self.hidden1 = nn.Linear(hps['num_features'], hps['num_hidden_nodes'])
        self.output = {
            channel: nn.Linear(hps['num_hidden_nodes'], hps['num_features'])
            for channel in hps['classes']
        }

    def forward(self, x, channel):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.output[channel](x))
        return x

if __name__ == '__main__':

    ## Load Data
    inputs = torch.tensor([
        [-1,-1,-1],
        [-1,-1,1],
        [-1,1,-1],
        [-1,1,1],
        [1,1,1],
        [1,1,-1],
        [1,-1,1],
        [1,-1,-1],
    ], dtype = torch.float)

    targets = inputs / 2 + .5

    labels = [0,0,0,0,1,1,1,1]


    ## Initialize Model Instance
    hps = { # <-- hyperparameters
        'num_features': inputs.shape[1],
        'num_hidden_nodes': 3,
        'classes': [0,1],
    }

    net = DIVA(**hps)

    # criterion = nn.MSELoss()
    def criterion(outputs, targets):
        return torch.sum((outputs - targets) ** 2)
    optimizer = optim.SGD(net.parameters(), lr=1.1, momentum=0.9)


    ## Train
    presentation_order = np.arange(inputs.shape[0])
    epochs = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        np.random.shuffle(presentation_order)
        for p in presentation_order:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(inputs[p:p+1], labels[p])

            # print(labels[p])
            # exit()
            loss = criterion(outputs[p:p+1], targets[p:p+1])
            loss.backward()

            optimizer.step()


    ## Plot
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(inputs); ax[0].set_title('inputs')
    ax[1].imshow(net.forward(inputs,0).detach()); ax[1].set_title('chan0\noutputs')
    ax[2].imshow(net.forward(inputs,1).detach()); ax[2].set_title('chan1\noutputs')

    for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
    plt.tight_layout()
    plt.savefig('_/test.png')
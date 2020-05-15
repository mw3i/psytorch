'''
Multilayer Perceptron (Classifier)
based on tutorial from pytorch website
'''
import numpy as np 

import torch
import torch.optim as optim


import torch.nn as nn
import torch.nn.functional as F


## Define Model
class MLC(nn.Module):
    
    def __init__(self, **hps):
        super(MLC, self).__init__()
        self.hidden1 = nn.Linear(hps['num_features'], hps['num_hidden_nodes'])
        self.output = nn.Linear(hps['num_hidden_nodes'], hps['num_classes'])

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = self.output(x)
        return x

if __name__ == '__main__':

    ## Load Data
    inputs = torch.tensor([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,1,1],
        [1,1,0],
        [1,0,1],
        [1,0,0],
    ], dtype = torch.float)

    targets = torch.tensor([
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
    ])
    labels = torch.argmax(targets,1).type(torch.long)


    ## Initialize Model Instance
    hps = { # <-- hyperparameters
        'num_features': inputs.shape[1],
        'num_hidden_nodes': 2,
        'num_classes': targets.shape[1],
    }

    net = MLC(**hps)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.4, momentum=0.9)



    ## Train
    presentation_order = np.arange(inputs.shape[0])
    epochs = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        np.random.shuffle(presentation_order)
        for p in presentation_order:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(inputs[p:p+1])

            # print(labels[p])
            # exit()
            loss = criterion(outputs, labels[p:p+1])
            loss.backward()
            optimizer.step()


    ## Plot
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(inputs); ax[0].set_title('inputs')
    ax[1].imshow(targets); ax[1].set_title('targets')
    ax[2].imshow(F.softmax(net.forward(inputs).detach(),1)); ax[2].set_title('outputs')

    for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
    plt.tight_layout()
    plt.savefig('_/test.png')
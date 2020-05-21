'''
based on Krushke (1992)
    ^ doesn't implement any of the important things, like a psychologically plausible response rule

Note: This really isn't meant to be used for psychological modeling; it's more of a example of how something like ALCOVE could be implemented in pytorch

'''
import numpy as np 

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F


## Define Model
class ALCOVE(nn.Module):
    
    def __init__(self, **hps):
        super(ALCOVE, self).__init__()
        self.attention = nn.Linear(1,hps['num_features'])
        self.association = nn.Linear(hps['num_exemplars'], hps['num_classes'])

    def forward(self, x, e, r, c):
        distances = torch.sum(
            self.attention.weight.T * torch.abs((x.unsqueeze(1) - e.unsqueeze(0)) ** r),
        -1) ** (1/r)
        x = torch.exp(-c * distances)
        x = self.association(x)
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

    exemplars = inputs

    targets = torch.tensor([
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
    ]) # <-- i think these are normally supposed to be -1,1 in the original ALCOVE paper, but 0,1 seems to work fine ¯\_(ツ)_/¯
    labels = torch.argmax(targets,1).type(torch.long)


    ## Initialize Model Instance
    hps = { # <-- hyperparameters
        'num_features': inputs.shape[1],
        'num_exemplars': exemplars.shape[0],
        'num_classes': targets.shape[1],

        'r': 1,
        'c': 1,
    }

    net = ALCOVE(**hps)

    # criterionCrossEnt = nn.CrossEntropyLoss()
    def criterionHumbleTeachers(output, targets):
        humbleTargets = (output * targets).clamp(1) * targets
        return torch.sum((output - targets) ** 2)

    optimizer = optim.SGD(net.parameters(), lr=.04, momentum=0.9) # <-- unlike the original ALCOVE paper, this also assumes the same lr for attention learning and association learning (this could be fixed by manually updating weights, but that sort of ruins the point of using pytrch in the first place)


    ## Train
    presentation_order = np.arange(inputs.shape[0]
        )
    epochs = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        np.random.shuffle(presentation_order)
        for p in presentation_order:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(inputs, exemplars, hps['r'], hps['c'])

            # print(labels[p])
            # loss = criterionCrossEnt(outputs, labels) # <-- softmax cross entropy (like a normal MLP)
            loss = criterionHumbleTeachers(outputs, targets) # <-- humble teacher SSE cost function from Krushke 1992 (at least i think)
            loss.backward()
            optimizer.step()


    ## Plot
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(1,4)

    ax[0].imshow(inputs); ax[0].set_title('inputs')
    ax[1].imshow(net.attention.weight.detach().T); ax[1].set_title('attn weights')
    ax[2].imshow(targets); ax[2].set_title('targets')
    ax[3].imshow(F.softmax(net.forward(inputs, exemplars, hps['r'], hps['c']).detach(),1)); ax[3].set_title('outputs')

    for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
    plt.tight_layout()
    plt.savefig('_/test.png')
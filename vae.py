'''
Variational Autoencoder
based on tutorial from pytorch website
and from tutorial by Raviraja G @ https://graviraja.github.io/vanillavae/#
'''
import numpy as np 

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

## Define Model
class VAE(nn.Module):
    
    def __init__(self, **hps):
        super(VAE, self).__init__()
        self.h1 = nn.Linear(hps['num_features'],hps['num_h1_nodes'])
        self.latent_m = nn.Linear(hps['num_h1_nodes'], hps['num_latent_nodes'])
        self.latent_v = nn.Linear(hps['num_h1_nodes'], hps['num_latent_nodes'])
        self.h3 = nn.Linear(hps['num_latent_nodes'],hps['num_h3_nodes'])
        self.output = nn.Linear(hps['num_h3_nodes'], hps['num_features'])

    def encode(self, x):
        x = F.relu(self.h1(x))
        z_m = self.latent_m(x)
        z_v = self.latent_v(x)
        return x, z_m, z_v

    def decode(self, x):
        x = F.relu(self.h3(x))
        x = torch.sigmoid(self.output(x))
        return x

    def forward(self, x):
        x, z_m, z_v = self.encode(x)

        # reparameterization trick
        x = z_m + torch.exp(z_v * .5) * torch.randn_like(z_m)

        x = self.decode(x)
        return x, z_m, z_v



if __name__ == '__main__':
    import utils

    ## Load Data
    stim = utils.load_shj_stim()
    inputs = stim.reshape(stim.shape[0],-1)

    ## Initialize Model Instance
    hps = { # <-- hyperparameters
        'num_features': inputs.shape[1],
        'num_h1_nodes': 30,
        'num_latent_nodes': 3,
        'num_h3_nodes': 30,
    }

    net = VAE(**hps)

    def criterion(targets, sample, z_m, z_v, beta = 1):
        recon_loss = F.binary_cross_entropy(sample, targets, reduction='sum')
        # recon_loss = torch.sum(-targets * torch.log(sample) - (1 - targets) * torch.log(1 - sample)) # <-- binary cross entropy by hand
        
        kl_div_loss = 0.5 * torch.sum(torch.exp(z_v) + z_m**2 - 1.0 - z_v)
        return ((beta * kl_div_loss) + recon_loss) / targets.shape[0]

    optimizer = optim.Adam(net.parameters(), lr=.01)


    ## Train
    epochs = 1000
    for epoch in range(epochs):  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, z_m, z_v = net.forward(inputs)
        loss = criterion(inputs, outputs, z_m, z_v)
        loss.backward()
        optimizer.step()


    ## Plot
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(3,8)

    for i in range(inputs.shape[0]):
        ax[0,i].imshow(inputs[i].reshape(7,7), cmap = 'binary')
        ax[1,i].imshow(net.forward(inputs[i:i+1])[0].detach().reshape(7,7), cmap = 'binary')

        ax[2,i].imshow(
            net.decode(torch.randn([1,hps['num_latent_nodes']]))[0].detach().reshape(7,7),
            cmap = 'binary',
        )

    ax[0,0].set_ylabel('inputs', fontsize = 14, fontweight = 'bold')
    ax[1,0].set_ylabel('reconstructions', fontsize = 14, fontweight = 'bold')
    ax[2,0].set_ylabel('samples', fontsize = 14, fontweight = 'bold')

    for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
    plt.tight_layout()
    plt.savefig('_/test.png')
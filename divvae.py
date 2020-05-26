'''
DIVergent Variational AutoEncoder (DIVVAE); based on DIVA (Kurtz 2007)
based on tutorial from pytorch website
and from tutorial by Raviraja G @ https://graviraja.github.io/vanillavae/#
'''
import numpy as np 

import torch
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

## Define Model
class DIVVAE(nn.Module):
    
    def __init__(self, **hps):
        super(DIVVAE, self).__init__()
        self.hidden1 = nn.Linear(hps['num_features'],hps['num_h1_nodes'])
        self.latent_m = nn.Linear(hps['num_h1_nodes'], hps['num_latent_nodes'])
        self.latent_v = nn.Linear(hps['num_h1_nodes'], hps['num_latent_nodes'])
        self.hidden3 = nn.Linear(hps['num_latent_nodes'],hps['num_h3_nodes'])
        self.output = nn.ModuleDict({
            str(channel): nn.Linear(hps['num_h3_nodes'], hps['num_features'])
            for channel in hps['classes']
        })

    def encode(self, x):
        x = F.relu(self.hidden1(x))
        z_m = self.latent_m(x)
        z_v = self.latent_v(x)
        return x, z_m, z_v

    def decode(self, x, channel):
        x = F.relu(self.hidden3(x))
        x = torch.sigmoid(self.output[channel](x))
        return x

    def forward(self, x, channel):
        x, z_m, z_v = self.encode(x)

        # reparameterization trick
        x = z_m + torch.exp(z_v * .5) * torch.randn_like(z_m)

        x = self.decode(x, channel)
        return x, z_m, z_v



if __name__ == '__main__':
    import utils

    ## Load SHJ
    stim = utils.load_shj_stim()
    inputs = stim.reshape(stim.shape[0],-1)
    labels = np.array([0,0,0,0,1,1,1,1])
    categories = np.unique(labels).astype(int).astype(str).tolist()
   

    ## Load MNIST
    # train_stim, train_labels, test_stim, test_labels = utils.load_mnist()   
    # inputs = train_stim / 255
    # inputs = inputs.reshape(inputs.shape[0], -1)
    # labels = train_labels
    # categories = torch.unique(labels).numpy().astype(int).astype(str).tolist()

    ## Initialize Model Instance
    hps = { # <-- hyperparameters
        'num_features': inputs.shape[1],
        'num_h1_nodes': 30,
        'num_latent_nodes': 3,
        'num_h3_nodes': 30,
        'classes': categories,
    }

    net = DIVVAE(**hps)

    # Define Loss Function
    def criterion(targets, sample, z_m, z_v, beta = 1):
        recon_loss = F.binary_cross_entropy(sample, targets, reduction='sum')
        # recon_loss = torch.sum(-targets * torch.log(sample) - (1 - targets) * torch.log(1 - sample)) # <-- binary cross entropy by hand
        
        kl_div_loss = 0.5 * torch.sum(torch.exp(z_v) + z_m**2 - 1.0 - z_v)
        return ((beta * kl_div_loss) + recon_loss) / targets.shape[0]

    optimizer = optim.SGD(net.parameters(), lr=.005, momentum = .001)


    ## Train
    presentation_order = np.arange(len(set(hps['classes'])))
    epochs = 1000
    batch_size = 4
    for epoch in range(epochs):  # loop over the dataset multiple times

        np.random.shuffle(presentation_order)
        for p in presentation_order:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(torch.where(labels == p, inputs, np.zeros(inputs.shape)))
            
            batch_idx = np.random.choice(inputs[labels == p].shape[0], size = batch_size, replace = False)
            batch = inputs[labels == p][batch_idx]

            outputs, z_m, z_v = net.forward(batch, hps['classes'][p])
            loss = criterion(batch, outputs, z_m, z_v)
            loss.backward()

            optimizer.step()

        if epoch % 100 == 0:
            print(epoch, '|', loss.item())

    ## Plot
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(2 + len(categories),len(categories))

    for i in range(len(categories)):
        ax[0,i].imshow(inputs[labels == i][0].reshape(stim.shape[-2],stim.shape[-1]), cmap = 'binary')
        ax[1,i].imshow(net.forward(inputs[labels == i][0], hps['classes'][i])[0].detach().reshape(stim.shape[-2],stim.shape[-1]), cmap = 'binary')
        
        for _, c in enumerate(categories):
            noise = torch.randn([1,hps['num_latent_nodes']])
            ax[int(2+_),i].imshow(
                net.decode(noise, c)[0].detach().reshape(stim.shape[-2],stim.shape[-1]),
                cmap = 'binary',
            )
        

    ax[0,0].set_ylabel('orig', fontsize = 4, fontweight = 'bold')
    ax[1,0].set_ylabel('recon', fontsize = 4, fontweight = 'bold')
    for _, c in enumerate(categories): ax[int(2+_), 0].set_ylabel(str(int(c)), fontsize = 6, fontweight = 'bold')

    for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
    # plt.tight_layout()
    plt.savefig('_/test.png')
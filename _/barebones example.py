'''
based on tutorial from pytorch website
'''
import numpy as np 

import torch
import torch.optim as optim


import torch.nn as nn
import torch.nn.functional as F


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


## Initialize Model
num_hidden_nodes = 2

hidden_layer = nn.Linear(inputs.shape[1], num_hidden_nodes)
output_layer = nn.Linear(num_hidden_nodes, targets.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([hidden_layer.weight, hidden_layer.bias, output_layer.weight, output_layer.bias], lr=.7, momentum=0.5)



## Train
presentation_order = np.arange(inputs.shape[0])
epochs = 10
for epoch in range(epochs):  # loop over the dataset multiple times
    
    np.random.shuffle(presentation_order)
    for p in presentation_order:

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        hidden_activation = torch.sigmoid(hidden_layer(inputs[p:p+1]))
        output_activation = output_layer(hidden_activation)

        # backward
        loss = criterion(output_activation, labels[p:p+1])
        loss.backward()

        # optimize
        optimizer.step()


## Plot
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(1,3)

ax[0].imshow(inputs); ax[0].set_title('inputs')
ax[1].imshow(targets); ax[1].set_title('targets')
ax[2].imshow(F.softmax(output_layer(torch.sigmoid(hidden_layer(inputs))),1).detach()); ax[2].set_title('outputs')

for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
plt.tight_layout()
plt.savefig('test.png')

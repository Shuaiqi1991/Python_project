# Linear regression
#%%
import numpy as np
import torch
#%%

## setup data
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)

## setup weight and bias
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)

## define model
def model(x):
    return x @ w.t() + b

## predict
preds = model(inputs)
print(preds)

## Loss function
def mse(x1, x2):
    diff = x1-x2
    return torch.sum(diff*diff)/diff.numel()

# Train for n epochs with defined minimal improvement
n = 1000
min_improve = 0.1
for i in range(n):
    preds = model(inputs)
    loss = mse(preds, targets)
    if i == 0:
        loss_pre = loss+2
    if (i != 1 and abs(loss_pre - loss) <= min_improve):
        print('loss_pre:'+str(loss_pre.detach().numpy()))
        print(loss)
        break
    else:
        if i%10 == 0: ## report loss every 10 epochs
            print(loss)
        ## update weight and bias based on gradients
        loss.backward()
        with torch.no_grad():
            w -=w.grad * 1e-5
            b -=b.grad * 1e-5
            w.grad.zero_()
            b.grad.zero_()
        
        loss_pre = loss
        
# Use pytorch package redo linear-regression
## Define data
from torch.utils.data import TensorDataset

train_ds = TensorDataset(inputs, targets)
train_ds[0:3]

## Split data into batches
from torch.utils.data import DataLoader

batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

## Define linear model
import torch.nn as nn

model_lin = nn.Linear(3,2)
list(model_lin.parameters())

## loss function
import torch.nn.functional as F

loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
print(loss)

## optimizer
opt = torch.optim.SGD(model_lin.parameters(), lr=1e-5)

## Define Model function
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Go through all epochs
    for epoch in range(num_epochs):
        # train with batch of data
        for xb,yb in train_dl:
            # predict
            pred = model_lin(xb)
            # update loss
            loss = loss_fn(pred,yb)
            # compute gradients
            loss.backward()
            # update parameters using gradients
            opt.step()
            # reset gradient
            opt.zero_grad()
        
        # report progress
        if (epoch+1)%10 == 0:
            print('Epoch [{}/{}], Loss: {: 4f}'.format(epoch+1, num_epochs, loss.item()))

## Fit
fit(100, model_lin, loss_fn, opt, train_dl)  

## Check predit vs targets
model_lin(inputs)
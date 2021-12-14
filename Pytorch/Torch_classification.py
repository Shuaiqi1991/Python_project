# Logistic regression
#%%
# Imports
from os import access
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms.transforms import ToTensor

#%%
## get data
dt_train = MNIST(root='/Users/peAce/Documents/Pytorch/Data', download=True)
dt_test = MNIST(root='/Users/peAce/Documents/Pytorch/Data', train=False)

len(dt_train)
len(dt_test)

## check data
import matplotlib.pyplot as plt
%matplotlib inline

image, label = dt_train[0]
plt.imshow(image, cmap='gray')
print(label)

## covert data to tensor - color with pixel color scale
import torchvision.transforms as transforms
data = MNIST(root='/Users/peAce/Documents/Pytorch/Data', 
             train=True,
             transform=transforms.ToTensor())

imagetensor, label = data[0]
imagetensor.shape
plt.imshow(imagetensor[0,0:10,0:10], cmap='gray')

# Train & Validate Model
## split data
from torch.utils.data import random_split
len(data)
dt_train, dt_val = random_split(data, [50000,10000])
len(dt_train), len(dt_val)

## create batch
from torch.utils.data import DataLoader
batch_size = 100
train_load = DataLoader(dt_train, batch_size, shuffle=True)
val_load = DataLoader(dt_val, batch_size)
test_load = DataLoader

## modeling - logistic regression
## regression
import torch.nn as nn
input_size = 28*28
num_class = 10

class MNMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_class)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784) #raw data is still in 28*28 matrix, we need to reshape into 784 columns of features
        out = self.linear(xb)
        return out
    
model_log = MNMODEL()

## loss function
### generate output
import torch.nn.functional as F

for images,labels in train_load:
    print(images.shape)
    output = model_log(images)
    break

probs = F.softmax(output, dim = 1)

### evaluation/ accuracy
def acc(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

acc(output, labels)

### loss-funciton
loss_fn = F.cross_entropy

loss = loss_fn(output, labels)

## Train model
### evaluation function
def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

def evaluate(model, val_loader):
    outputs = [model.valid_step(batch) for batch in val_loader]
    return model.valid_epoch(outputs)

### modeling process
class MNMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_class)
        
    def forward(self, xb):
        xb = xb.reshape(-1,784)
        out = self.linear(xb)
        return out
    
    def train_step (self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out,label)
        return loss
        
    def valid_step (self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out,label)
        acc = accuracy(out, label)
        return {'val_loss': loss, 'val_acc': acc}
    
    def valid_epoch (self, output):
        batch_loss = [x['val_loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in output]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MNMODEL()

### fit process
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    opt=opt_func(model.parameters(), lr)
    history = []
    
    for epoch in range(epochs):
        # train
        for batch in train_loader:
            loss = model.train_step(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        # validate
        result =  evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history

### Run 
run1 = fit(5, 0.001, model, train_load, val_load)
run2 = fit(5, 0.001, model, train_load, val_load)
run3 = fit(5, 0.001, model, train_load, val_load)

### plot pred
history = run1+run2+run3
accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-o')       

### test with individual row
cvt_tensor = transforms.ToTensor()

def pred (dt_test, obs, model):
    img, label = dt_test[obs]
    x = cvt_tensor(img)
    xb = x.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    # return {'label': label , 'pred': preds[0].item()}
    print("label: {}, predict {}".format(label ,preds[0].item()))

# result = pred(dt_test, 0, model)
# print("label: {}, predict {}".format(result['label'] ,result['pred']))
pred(dt_test, 1000, model)  

### Accuracy on entire dataset
test_data = MNIST(root='/Users/peAce/Documents/Pytorch/Data', train=False, 
                  transform=transforms.ToTensor())

test_load = DataLoader(test_data, batch_size=100)
result = evaluate(model, test_load)
result

# Save Model
torch.save(model.state_dict(), 
           '/Users/peAce/Documents/Pytorch/torch_logistic.pth')

# Load Model
model2 = MNMODEL()
model2.load_state_dict(torch.load('/Users/peAce/Documents/Pytorch/torch_logistic.pth'))
model2.state_dict()

result2 = evaluate(model2, test_load)
result2

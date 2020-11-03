import torch
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
batch_size = 64

kwargs = {'batch_size': 64}
if use_cuda:
  kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('./', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('./', train=False,
                       transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

images, labels = next(iter(train_loader))
#plt.imshow(images[43].reshape(28,28), cmap="gray")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #in_channel x out_channel x kernel_size x stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    pass

def test(model, device, test_loader):
    pass

def main():

	nepochs = 5 # Can change it
	train_loss = []
	test_loss= []
	# Fill in before the epoch loop to instantiate the network model and different optimizers.
	# Also use Step wise decay Learning rate scheduler
	for epoch in range(0,nepochs):
	    train_loss.append(train(model, device, train_loader, optimizer, epoch))
	    test_loss.append(test(model, device, test_loader))
	    epochs.append(epoch)
        #keep saving the model for which test_loss is the least and use it for final reporting


if __name__=='__main__':
	main()

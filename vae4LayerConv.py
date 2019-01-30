from __future__ import print_function
import argparse
import torch
import torch.utils.data
import time
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

startTime = time.time()
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

colors = dict({0:'#e6194B', 1:'#3cb44b', 2:'#ffe119', 3:'#4363d8', 4:'#f58231', 5:'#911eb4', 6:'#42d4f4', 7:'#f032e6', 8:'#bfef45', 9:'#fabebe'})
	
"""
First Convolutional Neural Network Variational Autoencoder
Uses 4 convolutional hidden layers in the encoder before encoding a distribution
Applies 1 hidden fully-connected layer to code before output layer.

Gave test set loss of 38.7562 after 10 epochs, where loss is MSE + KLD between the encoded distribution and unit Gaussian.
Has 428,500 trainable parameters, 402,192 of which are between the hidden fully-connected layer of the decoder to the output layer
10 epochs trained in 140.349 seconds
"""
	
	
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		#(1,28,28) -> (8,26,26)
		self.conv1 = nn.Conv2d(1, 8, 3)
		
		#(8,13,13) -> (16,12,12)
		self.conv2 = nn.Conv2d(8, 16, 2)
		
		#(16,6,6) -> (32,4,4)
		self.conv3 = nn.Conv2d(16, 32, 3)
		
		#(32,4,4) -> (64,2,2)
		self.conv4 = nn.Conv2d(32, 64, 3)

		#(64,2,2) -> 2-dim mean and logvar
		self.mean = nn.Linear(64*2*2, 2)
		self.variance = nn.Linear(64*2*2, 2)

		#2 -> 512
		self.fc3 = nn.Linear(2, 512)

		#512 -> 784
		self.fc4 = nn.Linear(512, 784)

	def encode(self, x):
		return self.mean(x), self.variance(x)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x):
		#(1,28,28) -> (8,26,26) -> (8,13,13)
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

		#(8,13,13) -> (16,12,12) -> (16,6,6)
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

		#(16,6,6) -> (32,4,4)
		x = F.relu(self.conv3(x))

		#(32,4,4) -> (64,2,2)
		x = F.relu(self.conv4(x))

		#(64,2,2) -> 2-dim mean and logvar
		mu, logvar = self.encode(x.view(-1, 64*2*2))

		#get code
		z = self.reparameterize(mu, logvar)

		#decode code
		return self.decode(z), mu, logvar, z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction = 'sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, max, startTime):
	model.eval()
	test_loss = 0
	zTensor = torch.empty(0,2).to(device)
	labelTensor = torch.empty(0, dtype = torch.long)
	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			data = data.to(device)
			recon_batch, mu, logvar, z = model(data)
			test_loss += loss_function(recon_batch, data, mu, logvar).item()
			zTensor = torch.cat((zTensor, z), 0)
			labelTensor = torch.cat((labelTensor, _), 0)
			#print(zTensor.size())
	print(zTensor.size())
	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	if(epoch == max):
		print("--- %s seconds ---" % (time.time() - startTime))
		z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
		z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
		colorArray = np.empty(labelTensor.size()[0], dtype='|U7')
		for i in range(0, labelTensor.size()[0]):
			colorArray[i] = colors[labelTensor[i].item()]
		scatterPlot = plt.scatter(z1, z2, c = colorArray)
		plt.show()

if __name__ == "__main__":
	summary(model,(1,28,28))
	for epoch in range(1, args.epochs + 1):
		train(epoch)
		test(epoch, args.epochs, startTime)

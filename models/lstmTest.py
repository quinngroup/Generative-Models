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
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import paired_euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
from movingMNISTExplorer import genLoaders

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.colors as colors

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
parser.add_argument('--testSplit', type=float, default=.2, metavar='%',
                    help='portion of dataset to test on (default: .2)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, default='', metavar='s',
                    help='saves the weights to a given filepath')
parser.add_argument('--load', type=str, default='', metavar='l',
                    help='loads the weights from a given filepath')
parser.add_argument('--beta', type=float, default=1.0, metavar='b',
                    help='sets the value of beta for a beta-vae implementation')
parser.add_argument('--lsdim', type = int, default=2, metavar='ld',
                    help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                    #current implementation may not be optimal for dims above 4
parser.add_argument('--dbscan', type= bool, default= False, metavar='db',
                    help='to run dbscan clustering') 
parser.add_argument('--spectral', type= bool, default= False, metavar='spc',
                    help='to run spectral clustering')     
parser.add_argument('--hdbscan', type=bool, default=False, metavar='hdb',
                    help='to run hdbscan clustering')
parser.add_argument('--eps', type=float, default=.0001, metavar='e',
                    help='small number to prevent divide by zero errors (default: .0001)')
parser.add_argument('--source', type=str, default='../data/mnist_test_seq.npy', metavar='S',
                    help='path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
parser.add_argument('--celldim', type = int, default=2, metavar='cd',
                    help='dimension of cell state (default:2)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader = genLoaders(args.batch_size, args.no_cuda, args.seed, args.testSplit, -1, '', args.source)
    
"""
First Attempt at Making an LSTM VAE on the moving MNIST dataset

@author Quinn Wyner
"""
    
class LSTMModule(nn.Module):
    def __init__(self):
        super(LSTMModule, self).__init__()
        
        self.forget1 = nn.Linear(2*64*64, args.celldim)
        
        self.memory1 = nn.Linear(2*64*64, args.celldim)
        self.memory2 = nn.Linear(2*64*64, args.celldim)
        
        #(2,64,64) -> (16,62,62)
        self.encode1 = nn.conv2D(2, 16, 3)
        
        #(16,31,31) -> (32,30,30)
        self.encode2 = nn.conv2D(16, 32, 2)
        
        #(32,15,15) -> (64,13,13)
        self.encode3 = nn.conv2D(32, 64, 3)
        
        #(64,13,13) -> (128,11,11)
        self.encode4 = nn.conv2D(64, 128, 3)
        
        #(128,11,11) -> (256,10,10)
        self.encode5 = nn.conv2D(128, 256, 2)
        
        #256*10*10 + celldim -> lsdim mean and logvar
        self.mean = nn.Linear(256*10*10 + celldim, lsdim)
        self.variance = nn.Linear(256*10*10 + celldim, lsdim)
        
        #(args.lsdim -> 4)
        self.fc1 = nn.Linear(args.lsdim, 4)
        #reshape elsewhere
        
        #(1,2,2) -> (256,10,10)
        self.convt1 = nn.ConvTranspose2d(1, 256, 9)
        
        #(256,10,10) -> (128,15,15)
        self.convt2 = nn.ConvTranspose2d(256, 128, 6)

        #(128,15,15) -> (64,21,21)
        self.convt3 = nn.ConvTranspose2d(128, 64, 7)

        #(64,21,21) -> (32,28,28) 
        self.convt4 = nn.ConvTranspose2d(64, 32, 8)
        
        #(32,28,28) -> (16,36,36)
        self.convt5 = nn.ConvTranspose2d(32, 16, 9)
        
        #(16,36,36) -> (8,45,45)
        self.convt6 = nn.ConvTranspose2d(16, 8, 10)
        
        #(8,45,45) -> (4,54,54)
        self.convt7 = nn.ConvTranspose2d(8,4,10)
        
        #(4,54,54) -> (2,60,60)
        self.convt8 = nn.ConvTranspose2d(4,2,7)
        
        #(2,60,60) -> (1,64,64)
        self.convt9 = nn.ConvTranspose2d(2,1,5)
        
    def cellStateUpdate(self, x, c):    
        #(2,64,64) -> 2*64*64
        x = x.view(-1, 2*64*64)
        
        forgetArray = F.sigmoid(self.forget1(x))
        c = c * forgetArray
        
        rememberArray1 = F.sigmoid(self.memory1(x))
        rememberArray2 = F.tanh(self.memory2(x))
        rememberArray2 = rememberArray1 * rememberArray2
        
        return c + rememberArray2
        
    def encode(self, x, c):
        #(2,64,64) -> (16,31,31)
        x = F.max_pool2d(F.relu(self.encode1(x)), (2,2))
        
        #(16,31,31) -> (32,15,15)
        x = F.max_pool2d(F.relu(self.encode2(x)), (2,2))
        
        #(32,15,15) -> (64,13,13)
        x = F.relu(self.encode3(x))
        
        #(64,13,13) -> (128,11,11)
        x = F.relu(self.encode4(x))
        
        #(128,11,11) -> (256,10,10)
        x = F.relu(self.encode5(x))
        
        #(256,10,10) x celldim -> 256*10*10 + celldim
        x = x.view(-1, 256*10*10)
        x = torch.cat((x,c),1)
        
        return self.mean(x), self.variance(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        #implement
        d1 = F.relu(self.fc1(z))
        d1r = d1.view(-1,1,2,2)
        d2 = F.relu(self.convt1(d1r))
        d3 = F.relu(self.convt2(d2))
        d4 = F.relu(self.convt3(d3))
        d5 = F.relu(self.convt4(d4))
        d6 = F.relu(self.convt5(d5))
        d7 = F.relu(self.convt6(d6))
        d8 = F.relu(self.convt7(d7))
        d9 = F.relu(self.convt8(d8))
        d10 = F.relu(self.convt9(d9))
        return d10
    
    '''def decode(self, z):
        base = z.view(-1, args.lsdim, 1, 1)
        for i in range(63):
            base = torch.cat((base,z), 2)
        fullBase = base
        for i in range(63):
            fullBase = torch.cat((fullBase,base), 3)
        for i in range(-32, 32):
            value = 31.5/(i+.5)'''
        
    def forward(self, x, h, c):
        #(1,64,64) & (1,64,64) -> (2,64,64)
        x = torch.cat((x,h),1)
        
        c = self.cellStateUpdate(x,c)
        
        mu,logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z), c, mu, logvar, z

        
class Model(nn.Module):
    def __init__(self):
        module = LSTMModule()
    def forward(self, video):
        h = torch.zeros_like(video[:,:,0])
        cBase = torch.zeros((1,args.celldim))
        c = cBase
        for i in range(video.shape()[0] - 1):
            c = torch.cat((c, cBase), 0)
        
        

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x.view(-1,784), x.view(-1, 784), reduction = 'sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + args.beta*KLD


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
    zTensor = torch.empty(0,args.lsdim).to(device)
    labelTensor = torch.empty(0, dtype = torch.long)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            zTensor = torch.cat((zTensor, z), 0)
            labelTensor = torch.cat((labelTensor, _), 0)
    
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if(epoch == max):
        if (args.dbscan == True) :
            zScaled = StandardScaler().fit_transform((torch.Tensor.cpu(zTensor).numpy())) #re-add StandardScaler().fit_transform
            db = DBSCAN(eps= 0.7, min_samples= 3).fit(zScaled)
            print(db.labels_)
            labelTensor = db.labels_
        if (args.spectral == True) :
            spectral = SpectralClustering().fit(torch.Tensor.cpu(zTensor).numpy())
            print(spectral)
            labelTensor = spectral.labels_
        print("--- %s seconds ---" % (time.time() - startTime))
        cmap = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe'])
        
        #Handling different dimensionalities
        if (args.lsdim < 3) :
            z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
            z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
            scatterPlot = plt.scatter(z1, z2, s = 2, c = labelTensor) #Regular 2dim plot, RE-ADD CMAP = CMAP
            plt.colorbar()
        elif (args.lsdim == 3) :
            fig=plt.figure()
            ax=fig.gca(projection='3d')
            z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
            z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
            z3 = torch.Tensor.cpu(zTensor[:, 2]).numpy()
            scatterPlot = ax.scatter(z1, z2, z3, s = 2, c = labelTensor, cmap = cmap) #Regular 3dim plot
        else:    
            Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())
            z1 = Z_embedded[:, 0]
            z2 = Z_embedded[:, 1]
            scatterPlot = plt.scatter(z1, z2, s = 2, c = labelTensor, cmap = cmap) #TSNE projection for >3dim 
            plt.colorbar()

        plt.show()
         
def dplot(x):
    img = decode(x)
    plt.imshow(img)

if __name__ == "__main__":
    summary(model,(1,28,28))
    if(args.load == ''):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch, args.epochs, startTime)
    elif(args.cuda == True):
        model.load_state_dict(torch.load(args.load))
        test(args.epochs, args.epochs, startTime)
    else:
        model.load_state_dict(torch.load(args.load, map_location= device))
        test(args.epochs, args.epochs, startTime)
    if(args.save != ''):
        torch.save(model.state_dict(), args.save)
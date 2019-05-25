from __future__ import print_function
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import time
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from mMNISTflat import genLoaders
from vamps.NatVampPrior import log_Normal_diag, VAE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.colors as colors

startTime = time.time()
parser = argparse.ArgumentParser(description='VtPVAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--testSplit', type=float, default=.2, metavar='%',
                    help='portion of dataset to test on (default: .2)')
parser.add_argument('--source', type=str, default='../data/mnist_test_seq.npy', metavar='S',
                    help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, default='', metavar='s',
                    help='saves the weights to a given filepath')
parser.add_argument('--load', type=str, default='', metavar='l',
                    help='loads the weights from a given filepath')
parser.add_argument('--beta', type=float, default=1.0, metavar='b',
                    help='sets the value of beta for a beta-vae implementation')
parser.add_argument('--pseudos', type=int, default=10, metavar='p',
                    help='Number of pseudo-inputs (default: 10)')
parser.add_argument('--lsdim', type = int, default=2, metavar='ld',
                    help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                    #current implementation may not be optimal for dims above 4
parser.add_argument('--gamma', type = float, default=10, metavar='g',
                    help='Pseudo-loss weight')
parser.add_argument('--dbscan', type= bool, default= False, metavar='db',
                    help='to run dbscan clustering')                                      
parser.add_argument('--input_length', type=int, default=64, metavar='il',
                    help='length and height of one image')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Loads moving MNIST dataset
mnist = np.load(args.source)

#movingMNISTDataset class    
class movingMNISTDataset(Dataset):
    """
    Initializes dataset
    @param npArray (numpy.array): moving MNIST dataset
    @param transform(callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, npArray, transform=None):
        self.npArray = npArray
        self.frameCount = (self.npArray.shape)[0]
        self.vidCount = (self.npArray.shape)[1]
        self.transform = transform
       
    """
    Gets number of observations in dataset
    @return number of observations
    """
    def __len__(self):
        return self.frameCount * self.vidCount
    
    """
    Gets the observation at a given index
    @param index (int): index corresponding to observation which is to be returned
    @return Tensor observation corresponding to given index
    """
    def __getitem__(self, index):
        obs = self.npArray[index % self.frameCount, index // self.frameCount,:,:,np.newaxis]
        if self.transform:
            obs = self.transform(obs)
        return obs

#Constructs Pytorch Dataset from moving MNIST data
data = movingMNISTDataset(npArray=mnist, transform=transforms.ToTensor())
train_loader, test_loader = genLoaders(data, args.batch_size, args.no_cuda, args.seed, args.testSplit)
    
"""
First attempt at Video-to-Path VAE on moving MNIST dataset.

@author Davis Jackson & Quinn Wyner
"""
    
#First attempt at replacement of NatVampPrior VAE
"""    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #(1,64,64) -> (8,60,60)
        self.conv1 = nn.Conv2d(1, 8, 5)
        
        #(8,30,30) -> (16,24,24)
        self.conv2 = nn.Conv2d(8, 16, 7)
        
        #(16,12,12) -> (32,6,6)
        self.conv3 = nn.Conv2d(16, 32, 7)
        
        #(32,6,6) -> (64,2,2)
        self.conv4 = nn.Conv2d(32, 64, 5)

        #64*2*2 -> lsdim mean and logvar
        self.mean = nn.Linear(64*2*2, args.lsdim)
        self.variance = nn.Linear(64*2*2, args.lsdim)

        #(args.lsdim -> 4)
        self.fc1 = nn.Linear(args.lsdim, 4)
        #reshape elsewhere
        
        #(1,2,2) -> (64,10,10)
        self.convt1 = nn.ConvTranspose2d(1, 64, 9)
        
        #(64,10,10) -> (32, 20, 20)
        self.convt2 = nn.ConvTranspose2d(64, 32, 11)

        #(32,20,20) -> (16, 32, 32)
        self.convt3 = nn.ConvTranspose2d(32, 16, 13)

        #(16,32,32) -> (8, 46, 46) 
        self.convt4 = nn.ConvTranspose2d(16, 8, 15)
        
        #(8,46,46) -> (4, 56, 56) 
        self.convt5 = nn.ConvTranspose2d(8, 4, 11)
        
        #(4,56,56) -> (1, 64, 64) 
        self.convt6 = nn.ConvTranspose2d(4, 1, 9)
        
        
        

    def encode(self, x):
        return self.mean(x), self.variance(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #implement
        x = F.relu(self.fc1(z))
        x = x.view(-1,1,2,2)
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt5(x))
        x = F.relu(self.convt6(x))
        return x
        

    def forward(self, x):
        #(1,64,64) -> (8,60,60) -> (8,30,30)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        
        #(8,30,30) -> (16,24,24) -> (16,12,12)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        
        #(16,12,12) -> (32,6,6)
        x = F.relu(self.conv3(x))
        
        #(32,6,6) -> (64,2,2)
        x = F.relu(self.conv4(x))

        #(64,2,2) -> lsdim mean and logvar
        mu, logvar = self.encode(x.view(-1, 64*2*2))

        #get code
        z = self.reparameterize(mu, logvar)

        #decode code
        return self.decode(z), mu, logvar, z
"""


model = VAE(args.input_length, args.lsdim, args.pseudos, args.beta, args.gamma, args.batch_size, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        pseudos=model.means(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        loss = model.loss_function(recon_batch, data, mu, logvar, z,pseudos,recon_pseudos, p_mu, p_logvar, p_z)
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
    if(args.save != ''):
            torch.save(model.state_dict(), args.save)
    model.eval()
    test_loss = 0
    zTensor = torch.empty(0,args.lsdim).to(device)
    pseudos=model.means(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
    recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += model.loss_function(recon_batch, data, mu, logvar,z,pseudos,recon_pseudos, p_mu, p_logvar, p_z).item()
            zTensor = torch.cat((zTensor, z), 0)
    
    if (args.dbscan == True) :
        zScaled = StandardScaler().fit_transform((torch.Tensor.cpu(zTensor).numpy())) #re-add StandardScaler().fit_transform
        db = DBSCAN(eps= 0.7, min_samples= 3).fit(zScaled)
        print(db.labels_)
        labelTensor = db.labels_
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if(epoch == max):
        print("--- %s seconds ---" % (time.time() - startTime))
        cmap = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe'])
        
        #Handling different dimensionalities
        if (args.lsdim < 3) :
            z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
            z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
            scatterPlot = plt.scatter(z1, z2, s = 4) #Regular 2dim plot, RE-ADD CMAP = CMAP
        elif (args.lsdim == 3) :
            fig=plt.figure()
            ax=fig.gca(projection='3d')
            z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
            z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
            z3 = torch.Tensor.cpu(zTensor[:, 2]).numpy()
            scatterPlot = ax.scatter(z1, z2, z3, s = 4) #Regular 3dim plot
        else:    
            Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())        
            z1 = Z_embedded[:, 0]
            z2 = Z_embedded[:, 1]
            scatterPlot = plt.scatter(z1, z2, s = 4) #TSNE projection for >3dim 

        plt.show()
        temp = model.means(model.idle_input).view(-1,args.input_length,args.input_length).detach().cpu()
        for x in range(args.pseudos):
            plt.matshow(temp[x].numpy())
            plt.show()
         
def dplot(x):
    img = decode(x)
    plt.imshow(img)

if __name__ == "__main__":
    summary(model,(1,args.input_length,args.input_length))
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
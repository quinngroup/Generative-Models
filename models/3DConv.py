from __future__ import print_function
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import time
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.nn import LeakyReLU
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from mMNISTflat import genLoaders

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.colors as colors

"""
3D Convolutional Network for use on Moving MNIST dataset

@author Quinn Wyner
"""
class VAE(nn.Module):
    def __init__(self, input_length, input_depth, lsdim, beta, batch_size, device):
        super(VAE, self).__init__()
        
        self.input_length = input_length
        self.input_depth = input_depth
        self.lsdim = lsdim
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        
        self.finalConvLength = ((input_length - 4)//2 - 6)//2 - 10
        self.finalConvDepth = (input_depth - 4)//2 - 6
        
        #(1,20,64,64) -> (8,18,56,56)
        self.conv1 = nn.Conv3d(1, 8, (3,9,9))
        
        #(8,18,28,28) -> (16,16,22,22)
        self.conv2 = nn.Conv3d(8, 16, (3,7,7))
        
        #(16,8,11,11) -> (32,4,6,6)
        self.conv3 = nn.Conv3d(16, 32, (5,6,6))
        
        #(32,4,6,6) -> (64,2,2,2)
        self.conv4 = nn.Conv3d(32, 64, (3,5,5))
        
        #64*2*2*2 -> lsdim mean and logvar
        self.mean = nn.Linear(64*self.finalConvDepth*self.finalConvLength*self.finalConvLength, lsdim)
        self.logvar = nn.Linear(64*self.finalConvDepth*self.finalConvLength*self.finalConvLength, lsdim)
        
        self.conv5 = nn.Conv3d(lsdim + 3, 16, 3, padding=1)
        self.conv6 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv7 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv8 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv9 = nn.Conv3d(16, 1, 1)
        
    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        #(1,20,64,64) -> (8,18,56,56) -> (8,18,28,28)
        x = F.max_pool3d(LeakyReLU(0.1)(self.conv1(x)), (1,2,2))
        
        #(8,18,30,30) -> (16,16,22,22) -> (16,8,11,11)
        x = F.max_pool3d(LeakyReLU(0.1)(self.conv2(x)), (2,2,2))
        
        #(16,8,11,11) -> (32,4,6,6)
        x = LeakyReLU(0.1)(self.conv3(x))
        
        #(32,4,6,6) -> (64,2,2,2)
        x = LeakyReLU(0.1)(self.conv4(x))
        
        #(64,2,2,2) -> 64*2*2*2
        x = x.view(-1, 64*self.finalConvDepth*self.finalConvLength*self.finalConvLength)
        
        #64*2*2*2 -> lsdim mean and logvar
        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        
        return z_q_mean, z_q_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # p(x|z)
    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        baseVector = z.view(-1, self.lsdim, 1, 1, 1)
        base = baseVector.repeat(1, 1, self.input_depth, self.input_length, self.input_length)
        
        depthStepTensor = torch.linspace(-1, 1, self.input_depth).to(self.device)
        lengthStepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)
        
        depthAxisVector = depthStepTensor.view(1, 1, self.input_depth, 1, 1)
        xAxisVector = lengthStepTensor.view(1, 1, 1, self.input_length, 1)
        yAxisVector = lengthStepTensor.view(1, 1, 1, 1, self.input_length)
        
        depthBase = depthAxisVector.repeat(base.shape[0], 1, 1, self.input_length, self.input_length)
        xBase = xAxisVector.repeat(base.shape[0], 1, self.input_depth, 1, self.input_length)
        yBase = yAxisVector.repeat(base.shape[0], 1, self.input_depth, self.input_length, 1)
        
        base = torch.cat((depthBase, xBase, yBase, base), 1)
        
        x = F.leaky_relu(self.conv5(base))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        
        return x
        
    def forward(self, x):
        
        #z~q(z|x)
        mu, logvar = self.q_z(x)

        z = self.reparameterize(mu, logvar)
        
        #decode code
        x_mean=self.p_x(z)
        return x_mean, mu, logvar, z
        
    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x.view(-1,self.input_depth*self.input_length*self.input_length), x.view(-1, self.input_depth*self.input_length*self.input_length), reduction = 'sum')
        
        #KL
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return MSE + args.beta*KL
        
def videoTransform(array):
    x = torch.from_numpy(array).float()
    x = x/255.0
    return x
        
#movingMNISTDataset class
class movingMNISTDataset(Dataset):
    """
    Initializes dataset
    @param npArray (numpy.array): moving MNIST dataset
    @param transform(callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, npArray, transform=None):
        self.npArray = npArray
        self.transform = transform
       
    """
    Gets number of observations in dataset
    @return number of observations
    """
    def __len__(self):
        return (self.npArray.shape)[1]
    
    """
    Gets the observation at a given index
    @param index (int): index corresponding to observation which is to be returned
    @return Tensor observation corresponding to given index
    """
    def __getitem__(self, index):
        obs = self.npArray[np.newaxis,:,index,:,:]
        if self.transform:
            obs = self.transform(obs)
        return obs        
        
if __name__ == "__main__":

    startTime = time.time()
    parser = argparse.ArgumentParser(description='3DConv')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--testSplit', type=float, default=.05, metavar='%',
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
    parser.add_argument('--lsdim', type = int, default=2, metavar='ld',
                        help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                        #current implementation may not be optimal for dims above 4
    parser.add_argument('--dbscan', type= bool, default= False, metavar='db',
                        help='to run dbscan clustering')                                      
    parser.add_argument('--input_length', type=int, default=64, metavar='il',
                        help='length and height of one image')
    parser.add_argument('--input_depth', type=int, default=20, metavar='id',
                        help='number of frames in one video')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #Loads moving MNIST dataset
    mnist = np.load(args.source)

    #Constructs Pytorch Dataset from moving MNIST data
    data = movingMNISTDataset(npArray=mnist, transform=videoTransform)
    train_loader, test_loader = genLoaders(data, args.batch_size, args.no_cuda, args.seed, args.testSplit)
    enumerate(train_loader)
    model = VAE(args.input_length, args.input_depth, args.lsdim, args.beta, args.batch_size, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
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
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)
                test_loss += model.loss_function(recon_batch, data, mu, logvar).item()
                zTensor = torch.cat((zTensor, z), 0)
        
        if (args.dbscan == True) :
            zScaled = StandardScaler().fit_transform((torch.Tensor.cpu(zTensor).numpy())) #re-add StandardScaler().fit_transform
            db = DBSCAN(eps= 0.7, min_samples= 3).fit(zScaled)
            print(db.labels_)
            labelTensor = db.labels_
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        if(epoch == max):
            if(args.save != ''):
                torch.save(model.state_dict(), args.save)
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

    summary(model,(1,args.input_depth,args.input_length,args.input_length))
    if(args.load == ''):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            #test(epoch, args.epochs, startTime)
    elif(args.cuda == True):
        model.load_state_dict(torch.load(args.load))
        test(args.epochs, args.epochs, startTime)
    else:
        model.load_state_dict(torch.load(args.load, map_location= device))
        test(args.epochs, args.epochs, startTime)
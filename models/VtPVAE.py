import torch
<<<<<<< HEAD
from torch import nn
from vamps.NatVampPrior import NatVampPrior
import VtPVAE_D
=======
import time
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
from torchsummary import summary
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datasetTemplate import frameDataset
from mMNISTflat import genLoaders

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys,os

#os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0,'../')
#print(os.getcwd())

#print(os.listdir())
from vamps.NatVampPrior import NatVampPrior
from vamps.NatVampPriorBatch import BatchVampPrior




'''
Appearance head for Video-to-Path model
Implements NatVampPrior on frames of a video dataset

@author Davis Jackson, Quinn Wyner
'''

startTime = time.time()
parser = argparse.ArgumentParser(description='VtPVAE')
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
parser.add_argument('--pseudos', type=int, default=10, metavar='p',
                    help='Number of pseudo-inputs (default: 10)')
parser.add_argument('--lsdim', type = int, default=10, metavar='ld',
                    help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                    #current implementation may not be optimal for dims above 4
parser.add_argument('--gamma', type = float, default=.05, metavar='g',
                    help='Pseudo-loss weight')
parser.add_argument('--lr', type = float, default=1e-3, metavar='lr',
                    help='learning rate')
parser.add_argument('--plr', type = float, default=4e-6, metavar='plr',
                    help='pseudoinput learning rate')
parser.add_argument('--logvar-bound', type=float, default=-1.0, metavar='lb',
                    help='Lower bound on logvar (default: -1.0)')
parser.add_argument('--dbscan', action='store_true', default= False,
                    help='to run dbscan clustering')      
parser.add_argument('--graph', action='store_true', default= False,
                    help='flag to determine whether or not to run automatic graphing')      
parser.add_argument('--input_length', type=int, default=64, metavar='il',
                    help='length and height of one image')
parser.add_argument('--repeat', action='store_true', default=False,
                    help='determines whether to enact further training after loading weights')
parser.add_argument('--pp', type = int, default=0, metavar='pp',
                    help='Plot pseudos. Controls the number of pseudo inputs to be displayed')
parser.add_argument('--log', type=str, default='!', metavar='lg',
                    help='flag to determine whether to use tensorboard for logging. Default \'!\' is read to mean no logging')      
parser.add_argument('--schedule', type = int, default=-1, metavar='sp',
                    help='use learning rate scheduler on loss stagnation with input patience')
parser.add_argument('--reg2', type = float, default=0, metavar='rg2',
                    help='coefficient for L2 weight decay')
parser.add_argument('--noEarlyStop', action='store_true', default=False,
                    help='disables early stopping')
parser.add_argument('--tolerance', type = float, default=.1, metavar='tol',
                    help='tolerance value for early stopping')
parser.add_argument('--patience', type = int, default = 10, metavar='pat',
                    help='patience value for early stopping')
parser.add_argument('--failCount', type=str, default='r', metavar='fc',
                    help='determines how to reset early-stopping failed epoch counter. Options are \'r\' for reset and \'c\' for cumulative')
parser.add_argument('--batchNorm', action='store_true', default=False,
                    help='Use batch normalization')
parser.add_argument('--batch-tracking', action='store_true', default=False,
                    help='Tracks running statistics in Batch Normalization layers.')
parser.add_argument('--tsne', action='store_true', default=False,
                    help='Uses TSNE projection instead of UMAP.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = "cuda" if args.cuda else "cpu"
>>>>>>> 649559589a89422a7c51b46566cf8806a7df8975


<<<<<<< HEAD
class VtPVAE(nn.Module):
    def  __init__(self):
        super(VtPVAE, self).__init__()
        
        app_head = NatVampPrior() # NEEDS MORE ARGUMENTS I THINK
        dyn_head = VTP_D()
=======
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Loads moving MNIST dataset
if args.source.endswith('.npy'):
    mnist = np.load(args.source,allow_pickle=True)

writer=None
if(args.log!='!'):
    if(args.log=='$'):
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir=args.log)


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
if args.source.endswith('.npy'):
    data = movingMNISTDataset(npArray=mnist, transform=transforms.ToTensor())
else:
    data = frameDataset(source=args.source, transform=transforms.ToTensor())
train_loader, test_loader = genLoaders(data, args.batch_size, args.no_cuda, args.seed, args.testSplit)
>>>>>>> 649559589a89422a7c51b46566cf8806a7df8975
    
        
<<<<<<< HEAD
        
        
=======
        #Handling different dimensionalities
        if(args.graph):
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
            elif args.tsne:    
                Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())        
                z1 = Z_embedded[:, 0]
                z2 = Z_embedded[:, 1]
                scatterPlot = plt.scatter(z1, z2, s = 4) #TSNE projection for >3dim 
            else:
                reducer = umap.UMAP()
                Z_embedded = reducer.fit_transform(zTensor.cpu())
                scatterPlot = plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], s = 4)

            plt.show()
        if(args.pp>0):
            t=min(args.pp,args.pseudos)
            temp = model.means(model.idle_input).view(-1,args.input_length,args.input_length).detach().cpu()
            for x in range(t):
                plt.matshow(temp[x].numpy())
                plt.show()
      
def dplot(x):
    img = decode(x)
    plt.imshow(img)
>>>>>>> 649559589a89422a7c51b46566cf8806a7df8975


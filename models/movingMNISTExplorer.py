import numpy as np
import imageio
import argparse
import torch
from math import ceil
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

"""
Loads video data from the observation in the moving MNIST dataset corresponding to given index
Saves the data as an mp4 named by the provided filename

Also constructs a Pytorch Dataset from the moving MNIST data
Splits the dataset into training and testing sets and constructs loaders for them

Uses imageio. Requires installation of ffmpeg.

@author Quinn Wyner
"""

#Parses user arguments
parser = argparse.ArgumentParser(description='Moving MNIST Explorer')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--testSplit', type=float, default=.2, metavar='%',
                    help='portion of dataset to test on (default: .2)')
parser.add_argument('--index', type=int, default=-1, metavar='i',
                    help = 'index from 0 to 9999 in moving mnist dataset from which to generate a video')
parser.add_argument('--filename', type=str, default='', metavar='F',
                    help = 'name of file to which the video should be saved')
parser.add_argument('--source', type=str, default='../data/mnist_test_seq.npy', metavar='S',
                    help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
torch.manual_seed(args.seed)

#Loads moving MNIST dataset
mnist = np.load(args.source)

#Saves video as mp4 or raises Exception if invalid index and filename are provided
if(args.index >= 0 and args.index < 10000):
    if(args.filename != ''):
        imageio.mimwrite(args.filename, mnist[:,args.index], fps=10)
    else:
        raise Exception('filename must be defined')
elif(args.filename != ''):
    raise Exception('valid index between 0 and 9999 must be defined')

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
        obs = self.npArray[:,index]
        if self.transform:
            obs = self.transform(obs)
        return obs

#Constructs Pytorch Dataset from moving MNIST data
data = movingMNISTDataset(npArray=mnist, transform=transforms.ToTensor())
length = data.__len__()

#Splits data into training and testing data
if(args.testSplit <= 0 or args.testSplit >= 1):
    raise Exception('testSplit must be between 0 and 1 (exclusively)')
testSize = ceil(args.testSplit * length)
trainSize = length - testSize
trainSet, testSet = random_split(data, [trainSize, testSize])

#Constructs DataLoaders for training and testing data
train_loader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(testSet, batch_size=args.batch_size, shuffle=True, **kwargs)
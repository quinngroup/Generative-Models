import numpy as np
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


@author Quinn Wyner, Davis Jackson
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
parser.add_argument('--source', type=str, default='../data/mnist_test_seq.npy', metavar='S',
                    help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
args = parser.parse_args()

def genLoaders(batch_size=128, no_cuda=False, seed=1, testSplit=.2, index=-1, filename='', source='../data/mnist_test_seq.npy'):
    cuda = not no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    torch.manual_seed(seed)

    #Loads moving MNIST dataset
    mnist = np.load(source)

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
            print(obs.shape)
            if self.transform:
                obs = self.transform(obs)
            return obs
    
    #Constructs Pytorch Dataset from moving MNIST data
    data = movingMNISTDataset(npArray=mnist, transform=transforms.ToTensor())
    length = data.__len__()

    #Splits data into training and testing data
    if(testSplit <= 0 or testSplit >= 1):
        raise Exception('testSplit must be between 0 and 1 (exclusively)')
    testSize = ceil(testSplit * length)
    trainSize = length - testSize
    trainSet, testSet = random_split(data, [trainSize, testSize])

    #Constructs DataLoaders for training and testing data
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True, **kwargs)
    
    print(train_loader.__len__())
    print(test_loader.__len__())
    return train_loader, test_loader

genLoaders(args.batch_size, args.no_cuda, args.seed, args.testSplit, args.source)
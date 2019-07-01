from numpy import load, newaxis, sort
import os
from torch.utils.data import Dataset

'''
Cilia dataset with overlapping clips of fixed length

@param source directory containing the source videos
@param clipLength length of each clip to be constructed
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class overlapDataset(Dataset):
    def __init__(self, source, clipLength, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.clipLength = clipLength
        self.videoLengths = [load(source + '/' + self.videos[n],mmap_mode='r').shape[0] - clipLength + 1 for n in range(len(self.videos))]
        self.transform = transform
    def __len__(self):
        return sum(self.videoLengths)
    def __getitem__(self, index):
        tempIndex = index
        currVideo = -1
        while(tempIndex >= 0):
            currVideo += 1
            tempIndex -= self.videoLengths[currVideo]
        tempIndex += self.videoLengths[currVideo]
        obs = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')[tempIndex:tempIndex+self.clipLength]
        if self.transform:
            obs = self.transform(obs)
        return obs
        
'''
Cilia dataset with non-overlapping clips of fixed length

@param source directory containing the source videos
@param clipLength length of each clip to be constructed
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class nonOverlapDataset(Dataset):
    def __init__(self, source, clipLength, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.clipLength = clipLength
        self.videoLengths = [load(source + '/' + self.videos[n],mmap_mode='r').shape[0] // clipLength for n in range(len(self.videos))]
        self.transform = transform
    def __len__(self):
        return sum(self.videoLengths)
    def __getitem__(self, index):
        tempIndex = index
        currVideo = -1
        while(tempIndex >= 0):
            currVideo += 1
            tempIndex -= self.videoLengths[currVideo]
        tempIndex += self.videoLengths[currVideo]
        obs = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')[(tempIndex*self.clipLength):(tempIndex*self.clipLength)+self.clipLength]
        if self.transform:
            obs = self.transform(obs)
        return obs
        
'''
Cilia dataset split into individual frames

@param source directory containing the source videos
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class frameDataset(Dataset):
    def __init__(self, source, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.videoLengths = [load(source + '/' + self.videos[n],mmap_mode='r').shape[0] for n in range(len(self.videos))]
        print(self.videoLengths)
        self.transform = transform
    def __len__(self):
        return sum(self.videoLengths)
    def __getitem__(self, index):
        tempIndex = index
        currVideo = -1
        while(tempIndex >= 0):
            currVideo += 1
            tempIndex -= self.videoLengths[currVideo]
        tempIndex += self.videoLengths[currVideo]
        obs = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')[tempIndex, :, :, newaxis]
        if self.transform:
            obs = self.transform(obs)
        return obs
        
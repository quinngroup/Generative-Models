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
        
'''
Cilia dataset split into non-overlapping mxn windows of individual frames

@param source directory containing the source videos
@param m the height of each window
@param n the width of each window
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class nonOverlapWindowDataset(Dataset):
    def __init__(self, source, m, n, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.windowHeight = m
        self.windowWidth = n
        self.videoLengths = []
        for i in range(len(self.videos)):
            array = load(source + '/' + self.videos[i],mmap_mode='r')
            self.videoLengths.append(array.shape[0] * (array.shape[1]//m) * (array.shape[2]//n))
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
        array = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')
        
        horFrames = array.shape[2]//self.windowWidth
        frameSeparator = (array.shape[1]//self.windowHeight) * horFrames
        
        row = ((tempIndex % frameSeparator) // horFrames) * self.windowHeight
        col = (tempIndex % horFrames) * self.windowWidth
        
        obs = array[tempIndex // frameSeparator, row:(row+self.windowHeight), col:(col+self.windowWidth), newaxis]
        if self.transform:
            obs = self.transform(obs)
        return obs
        
'''
Cilia dataset split into overlapping mxn windows of individual frames

@param source directory containing the source videos
@param m the height of each window
@param n the width of each window
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class overlapWindowDataset(Dataset):
    def __init__(self, source, m, n, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.windowHeight = m
        self.windowWidth = n
        self.videoLengths = []
        for i in range(len(self.videos)):
            array = load(source + '/' + self.videos[i],mmap_mode='r')
            self.videoLengths.append(array.shape[0] * (array.shape[1] - m + 1) * (array.shape[2] - n + 1))
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
        array = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')
        
        horFrames = array.shape[2] - self.windowWidth + 1
        frameSeparator = (array.shape[1] - self.windowHeight + 1) * horFrames
        
        row = ((tempIndex % frameSeparator) // horFrames)
        col = tempIndex % horFrames
        
        obs = array[tempIndex // frameSeparator, row:(row+self.windowHeight), col:(col+self.windowWidth), newaxis]
        if self.transform:
            obs = self.transform(obs)
        return obs
        
'''
Cilia dataset split into non-overlapping clips of length clipLength and frame-size mxn

@param source directory containing the source videos
@param clipLength duration of a clip
@param m the height of each window
@param n the width of each window
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class nonOverlapClipDataset(Dataset):
    def __init__(self, source, clipLength, m, n, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.clipLength = clipLength
        self.windowHeight = m
        self.windowWidth = n
        self.videoLengths = []
        for i in range(len(self.videos)):
            array = load(source + '/' + self.videos[i], mmap_mode='r')
            self.videoLengths.append((array.shape[0] // self.clipLength) * (array.shape[1] // m) * (array.shape[2] // n))
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
        array = load(self.source + '/' + self.videos[currVideo], mmap_mode='r+')
        
        horFrames = array.shape[2]//self.windowWidth
        frameSeparator = (array.shape[1]//self.windowHeight) * horFrames
        
        row = ((tempIndex % frameSeparator) // horFrames) * self.windowHeight
        col = (tempIndex % horFrames) * self.windowWidth
        time = (tempIndex // frameSeparator) * self.clipLength
        
        obs = array[time:(time+self.clipLength), row:(row+self.windowHeight), col:(col+self.windowWidth)]
        
        if self.transform:
            obs = self.transform(obs)
        return obs
        
'''
Cilia dataset split into overlapping clips of length clipLength and frame-size mxn

@param source directory containing the source videos
@param clipLength duration of a clip
@param m the height of each window
@param n the width of each window
@param transform transform to be performed on observations

@author Quinn Wyner
'''
class overlapClipDataset(Dataset):
    def __init__(self, source, clipLength, m, n, transform=None):
        self.source = source
        self.videos = os.listdir(source)
        sort(self.videos)
        self.clipLength = clipLength
        self.windowHeight = m
        self.windowWidth = n
        self.videoLengths = []
        for i in range(len(self.videos)):
            array = load(source + '/' + self.videos[i],mmap_mode='r')
            self.videoLengths.append((array.shape[0] - clipLength + 1) * (array.shape[1] - m + 1) * (array.shape[2] - n + 1))
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
        array = load(self.source + '/' + self.videos[currVideo],mmap_mode='r+')
        
        horFrames = array.shape[2] - self.windowWidth + 1
        frameSeparator = (array.shape[1] - self.windowHeight + 1) * horFrames
        
        row = ((tempIndex % frameSeparator) // horFrames)
        col = tempIndex % horFrames
        time = (tempIndex // frameSeparator)
        
        obs = array[time:(time+self.clipLength), row:(row+self.windowHeight), col:(col+self.windowWidth)]
        if self.transform:
            obs = self.transform(obs)
        return obs
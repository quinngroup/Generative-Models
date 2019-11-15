import argparse
import numpy as np
from random import randint

'''
Generates a clipped dataset from a video dataset focusing on a clip of n frames of each video
Treats 0th axis as time, 1st axis as height, 2nd as width

@author Quinn Wyner
'''

def clipper(filename, clipLength):
    data = np.load(filename, mmap_mode='r')
    
    clipStart = randint(0, shape[0] - clipLength)
    
    return data[clipStart:clipStart+clipLength]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clippedDataGen')
    parser.add_argument('--loadDirectory', type=str, default='.', metavar='ld',
                        help = 'Name of numpy array file to load (default=\'.\')')
    parser.add_argument('--clipLength', type=int, default=40, metavar='cl',
                        help = 'Length of a single clip (default=40)')
    parser.add_argument('--saveDirectory', type=str, default='clipped/', metavar='sd',
                        help = 'Directory in which to save files (default=\'clipped/\')')
    args = parser.parse_args()
    
    for file in os.listdir(args.loadDirectory):
        if file.endswith('.npy'):
            np.save(args.saveDirectory + file, clipper(file, args.clipLength))
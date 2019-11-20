import argparse
import numpy as np
from random import randint

'''
Generates a cropped dataset from a video dataset focusing on an m x n patch of each video
Treats 0th axis as time, 1st axis as height, 2nd as width

@author Quinn Wyner
'''

def cropper(filename, patchHeight, patchWidth):
    data = np.load(args.loadDirectory+filename, mmap_mode='r')
    
    heightStart = randint(0, data.shape[1] - patchHeight)
    widthStart = randint(0, data.shape[2] - patchWidth)
    
    return data[:, heightStart:heightStart+patchHeight, widthStart:widthStart+patchWidth]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='croppedDataGen')
    parser.add_argument('--loadDirectory', type=str, default='./', metavar='ld',
                        help = 'Name of numpy array file to load (default=\'.\')')
    parser.add_argument('--patchHeight', type=int, default=128, metavar='ph',
                        help = 'Height of a single patch (default=128)')
    parser.add_argument('--patchWidth', type=int, default=128, metavar='pw',
                        help = 'Width of a single patch (default=128)')
    parser.add_argument('--saveDirectory', type=str, default='cropped/', metavar='sd',
                        help = 'Directory in which to save files (default=\'cropped/\')')
    args = parser.parse_args()
    
    for file in os.listdir(args.loadDirectory):
        if file.endswith('.npy'):
            np.save(args.saveDirectory + file, cropper(file, args.patchHeight, args.patchWidth))
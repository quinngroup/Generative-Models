import argparse
import numpy as np
import os

'''
Generates a cropped dataset from a video dataset that splits each video into m x n patches
Treats 0th axis as time, 1st axis as height, 2nd as width

@author Quinn Wyner
'''

parser = argparse.ArgumentParser(description='croppedDataGen')
parser.add_argument('--loadDirectory', type=str, default='./', metavar='ld',
                    help = 'Name of numpy array file to load (default=\'.\')')
parser.add_argument('--patchHeight', type=int, default=128, metavar='ph',
                    help = 'Height of a single patch (default=128)')
parser.add_argument('--patchWidth', type=int, default=128, metavar='pw',
                    help = 'Width of a single patch (default=128)')
parser.add_argument('--saveDirectory', type=str, default='patched/', metavar='sd',
                    help = 'Directory in which to save files (default=\'patched/\')')
args = parser.parse_args()

for file in os.listdir(args.loadDirectory):
    if file.endswith('.npy'):
        data = np.load(args.loadDirectory+file, mmap_mode='r')
        index = 1
        for i in range(data.shape[1] // patchHeight):
            for j in range(data.shape[2] // patchWidth):
                np.save(args.saveDirectory + file.replace('.npy', '') + '_' + index, data[:, i*patchHeight:(i+1)*patchHeight, j*patchWidth:(j+1)*patchWidth])
                index += 1
import numpy as np
import imageio
import argparse

"""
Loads video data from the observation in the moving MNIST dataset corresponding to given index
Saves the data as an mp4 named by the provided filename

Uses imageio. Requires installation of ffmpeg.

@author Quinn Wyner
"""

#Parses user arguments
parser = argparse.ArgumentParser(description='Moving MNIST Explorer')
parser.add_argument('--index', type=int, default=-1, metavar='i',
                    help = 'index from 0 to 9999 in moving mnist dataset from which to generate a video')
parser.add_argument('--filename', type=str, default='', metavar='F',
                    help = 'name of file to which the video should be saved')
args = parser.parse_args()

#Loads moving MNIST dataset
mnist = np.load("../data/mnist_test_seq.npy")

#Saves video as mp4 or raises Exception if invalid index and filename are provided
if(args.index >= 0 and args.index < 10000):
    if(args.filename != ''):
        imageio.mimwrite(args.filename, mnist[:,args.index], fps=10)
    else:
        raise Exception('filename must be defined')
elif(args.filename != ''):
    raise Exception('valid index between 0 and 9999 must be defined')
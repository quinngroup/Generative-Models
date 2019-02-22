import numpy as np
import imageio

"""
Loads the first video data from the moving mnist dataset
Saves the data as an mp4 named 'number.mp4'

Uses imageio. Requires installation of ffmpeg.

@author Quinn Wyner
"""
mnist = np.load("../data/mnist_test_seq.npy")
test = mnist[:,0]

imageio.mimwrite('number.mp4', test, fps=10)
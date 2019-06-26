import cv2
import numpy as np
import time

'''
Transforms Moving MNIST dataset into a new dataset of optical flow images

Generates a numpy array with 5 axes:
0th axis is temporal, representing individual frames of each video
1st axis is observational, representing the choice of video observation
2nd and 3rd axes are spatial, being the length and height for pixels of each frame
4th axis is coordinate, representing the x and y coordinates of the optical flow vector

@author Quinn Wyner
'''

result = np.zeros((10000,19,64,64,2), dtype=np.uint8)
source='../data/mnist_test_seq.npy'
mnist = np.load(source)

for i in range(mnist.shape[1]):
    video = np.zeros((19,64,64,2), dtype=np.uint8)
    prvs = mnist[0,i]
    for j in range(1, mnist.shape[0]):
        next = mnist[j,i]
        video[j - 1] = cv2.normalize(cv2.calcOpticalFlowFarneback(prvs,next, None, 0.8, 6, 15, 10, 5, 1.2, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        prvs = next
    result[i] = video
    if i % 500 == 0:
        print(i)
result = np.swapaxes(result, 0, 1)
np.save('../data/opticalFlow', result)     

#Concatenates the moving mnist numpy array with its optical flow   
mnist = np.load('../data/mnist_test_seq.npy')
array = np.concatenate((np.expand_dims(np.take(mnist,range(1,20),0),4), result), 4)
array1, array2 = np.split(array, 2, 1)
np.save('../data/opfaug1', array1)
np.save('../data/opfaug2', array2)
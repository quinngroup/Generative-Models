import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='transcribe')
parser.add_argument('--source', type=str, default=None, metavar='S',
                    help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
parser.add_argument('--dest', type=str, default=None, metavar='S',
                    help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
args = parser.parse_args()

assert args.source is not None, 'Please provide a source directory'
assert args.source is not None, 'Please provide a destination directory'
for file in os.listdir(args.source):
    print(file)
    np.save(args.dest+'/'+file,np.load(args.source+'/'+file).astype(np.uint8))

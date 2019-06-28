import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

if(__name__=="__main__"):
    sys.path.insert(0,'../')
from vamps.NatVampPrior import VAE
'''
Converts a (duration, input_length, input_length) video to a path through the latent space of a trained VtPVAE appearance head

@param array the (duration, input_length, input_length)-dimensional numpy array video
@param weights the filename of the saved weights for a VtPVAE appearance head
@param pseudos the number of pseudoinputs used by the trained VtPVAE appearance head
@param lsdim the latent space dimensionality of the trained VtPVAE appearance head
@param no_cuda manual option to avoid using cuda
@return a latent-space path encoding of the video

@author Quinn Wyner
'''
def pathMaker(array, weights, input_length, lsdim, pseudos, no_cuda, savefile, plot):
    useCuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")
    
    model = VAE(input_length, lsdim, pseudos, 0, 0, 0, device).to(device)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    input = torch.from_numpy(array.astype(np.float32) / 255.0).to(device)
    
    with torch.no_grad():
        _, _, _, z = model(input)
        z = (z - z[0]).cpu()
    
    if plot and lsdim == 2:
        plt.plot(z[:, 0], data[:, 1])
        plt.show()
    
    np.save(savefile, z)
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pathMaker parser')
    parser.add_argument('--arraySource', type=str, default='../data/mnist_test_seq.npy', metavar='as',
        help='moving mnist source filename (default: \'../data/mnist_test_seq.npy\')')
    parser.add_argument('--load', type=str, default='runs/exp15/lr4.h5', metavar='l',
        help='loads the weights from a given filepath (default: \'runs/exp15/lr4.h5\')')
    parser.add_argument('--lsdim', type = int, default=16, metavar='ld',
        help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA training')
    parser.add_argument('--observation', type=int, default=0, metavar='o',
        help='observation between 0 and 9999 (default: 0)')
    parser.add_argument('--plot', action='store_true', default=False,
        help='enables plotting of latent path (for lsdim 2 only)')
    parser.add_argument('--pseudos', type=int, default=10, metavar='p',
        help='Number of pseudo-inputs (default: 16)')
    parser.add_argument('--savefile', type=str, default='temps/path', metavar='sf',
        help='filename prefix for saved path (default: \'temp/path\')')
    
    args = parser.parse_args()
    array = np.expand_dims(np.load(args.arraySource)[:,args.observation], 1)
    pathMaker(array, args.load, array.shape[2], args.lsdim, args.pseudos, args.no_cuda, args.savefile + str(args.observation), args.plot)
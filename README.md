README

Repository for the JWZ group.

[1](https://arxiv.org/pdf/1606.05908.pdf) A tutorial resource for VAEs emphasizing the mathematical foundation

[2](https://github.com/sksq96/pytorch-summary/tree/master/torchsummary) A python package for summarizing your models similar to the Keras .summary() method. Download the package into your main project wd e.g
```
--Src
  --Models
    myClass.py
    --pytorch-summary-master
```
[3](https://arxiv.org/pdf/1705.07120.pdf) VAE with a modified mixture of variationals prior (VampPrior)

[4](https://github.com/jmtomczak/vae_vampprior) VAE with a modified mixture of variationals prior (VampPrior) github page

[5](https://openreview.net/pdf?id=Sy2fzU9gl) Beta-VAE paper. Also discussing the notions of disentanglement

[6](https://arxiv.org/pdf/1706.02262.pdf) InfoVAE paper covering maximization of mutual information between inputs and intermediate codes a la InfoGAN techniques.

[7](https://arxiv.org/pdf/1701.03077.pdf) "A General and Adaptive Robust Loss Function." Worth a read. Seems like a very interesting dynamic loss function to use. It can be altered and scheduled to smoothly interpolate between various loss functions which occur as special cases of this one.

[8](http://colah.github.io/posts/2015-08-Backprop/) A very good explanation of computational graphs. 

[9](https://github.com/rszeto/moving-symbols) Github repository for generating moving digits. We will use these scripts to generate train/test data for our video based generative models. We will be able to control key independent generative features and hence be able to see if our model *truly* captures critical features.

[10](https://github.com/pytorch/examples/blob/master/vae/main.py) Basic VAE example in PyTorch

[11](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0) Basic intro to Transpose Conv

[12](https://towardsdatascience.com/transpose-convolution-77818e55a123) Basic intro to Transpose Conv

[13](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html) Basic intro to Conv arithmatic of all forms

[14](https://arxiv.org/pdf/1710.04019.pdf) Introduction to data topology

[15](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) LSTM tutorial

[16](https://arxiv.org/pdf/0711.0189.pdf) Spectral clustering tutorial

[17](https://papers.nips.cc/paper/2183-half-lives-of-eigenflows-for-spectral-clustering.pdf) Eigen-cuts algorithm

[18](https://github.com/ncullen93/torchsample/blob/master/README.md) Utility package we ought to implement

[19](https://arxiv.org/pdf/1706.06982.pdf) Two-stream dynamic texture synthesis

[20](https://arxiv.org/pdf/1901.11390.pdf) MONet: Unsupervised scene decomposition and representation

[21](https://arxiv.org/pdf/1901.07017.pdf) Spatial Broadcast Decoder

[22](https://arxiv.org/pdf/1903.00450.pdf) IODINE Network

[23](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_mnistplus.py) Augmented MNIST data set, including texture-in-texture generation

[24](http://legacydirs.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf) Further explanation for VAE objective functino, ELBO and loss


Descriptions of Uploads:

CNN_T.py  - Uploaded base code from Meekail on a basic Convolution Net using the MNIST Data set

neg-inf-loss-model.py - Early attempt at reproducing CNN from CNN_T.py, resulting in a negative ifinite loss.

testNet2.py - Successful CNN using the MNIST data set

vaeTest1.py - First attempt at implementing a Variational Auto Encoder on  a very basic neural net, using the MNIST data.

vaeTest2.py - First SUCCESSFUL attempt at implementing a Variational Auto Encoder on a very basic neural net, using the MNIST data.

vae4LayerConv.py - A Variational Auto Encoder with 4 hidden convolutional layers in the encoder and 1 hidden fully-connected layer in 
the decoder, as well as graphical embedding after 10 epochs

vae10LayerConv.py - A Variational Auto Encoder with 10 hidden convolutional layers in the encoder and 1 hidden fully-connected layer in the decoder, as well as graphical embedding after 10 epochs



1/23/19: Covered basic VAE [3], and beta-vae [5], started implementation of a simple FF-VAE. Discussed goals for next month.
  February Goals: Establish potential model architectures, run initial tests.
  
1/30/19: Uploaded vaeTest2.py, the first successful Feet-Forward Variational Auto Encoder. Finished vae4LayerConv.py and vae10LayerConv.py, 2 Variational Auto Encoders with 4 and 10 hidden convolutional layers in their encoders, respectively. Added graphical embedding functionality to vae4LayerConv.py and vae10LayerConv.py to visualize the constructed latent spaces after 10 epochs. Began attempting to determine how to add legends to the scatter plots used to present the graphical embeddings.

2/6/19: Uploaded 10Conv4TConv.py, the first tutorial model to implement transpose convolutional layers in the decoder. Uploaded more graphical embeddings. Updated models with colorbars to display which color corresponds to which number in the graphical embeddings.

2/13/19: Uploaded 4Conv4TConv.py, which uses 4 convolutional layers instead of 10 in the encoder to increase training rate with minimal change in final result. Uploaded 4C4TTSNE.py, which uses TSNE projection to display a 2-dimensional projection of the graphical embedding of higher-dimensional latent spaces.

2/22/19: Uploaded mnist_test_seq.npy, a numpy array containing 10,000 observations of 20-frame 64x64 videos of moving handwritten digits, called hereafter by the moving MNIST dataset. Uploaded movingMNISTExploration.py, which allows conversion between an observation of the moving MNIST dataset and an mp4 file and constructs training and testing DataLoaders from the moving MNIST dataset.

3/20/19: Began work on a Long Short-Term Memory Variational Autoencoder for the moving MNIST dataset. Constructed a bare-bones cell state stream and began a new encoder to encode 64x64 images. Updated movingMNISTExplorer.py to allow generation of DataLoaders from outside programs. Deleted a redundant model. Reviewed research regarding Two-Stream and VampPrior.

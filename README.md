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

[25](https://arxiv.org/pdf/1712.00636.pdf) Paper on using compressed representation of video data to truncate noise from low-frequency motion

[26](https://arxiv.org/pdf/1807.04689.pdf) Paper exploring the idea and implementation of general homemorphic manifolds extending the standard guassian prior

[27](https://arxiv.org/pdf/1703.06114.pdf) Paper creating an model which operates on clustering/classifying sets as objects rather than vectors

[28](https://arxiv.org/pdf/1702.08389.pdf) Paper on the equivariance of models (through parameter sharing) using resistance to group actions

[29](https://arxiv.org/pdf/1901.06082.pdf) "PROBABILISTIC SYMMETRY AND INVARIANT NEURAL
NETWORKS" 

[30](https://arxiv.org/pdf/1703.06211.pdf) "Deformable Convolutional Networks"

[31](https://arxiv.org/pdf/1609.04836.pdf) "ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA"

[32](http://www.cs.columbia.edu/~gravano/Papers/2017/tods17.pdf) Paper exploring (fast) time-series clustering

[33](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8600380) B3DF paper

[34](https://arxiv.org/pdf/1901.11390.pdf) MonNet architecture, which is a good look at iterative refinmenet in models, along with a great introduction to the utility of masks

[35](https://arxiv.org/pdf/1903.00450.pdf) IODINE network architecture, which is an evolutoin of 

[36](https://arxiv.org/pdf/1906.00446.pdf) High-fidelity reconstruction using discretized encoding vectors

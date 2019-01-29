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



Descriptions of Uploads:

CNN_T.py  - Uploaded base code from Meekail on a basic Convolution Net using the MNIST Data set
neg-inf-loss-model.py - Early attempt at reproducing CNN from CNN_T.py, resulting in a negative ifinite loss.
testNet2.py - Successful CNN using the MNIST data set
vaeTest1.py - First attempt at implementing a Variational Auto Encoder on  a very basic neural net, using the MNIST data.


1/23/19: Covered basic VAE [3], and beta-vae [5], started implementation of a simple FF-VAE. Discussed goals for next month.
  February Goals: Establish potential model architectures, run initial tests.

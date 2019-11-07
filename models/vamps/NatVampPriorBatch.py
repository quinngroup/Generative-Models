CHECKSUM = 'BatchVamp1'

import argparse
import torch
import time
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.manifold import TSNE


import sys,os
#print(os.getcwd())
#print(os.listdir())

if(__name__=="__main__"):
    sys.path.insert(0,'../../')

#print(os.listdir())
#print(os.getcwd())
from utils.nn import spatial_broadcast_decoder
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
    
    
    
"""
VampPrior implementation with BatchNormalization

@author Quinn Wyner
"""
    
    
class VAE(nn.Module):
    def __init__(self, input_length, lsdim, device):
        super(VAE, self).__init__()
                    
        self.input_length = input_length
        self.lsdim = lsdim
        self.device = device
        self.finalConvLength = ((input_length - 2)//2 - 1)//2 - 4

        #(1,28,28) -> (8,26,26)
        self.conv1 = nn.Conv2d(1, 8, 3)
             
        #(8,13,13) -> (16,12,12)
        self.conv2 = nn.Conv2d(8, 16, 2)
        
        #(16,12,12) -> (16,12,12)
        self.batch1 = nn.BatchNorm2d(16, track_running_stats = False)
        
        #(16,6,6) -> (32,4,4)
        self.conv3 = nn.Conv2d(16, 32, 3)
        
        #(32,4,4) -> (64,2,2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        
        #(64,2,2) -> (64,2,2)
        self.batch2 = nn.BatchNorm2d(64, track_running_stats = False)

        #(80,4,4) -> lsdim mean and logvar
        self.mean = nn.Linear(64*self.finalConvLength*self.finalConvLength, lsdim)
        self.logvar = nn.Linear(64*self.finalConvLength*self.finalConvLength, lsdim)

        self.sbd=spatial_broadcast_decoder(input_length=self.input_length,device=self.device,lsdim=self.lsdim)
        # create an idle input for calling pseudo-inputs


    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean
    
    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
    
        #(1,28,28) -> (8,26,26) -> (8,13,13)
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), (2,2)))
        
        #(8,13,13) -> (16,12,12) -> (16,6,6)
        x = F.leaky_relu(F.max_pool2d(self.batch1(self.conv2(x)), (2,2)))
        
        #(16,6,6) -> (32,4,4)
        x = F.leaky_relu(self.conv3(x))
        
        #(32,4,4) -> (64,2,2)
        x = F.leaky_relu(self.batch2(self.conv4(x)))
        x=x.view(-1, 64*self.finalConvLength*self.finalConvLength)
        
        z_q_mean = self.mean(x)
        z_q_logvar = F.elu(self.logvar(x), 3.0)
        return z_q_mean, z_q_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # p(x|z)
    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        
        return self.sbd(z)
            
    
    def forward(self, x):

        #z~q(z|x)
        mu, logvar = self.q_z(x)
        z=self.reparameterize(mu,logvar)

        x_mean=self.p_x(z)
        #decode code
        return x_mean, mu, logvar, z

    
    
        
 
 
class PseudoGen(nn.Module):
    def __init__(self, input_length, pseudos,device):
        super(PseudoGen, self).__init__()
        
        self.means = nn.Linear(pseudos, input_length*input_length, bias=False)
        self.idle_input = torch.eye(pseudos,pseudos,requires_grad=True)
        self.idle_input = self.idle_input.to(device)

    def forward(self, x):
        return torch.sigmoid(self.means(x))
        

class NatVampPrior(nn.Module):
    def __init__(self, input_length, lsdim, pseudos, beta, gamma, device):
        super(NatVampPrior, self).__init__()
        
        self.pseudos = pseudos
        self.beta = beta
        self.gamma = gamma
        self.input_length=input_length
        
        self.vae = VAE(input_length, lsdim, device)
        self.pseudoGen = PseudoGen(input_length, pseudos,device)
        
        self.idle_input = torch.eye(pseudos, pseudos, requires_grad=True).cuda()

    def forward(self, x):
        return self.vae.forward(x)
  
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, z_q, pseudo,recon_pseudo, p_mu, p_logvar, p_z, gamma=None):
        RE = F.mse_loss(recon_x.view(-1,self.input_length*self.input_length), x.view(-1, self.input_length*self.input_length), reduction = 'sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = torch.sum(log_Normal_diag(z_q, mu, logvar, dim=1),0)
        KL = -(log_p_z - log_q_z)

        #pseudo-loss
        pRE = F.mse_loss(recon_pseudo.view(-1,self.input_length*self.input_length), pseudo.view(-1, self.input_length*self.input_length), reduction = 'sum')

        plog_p_z = self.log_p_z(p_z)
        plog_q_z = torch.sum(log_Normal_diag(p_z, p_mu, p_logvar, dim=1),0)
        pKL= -(plog_p_z - plog_q_z)

        if gamma is None:
            return (RE + self.beta*KL)+self.gamma*(pRE + self.beta*pKL)
        else:
            return (RE + self.beta*KL)+gamma*(pRE + self.beta*pKL)

    def log_p_z(self,z):
        # calculate params
        X = self.pseudoGen.forward(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.vae.q_z(X.view(-1,1,self.input_length,self.input_length))  # C x M

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)
        
        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(self.pseudos)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculate log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return torch.sum(log_prior, 0)



def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    #print(log_var)
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )

    if average:

        return torch.mean( log_normal, dim )

    else:

        return torch.sum( log_normal, dim )

if __name__ == "__main__":

    startTime = time.time()
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--testSplit', type=float, default=.05, metavar='%',
                        help='portion of dataset to test on (default: .2)')
    parser.add_argument('--source', type=str, default='../data/mnist_test_seq.npy', metavar='S',
                        help = 'path to moving MNIST dataset (default: \'../data/mnist_test_seq.npy\')')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=str, default='', metavar='s',
                        help='saves the weights to a given filepath')
    parser.add_argument('--load', type=str, default='', metavar='l',
                        help='loads the weights from a given filepath')
    parser.add_argument('--beta', type=float, default=1.0, metavar='b',
                        help='sets the value of beta for a beta-vae implementation')
    parser.add_argument('--pseudos', type=int, default=10, metavar='p',
                        help='Number of pseudo-inputs (default: 10)')
    parser.add_argument('--lsdim', type = int, default=2, metavar='ld',
                        help='sets the number of dimensions in the latent space. should be >1. If  <3, will generate graphical representation of latent without TSNE projection')
                        #current implementation may not be optimal for dims above 4
    parser.add_argument('--gamma', type = float, default=10, metavar='g',
                        help='Pseudo-loss weight')
    parser.add_argument('--lr', type = float, default=1e-3, metavar='lr',
                        help='learning rate')  
    parser.add_argument('--graph', action='store_true', default= False,
                        help='flag to determine whether or not to run automatic graphing')      
    parser.add_argument('--input_length', type=int, default=28, metavar='il',
                        help='length and height of one image')
    parser.add_argument('--repeat', action='store_true', default=False,
                        help='determines whether to enact further training after loading weights')
    parser.add_argument('--plr', type = float, default = 4e-6, metavar='plr',
                        help='pseudoinput learning rate')
    parser.add_argument('--reg2', type = float, default=0, metavar='rg2',
                        help='coefficient for L2 weight decay')
    parser.add_argument('--noEarlyStop', action='store_true', default=False,
                        help='disables early stopping')
    parser.add_argument('--tolerance', type = float, default=.1, metavar='tol',
                        help='tolerance value for early stopping')
    parser.add_argument('--patience', type = int, default = 10, metavar='pat',
                        help='patience value for early stopping')
    parser.add_argument('--failCount', type=str, default='r', metavar='fc',
                        help='determines how to reset early-stopping failed epoch counter. Options are \'r\' for reset and \'c\' for cumulative')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    device = "cuda" if args.cuda else "cpu"
    
    if(args.cuda):
        with torch.cuda.device(0):
            torch.tensor([1.]).cuda()

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/', train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data/', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
        
    model = NatVampPrior(args.input_length, args.lsdim, args.pseudos, args.beta, args.gamma, device).to(device)
    optimizer = optim.Adam([{'params': model.vae.parameters()},
                            {'params': model.pseudoGen.parameters(), 'lr': args.plr}],
                            lr=args.lr, weight_decay=args.reg2)
    stopEarly = False
    failedEpochs=0
    lastLoss = 0            
    
    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            #print()
            #for param in model.parameters():
            #    print(torch.norm(param))
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            pseudos=model.pseudoGen.forward(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
            recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
            loss = model.loss_function(recon_batch, data, mu, logvar, z,pseudos,recon_pseudos, p_mu, p_logvar, p_z)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
            
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
     

    def test(epoch, max, startTime):
        model.eval()
        test_loss = 0
        global lastLoss
        global failedEpochs
        global stopEarly
        zTensor = torch.empty(0,args.lsdim).to(device)
        labelTensor = torch.empty(0, dtype = torch.long)
        pseudos=model.pseudoGen.forward(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)

                test_loss += model.loss_function(recon_batch, data, mu, logvar,z,pseudos,recon_pseudos, p_mu, p_logvar, p_z).item()
                zTensor = torch.cat((zTensor, z), 0)
                labelTensor = torch.cat((labelTensor, _), 0)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        if(epoch == 1):
            lastLoss = test_loss
        elif not args.noEarlyStop:
            if lastLoss-test_loss < args.tolerance:
                failedEpochs += 1
                if failedEpochs >= args.patience:
                    stopEarly = True
            elif args.failCount == 'r':
                failedEpochs = 0
        if(epoch == max):
            print("--- %s seconds ---" % (time.time() - startTime))
            cmap = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe'])
            
            #Handling different dimensionalities
            if(args.graph):
                if (args.lsdim < 3) :
                    z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                    z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                    scatterPlot = plt.scatter(z1, z2, s = 4, c = labelTensor, cmap = cmap) #Regular 2dim plot, RE-ADD CMAP = CMAP
                    plt.colorbar()
                elif (args.lsdim == 3) :
                    fig=plt.figure()
                    ax=fig.gca(projection='3d')
                    z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                    z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                    z3 = torch.Tensor.cpu(zTensor[:, 2]).numpy()
                    scatterPlot = ax.scatter(z1, z2, z3, s = 4, c = labelTensor, cmap = cmap) #Regular 3dim plot
                else:    
                    Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())        
                    z1 = Z_embedded[:, 0]
                    z2 = Z_embedded[:, 1]
                    scatterPlot = plt.scatter(z1, z2, s = 4, c = labelTensor, cmap = cmap) #TSNE projection for >3dim 
                    plt.colorbar()

                plt.show()
            temp = model.pseudoGen.forward(model.idle_input).view(-1,args.input_length,args.input_length).detach().cpu()
            for x in range(args.pseudos):
                plt.matshow(temp[x].numpy())
                plt.show()
            
            
    def dplot(x):
        img = p_x(x)
        plt.imshow(img)
    print(device)
    summary(model,(1,args.input_length,args.input_length),device=device)
    if(args.load == ''):
        for epoch in range(1, args.epochs + 1):
            if(not stopEarly):
                train(epoch)
                test(epoch, args.epochs, startTime)
    else:
        model.load_state_dict(torch.load(args.load))
        test(args.epochs, args.epochs, startTime)
        if(args.repeat==True):
            for epoch in range(1, args.epochs + 1):
                train(epoch)
                test(epoch, args.epochs, startTime)
    if(args.save != ''):
        torch.save(model.state_dict(), CHECKSUM + args.save)

        
    

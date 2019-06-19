from __future__ import print_function
import argparse
import torch
import torch.utils.data
import time
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import LeakyReLU
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchsummary import summary
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage


import sys,os
#print(os.getcwd())
#print(os.listdir())

if(__name__=="__main__"):
    sys.path.insert(0,'../../')

#print(os.listdir())
#print(os.getcwd())
from utils.nn import spatial_broadcast_decoder
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.colors as colors
    
    
    
"""
Ignite implementation of NatVampPrior

@author Davis Jackson, Quinn Wyner
"""
    
    
class VAE(nn.Module):
    def __init__(self, input_length, lsdim, pseudos, beta, gamma, batch_size, device):
        super(VAE, self).__init__()
                    
        self.input_length = input_length
        self.lsdim = lsdim
        self.pseudos = pseudos
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.finalConvLength = ((input_length - 2)//2 - 1)//2 - 4

        #(1,28,28) -> (8,26,26)
        self.conv1 = nn.Conv2d(1, 8, 3)
        
        #(8,13,13) -> (16,12,12)
        self.conv2 = nn.Conv2d(8, 16, 2)
        
        #(16,6,6) -> (32,4,4)
        self.conv3 = nn.Conv2d(16, 32, 3)
        
        #(32,4,4) -> (64,2,2)
        self.conv4 = nn.Conv2d(32, 64, 3)

        #(80,4,4) -> lsdim mean and logvar
        self.mean = nn.Linear(64*self.finalConvLength*self.finalConvLength, lsdim)
        self.logvar = nn.Linear(64*self.finalConvLength*self.finalConvLength, lsdim)

        self.means = nn.Linear(pseudos, input_length * input_length, bias=False)
        self.sbd=spatial_broadcast_decoder(input_length=self.input_length,device=self.device,lsdim=self.lsdim)
        # create an idle input for calling pseudo-inputs
        self.idle_input = torch.eye(pseudos,pseudos,requires_grad=True)
        self.idle_input = self.idle_input.to(device)


    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean
    
    # ADDITIONAL METHODS
    def generate_x(self, N=None):
        if N is None:
            N = self.pseudos
        means = self.means(self.idle_input)[0:N]
        z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
        z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand
        
    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
    
        #(1,28,28) -> (8,26,26) -> (8,13,13)
        x = F.max_pool2d(LeakyReLU(0.1)(self.conv1(x)), (2,2))
        
        #(8,13,13) -> (16,12,12) -> (16,6,6)
        x = F.max_pool2d(LeakyReLU(0.1)(self.conv2(x)), (2,2))
        
        #(16,6,6) -> (32,4,4)
        x = LeakyReLU(0.1)(self.conv3(x))
        
        #(32,4,4) -> (64,2,2)
        x = LeakyReLU(0.1)(self.conv4(x))
        x=x.view(-1, 64*self.finalConvLength*self.finalConvLength)
        
        z_q_mean = self.mean(x)
        z_q_logvar = self.logvar(x)
        return z_q_mean, z_q_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # p(x|z)
    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        
        return self.sbd(z)
            
    def log_p_z(self,z):
        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.q_z(X.view(-1,1,self.input_length,self.input_length))  # C x M

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(self.pseudos)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculate log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return torch.sum(log_prior, 0)
    
    def forward(self, x):

        #z~q(z|x)
        mu, logvar = self.q_z(x)
        z=self.reparameterize(mu,logvar)

        x_mean=self.p_x(z)
        #decode code
        return x_mean, mu, logvar, z

    
    
        
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
        #return (RE + self.beta*KL)+self.gamma*(pRE + self.beta*pKL)/self.batch_size
     
def log_Normal_diag(x, mean, log_var, average=False, dim=None):

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
    parser.add_argument('--dbscan', action='store_true', default= False,
                        help='to run dbscan clustering')      
    parser.add_argument('--graph', action='store_true', default= False,
                        help='flag to determine whether or not to run automatic graphing')      
    parser.add_argument('--input_length', type=int, default=28, metavar='il',
                        help='length and height of one image')
    parser.add_argument('--repeat', action='store_true', default=False,
                        help='determines whether to enact further training after loading weights')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    device = "cuda" if args.cuda else "cpu"
    
    if(args.cuda):
        with torch.cuda.device(0):
            torch.tensor([1.]).cuda()

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    train_data = MNIST(download=False, root="../../data/", transform=data_transform, train=True)
    val_data = MNIST(download=False, root="../../data/", transform=data_transform, train=False)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        
    model = VAE(args.input_length, args.lsdim, args.pseudos, args.beta, args.gamma, args.batch_size, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)            
            
    def train(engine, batch):
        model.train()

        data, _ = batch 
        data = data.to(device) #change to data = batch maybe???
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        pseudos=model.means(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        loss = model.loss_function(recon_batch, data, mu, logvar, z,pseudos,recon_pseudos, p_mu, p_logvar, p_z)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            genLoss=model.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma=0)
        return loss.item(), genLoss.item()
     

    def test(engine, batch):
        model.eval()
        with torch.no_grad():
            x, _ = batch
            x = x.to(device)
            recon_x, mu, logvar, z = model(x)
            pseudos=model.means(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
            recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
            kwargs = {'mu': mu, 'logvar': logvar, 'z_q': z, 'pseudo': pseudos, 'recon_pseudo': recon_pseudos, 'p_mu': p_mu, 'p_logvar': p_logvar, 'p_z': p_z}
        return recon_x, x, kwargs
        
    trainer = Engine(train)
    evaluator = Engine(test)
    training_history = {'loss': [], 'genLoss': []}
    validation_history = {'loss': [], 'genLoss': []}
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'genLoss')
    
    Loss(model.loss_function, lambda x: [x[0], x[1], x[2]]).attach(evaluator, 'loss')
    Loss(model.loss_function, lambda x: [x[0], x[1], x[2]]).attach(evaluator, 'genLoss')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_trainer_logs(engine):
        avg_loss = engine.state.metrics['loss']
        avg_genLoss = engine.state.metrics['genLoss']
        print("Trainer Results - Epoch {} - Avg loss: {:.6f} Avg genLoss:{:.6f}"
              .format(engine.state.epoch, avg_loss, avg_genLoss))
    
    def print_logs(engine, dataloader, mode, history_dict):
        evaluator.run(dataloader, max_epochs=1)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        avg_genLoss = metrics['genLoss']
        print(
            mode + " Results - Epoch {} - Avg Loss: {:.6f} Avg genLoss: {:.6f}"
            .format(engine.state.epoch, avg_loss, avg_genLoss))
        for key in evaluator.state.metrics.keys():
            history_dict[key].append(evaluator.state.metrics[key])
    trainer.add_event_handler(Events.ITERATION_COMPLETED, print_logs, train_loader, 'Training', training_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training', training_history)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, val_loader, 'Validation', validation_history)

    def dplot(x):
        img = p_x(x)
        plt.imshow(img)

    summary(model,(1,args.input_length,args.input_length),device=device)
    if(args.load == ''):
        e = trainer.run(train_loader, max_epochs=2)
    else:
        model.load_state_dict(torch.load(args.load))
        test(args.epochs, args.epochs, startTime)
        if(args.repeat==True):
            for epoch in range(1, args.epochs + 1):
                train(epoch)
                test(epoch, args.epochs, startTime)
    if(args.save != ''):
        torch.save(model.state_dict(), args.save)

        
    

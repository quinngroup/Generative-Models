import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import time
import torch
from torchsummary import summary
from torchvision import datasets, transforms

if(__name__=="__main__"):
    sys.path.insert(0,'../../')

from NatVampPrior import NatVampPrior, log_Normal_diag 
    
"""
VampPrior implementation with adjusted loss function to normalize the prior with a standard unit Gaussian

@author Quinn Wyner
"""

def newLoss(model, recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma = None, delta1=0, delta2=0):
    oldLoss = model.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z)
    newKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    p_newKL = -0.5 * torch.sum(1 + p_logvar - p_mu.pow(2) - p_logvar.exp())
    if gamma:
        return oldLoss + delta1*newKL + gamma*delta2*newKL
    else:
        return oldLoss + delta1*newKL + model.gamma*delta2*newKL
        
if __name__ == "__main__":

    startTime = time.time()
    parser = argparse.ArgumentParser(description='Normalized Implementation Arguments')
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
    parser.add_argument('--delta1', type = float, default = 1.0, metavar='d1',
                        help='Sets the weight of KL divergence of posterior against standard unit Gaussian in loss function')
    parser.add_argument('--delta2', type = float, default = 1.0, metavar='d2',
                        help='Sets the weight of KL divergence of pseudoinput posterior against standard unit Gaussian in loss function')

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
    optimizer = torch.optim.Adam([{'params': model.vae.parameters()},
                            {'params': model.pseudoGen.parameters(), 'lr': args.plr}],
                            lr=args.lr, weight_decay=args.reg2)
            
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
            loss = newLoss(model, recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, delta1=args.delta1, delta2=args.delta2)
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
        zTensor = torch.empty(0,args.lsdim).to(device)
        labelTensor = torch.empty(0, dtype = torch.long)
        pseudos=model.pseudoGen.forward(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)
                loss = newLoss(model, recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, delta1=args.delta1, delta2=args.delta2)
                test_loss += loss.item()
                zTensor = torch.cat((zTensor, z), 0)
                labelTensor = torch.cat((labelTensor, _), 0)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
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
        torch.save(model.state_dict(), args.save)
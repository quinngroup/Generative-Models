CHECKSUM = 'VtP-A2'

import argparse
import mlflow
import torch
import time
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mMNISTflat import genLoaders

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys,os

#os.chdir(os.path.dirname(sys.argv[0]))
sys.path.insert(0,'../')
#print(os.getcwd())

#print(os.listdir())
from vamps.NatVampPrior import NatVampPrior




'''
Appearance head for Video-to-Path model
Implements NatVampPrior on frames of a video dataset

@author Davis Jackson, Quinn Wyner
'''

startTime = time.time()
parser = argparse.ArgumentParser(description='VtPVAE')
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
parser.add_argument('--gamma', type = float, default=.05, metavar='g',
                    help='Pseudo-loss weight')
parser.add_argument('--lr', type = float, default=1e-3, metavar='lr',
                    help='learning rate')
parser.add_argument('--plr', type = float, default=4e-6, metavar='lr',
                    help='learning rate')
parser.add_argument('--dbscan', action='store_true', default= False,
                    help='to run dbscan clustering')      
parser.add_argument('--graph', action='store_true', default= False,
                    help='flag to determine whether or not to run automatic graphing')      
parser.add_argument('--input_length', type=int, default=64, metavar='il',
                    help='length and height of one image')
parser.add_argument('--repeat', action='store_true', default=False,
                    help='determines whether to enact further training after loading weights')
parser.add_argument('--pp', type = int, default=0, metavar='pp',
                    help='Plot pseudos. Controls the number of pseudo inputs to be displayed')
parser.add_argument('--log', type=str, default='!', metavar='lg',
                    help='flag to determine whether to use tensorboard for logging. Default \'!\' is read to mean no logging')      
parser.add_argument('--schedule', type = int, default=-1, metavar='sp',
                    help='use learning rate scheduler on loss stagnation with input patience')
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
parser.add_argument('--experiment', type=str, default=None,
                    help='Name of experiment being run. Default = \'\'')
parser.add_argument('--runName', type=str, default=None,
                    help='Name of run to be logged. Default = \'\'')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = "cuda" if args.cuda else "cpu"

if(args.cuda):
    with torch.cuda.device(0):
        torch.tensor([1.]).cuda()
        
if args.experiment:
    mlflow.set_experiment(args.experiment)
    if args.runName:
        mlflow.start_run(run_name = args.runName)
    else:
        mlflow.start_run()
    for arg in vars(args):
        mlflow.log_param(arg, getattr(args, arg))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#Loads moving MNIST dataset
mnist = np.load(args.source,allow_pickle=True)

writer=None
if(args.log!='!'):
    if(args.log=='$'):
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir='runs/'+args.log)


#movingMNISTDataset class    
class movingMNISTDataset(Dataset):
    """
    Initializes dataset
    @param npArray (numpy.array): moving MNIST dataset
    @param transform(callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, npArray, transform=None):
        self.npArray = npArray
        self.frameCount = (self.npArray.shape)[0]
        self.vidCount = (self.npArray.shape)[1]
        self.transform = transform
       
    """
    Gets number of observations in dataset
    @return number of observations
    """
    def __len__(self):
        return self.frameCount * self.vidCount
    
    """
    Gets the observation at a given index
    @param index (int): index corresponding to observation which is to be returned
    @return Tensor observation corresponding to given index
    """
    def __getitem__(self, index):
        obs = self.npArray[index % self.frameCount, index // self.frameCount,:,:,np.newaxis]
        if self.transform:
            obs = self.transform(obs)
        return obs

#Constructs Pytorch Dataset from moving MNIST data
data = movingMNISTDataset(npArray=mnist, transform=transforms.ToTensor())
train_loader, test_loader = genLoaders(data, args.batch_size, args.no_cuda, args.seed, args.testSplit)
    

model = NatVampPrior(args.batch_size, args.input_length, args.lsdim, args.pseudos, args.beta, args.gamma, device).to(device)
optimizer = optim.Adam([{'params': model.vae.parameters()},
                        {'params': model.pseudoGen.parameters(), 'lr': args.plr}],
                        lr=args.lr, weight_decay=args.reg2)
stopEarly = False
failedEpochs=0
lastLoss = 0

scheduler=None
if(args.schedule>0):
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=args.schedule)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        pseudos=model.pseudoGen.forward(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
        recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
        loss = model.loss_function(recon_batch, data, mu, logvar, z,pseudos,recon_pseudos, p_mu, p_logvar, p_z)
        loss.backward()
        train_loss += loss.item()
        genLoss = model.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma=0).item() / len(data)
        if args.experiment:
            mlflow.log_metric('trainLoss', loss.item()/len(data))
            mlflow.log_metric('genLoss', genLoss)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGenLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                genLoss))
        step=epoch*len(train_loader)+batch_idx
        if(args.log!='!'):
            per_item_loss=loss.item()/len(data)
            writer.add_scalar('item_loss',per_item_loss,global_step=step)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    if(args.schedule>0):
          scheduler.step(train_loss / len(train_loader.dataset))

def test(epoch, max, startTime):
    model.eval()
    test_loss = 0
    gen_loss = 0
    global lastLoss
    global failedEpochs
    global stopEarly
    zTensor = torch.empty(0,args.lsdim).to(device)
    pseudos=model.pseudoGen.forward(model.idle_input).view(-1,1,args.input_length,args.input_length).to(device)
    recon_pseudos, p_mu, p_logvar, p_z=model(pseudos)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += model.loss_function(recon_batch, data, mu, logvar,z,pseudos,recon_pseudos, p_mu, p_logvar, p_z).item()
            gen_loss += model.loss_function(recon_batch, data, mu, logvar, z, pseudos, recon_pseudos, p_mu, p_logvar, p_z, gamma=0).item()
            zTensor = torch.cat((zTensor, z), 0)
    if (args.dbscan == True) :
        zScaled = StandardScaler().fit_transform((torch.Tensor.cpu(zTensor).numpy())) #re-add StandardScaler().fit_transform
        db = DBSCAN(eps= 0.7, min_samples= 3).fit(zScaled)
        print(db.labels_)
        labelTensor = db.labels_
    test_loss /= len(test_loader.dataset)
    gen_loss /= len(test_loader.dataset)
    if args.experiment:
        mlflow.log_metric('testLoss', test_loss)
        mlflow.log_metric('testGenLoss', gen_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Generation loss: {:.4f}'.format(gen_loss))
    if(epoch == 1):
        lastLoss = test_loss
    elif not args.noEarlyStop:
        if lastLoss-test_loss < args.tolerance:
            failedEpochs += 1
            if failedEpochs >= args.patience:
                stopEarly = True
                epoch = max
        elif args.failCount == 'r':
            failedEpochs = 0
    if(epoch == max):
        if(args.save != ''):
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()
                        }, args.save)
        print("--- %s seconds ---" % (time.time() - startTime))
        cmap = colors.ListedColormap(['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe'])
        
        #Handling different dimensionalities
        if(args.graph):
            if (args.lsdim < 3) :
                z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                scatterPlot = plt.scatter(z1, z2, s = 4) #Regular 2dim plot, RE-ADD CMAP = CMAP
            elif (args.lsdim == 3) :
                fig=plt.figure()
                ax=fig.gca(projection='3d')
                z1 = torch.Tensor.cpu(zTensor[:, 0]).numpy()
                z2 = torch.Tensor.cpu(zTensor[:, 1]).numpy()
                z3 = torch.Tensor.cpu(zTensor[:, 2]).numpy()
                scatterPlot = ax.scatter(z1, z2, z3, s = 4) #Regular 3dim plot
            else:    
                Z_embedded = TSNE(n_components=2, verbose=1).fit_transform(zTensor.cpu())        
                z1 = Z_embedded[:, 0]
                z2 = Z_embedded[:, 1]
                scatterPlot = plt.scatter(z1, z2, s = 4) #TSNE projection for >3dim 

            plt.show()
        if(args.pp>0):
            t=min(args.pp,args.pseudos)
            temp = model.means(model.idle_input).view(-1,args.input_length,args.input_length).detach().cpu()
            for x in range(t):
                plt.matshow(temp[x].numpy())
                plt.show()
      
def dplot(x):
    img = decode(x)
    plt.imshow(img)

if __name__ == "__main__":
    summary(model,(1,args.input_length,args.input_length))
    '''
    if(args.load == ''):
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch, args.epochs, startTime)
    elif(args.cuda == True):
        model.load_state_dict(torch.load(args.load))
        test(args.epochs, args.epochs, startTime)
    else:
        model.load_state_dict(torch.load(args.load, map_location= device))
        test(args.epochs, args.epochs, startTime)
    '''
    if(args.load == ''):
        for epoch in range(1, args.epochs + 1):
            if(not stopEarly):
                train(epoch)
                test(epoch, args.epochs, startTime)
    else:
        checkpoint=torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        test(args.epochs, args.epochs, startTime)
        if(args.repeat==True):
            for epoch in range(1, args.epochs + 1):
                train(epoch)
                test(epoch, args.epochs, startTime)
                
    if(args.log!='!'):
        #res = torch.autograd.Variable(torch.Tensor(1,1,20,64,64), requires_grad=True).to(device)
        #writer.add_graph(model,res,verbose=True)
        writer.close()
    if args.experiment:
        mlflow.end_run()
    

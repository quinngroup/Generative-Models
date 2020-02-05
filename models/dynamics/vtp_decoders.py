import torch
from torch import nn
import torch.nn.functional as f
from torchdiffeq import odeint
from torchsummary import summary

APP_LSD = 2
DYN_LSD = 2
PROC_PATH_LENGTH = 19

'''
Number of paramaters for decoder (using torch.summary)
    Sequential: 110,108
    Hidden State: 33,484
    Recurrent: 51,194

Variable Conventions:
    h: hidden state
    t: time
    w: something in dynamics latent space
    z: something in appearance latent space
'''


class ObservationModule(nn.Module):
    def __init__(self):
        super(ObservationModule, self).__init__()
        self.lin1 = nn.Linear(2 * APP_LSD, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, APP_LSD)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = torch.sigmoid(self.lin3(h))
        return h

class Sequential(nn.Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.lin1 = nn.Linear(DYN_LSD, APP_LSD)
        self.mods = nn.ModuleList([ObservationModule() for _ in range(PROC_PATH_LENGTH)])
    
    def forward(self, w):
        z_1 = torch.sigmoid(self.lin1(w))
        recon_path_list = [z_1]
        for module in self.mods:
            module_input = torch.cat((w, recon_path_list[-1]), dim=1)
            new_z = module(module_input)
            recon_path_list.append(new_z)
        recon_path = torch.stack(recon_path_list, dim=1)
        return recon_path


class HiddenState(nn.Module):
    def __init__(self):
        super(HiddenState, self).__init__()
        
        self.lin1 = nn.Linear(DYN_LSD, 100)
        self.lin2 = nn.Linear(100, 150)
        self.lin3 = nn.Linear(150, 80)
        self.lin4 = nn.Linear(80, PROC_PATH_LENGTH * APP_LSD)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = f.leaky_relu(self.lin3(h))
        h = torch.sigmoid(self.lin4(h))        
        h = h.view((-1, PROC_PATH_LENGTH, APP_LSD))
        return h
        

class Recurrent(nn.Module):
    def __init__(self):
        super(Recurrent, self).__init__()
        
        self.h_to_h_1 = nn.Linear(DYN_LSD, 50)
        self.h_to_h_2 = nn.Linear(50, 20)
        self.h_to_h_3 = nn.Linear(20, DYN_LSD)
        
        self.h_to_z_1 = nn.Linear(DYN_LSD, 50)
        self.h_to_z_2 = nn.Linear(50, 20)
        self.h_to_z_3 = nn.Linear(20, APP_LSD)    
        
    def forward(self, w):
        h = w
        points = []        
        for _ in range(PROC_PATH_LENGTH):
            h = f.leaky_relu(self.h_to_h_1(h))
            h = f.leaky_relu(self.h_to_h_2(h))
            h = f.leaky_relu(self.h_to_h_3(h))
            
            z = torch.sigmoid(self.h_to_z_1(h))
            z = torch.sigmoid(self.h_to_z_2(z))
            z = torch.sigmoid(self.h_to_z_3(z))
            
            points.append(z)
            
        recon_path = torch.stack(points, dim=1)
        return recon_path
            

class Integrand(nn.Module):
    def __init__(self):
        super(Integrand, self).__init__()
        
        self.lin1 = nn.Linear(APP_LSD + 1, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, APP_LSD)
    
    def forward(self, z, t):
        print(z.size(), t.size())
        input = torch.stack((z, t))
        out = f.leaky_relu(self.lin1(input))
        out = f.leaky_relu(self.lin2(out))
        out = torch.sigmoid(self.lin3(out))
        
        return out


class NeuralIntegrator(nn.Module):
    def __init__(self):
        super(NeuralIntegrator, self).__init__()
        
        self.integrand = Integrand()
        self.lin1 = nn.Linear(DYN_LSD, APP_LSD)
    
    def forward(self, w):
        z_1 = self.lin1(w)
        time_output = torch.arange(0, PROC_PATH_LENGTH, dtype=torch.float)
        return odeint(self.integrand, z_1, time_output)


if __name__ == "__main__":
    decoders = [Sequential(), HiddenState(), Recurrent(), NeuralIntegrator()]
    for decoder in decoders:
        print(type(decoder))
        summary(decoder, (DYN_LSD,))
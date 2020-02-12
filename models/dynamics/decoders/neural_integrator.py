import torch
from torch import nn
import torch.nn.functional as f
from torchdiffeq import odeint
from torchsummary import summary
import const

class Integrand(nn.Module):
    def __init__(self):
        super(Integrand, self).__init__()
        
        self.lin1 = nn.Linear(1 + const.APP_LSD + const.DYN_LSD, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, const.APP_LSD)
    
    def forward(self, t, z, w):
        batch_size = z.shape[0]
        t = torch.full((batch_size, 1), t.item())
        print(t.size(), z.size(), w.size())
        inputs = torch.cat((t, z, w), dim=1)
        print(inputs.size())
        out = f.leaky_relu(self.lin1(inputs))
        out = f.leaky_relu(self.lin2(out))
        out = torch.sigmoid(self.lin3(out))       
        return out

class NeuralIntegrator(nn.Module):
    def __init__(self):
        super(NeuralIntegrator, self).__init__()
        
        self.integrand = Integrand()
        self.lin1 = nn.Linear(const.DYN_LSD, const.APP_LSD)
        
    def odefunc(self, w):
        return lambda t, z: self.integrand(t, z, w)
        
    def forward(self, w):
        z_1 = self.lin1(w)
        time_output = torch.arange(0, const.PROC_PATH_LENGTH, dtype=torch.float)
        output = odeint(self.odefunc(w), z_1, time_output)
        output = output.transpose(0, 1)
        return output


if __name__ == "__main__":
    decoder = NeuralIntegrator()
    w = torch.randn((7, 2))
    out = decoder(w)
    print(out.shape)
import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary
import const

class ObservationModule(nn.Module):
    def __init__(self):
        super(ObservationModule, self).__init__()
        self.lin1 = nn.Linear(2 * const.APP_LSD, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, const.APP_LSD)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = torch.sigmoid(self.lin3(h))
        return h
        
class Sequential(nn.Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.lin1 = nn.Linear(const.DYN_LSD, const.APP_LSD)
        self.mods = nn.ModuleList([ObservationModule() for _ in range(const.PROC_PATH_LENGTH)])
    
    def forward(self, w):
        z_1 = torch.sigmoid(self.lin1(w))
        recon_path_list = [z_1]
        for module in self.mods:
            module_input = torch.cat((w, recon_path_list[-1]), dim=1)
            new_z = module(module_input)
            recon_path_list.append(new_z)
        recon_path = torch.stack(recon_path_list, dim=1)
        return recon_path


if __name__ == "__main__":
    decoder = Sequential()
    summary(decoder, (const.DYN_LSD,))
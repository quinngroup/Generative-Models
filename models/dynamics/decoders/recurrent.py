import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary
import const

class Recurrent(nn.Module):
    def __init__(self):
        super(Recurrent, self).__init__()
        
        self.h_to_h_1 = nn.Linear(const.DYN_LSD, 50)
        self.h_to_h_2 = nn.Linear(50, 20)
        self.h_to_h_3 = nn.Linear(20, const.DYN_LSD)
        
        self.h_to_z_1 = nn.Linear(const.DYN_LSD, 50)
        self.h_to_z_2 = nn.Linear(50, 20)
        self.h_to_z_3 = nn.Linear(20, const.APP_LSD)    
        
    def forward(self, w):
        h = w
        points = []        
        for _ in range(const.PROC_PATH_LENGTH):
            h = f.leaky_relu(self.h_to_h_1(h))
            h = f.leaky_relu(self.h_to_h_2(h))
            h = f.leaky_relu(self.h_to_h_3(h))
            
            z = torch.sigmoid(self.h_to_z_1(h))
            z = torch.sigmoid(self.h_to_z_2(z))
            z = torch.sigmoid(self.h_to_z_3(z))
            
            points.append(z)
            
        recon_path = torch.stack(points, dim=1)
        return recon_path
        
if __name__ == "__main__":
    decoder = Recurrent()
    summary(decoder, (const.DYN_LSD,))
import torch
from torch import nn
import torch.nn.functional as f

class ObservationModule(nn.Module):
    def __init__(self):
        super(ObservationModule, self).__init__()
        self.lin1 = nn.Linear(4, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 2)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = torch.sigmoid(self.lin3(h))
        return h

class Observation(nn.Module):
    def __init__(self, num_points):
        super(Observation, self).__init__()
        self.NUM_POINTS = num_points
        self.lin1 = nn.Linear(2, 2)
        self.modules = [ObservationModule() for _ in range(self.NUM_POINTS - 1)]
    
    def forward(self, w):
        z_1 = torch.sigmoid(self.lin1(w))
        recon_path_list = [z_1]
        for module in self.modules:
            module_input = torch.cat((w, recon_path_list[-1]), dim=1)
            new_z = torch.sigmoid(module(module_input))
            recon_path_list.append(new_z)
        recon_path = torch.stack(recon_path_list, dim=1)
        return recon_path


class HiddenState(nn.Module):
    def __init__(self, num_points):
        super(HiddenState, self).__init__()
        self.NUM_POINTS = num_points
        
        self.lin1 = nn.Linear(2, 100)
        self.lin2 = nn.Linear(100, 150)
        self.lin3 = nn.Linear(150, 80)
        self.lin4 = nn.Linear(80, 40)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = f.leaky_relu(self.lin3(h))
        h = torch.sigmoid(self.lin4(h))        
        h = h.view((-1, 20, 2))
        return h
        



class Recurrent(nn.Module):
    def __init__(self, num_points):
        super(Recurrent, self).__init__()
        self.NUM_POINTS = num_points
        
        self.h_to_h_1 = nn.Linear(2, 50)
        self.h_to_h_2 = nn.Linear(50, 20)
        self.h_to_h_3 = nn.Linear(20, 2)
        
        self.h_to_x_1 = nn.Linear(2, 50)
        self.h_to_x_2 = nn.Linear(50, 20)
        self.h_to_x_3 = nn.Linear(20, 2)    
        
    def forward(self, w):
        h = w
        points = []        
        for _ in range(self.NUM_POINTS):
            h = f.leaky_relu(self.h_to_h_1(h))
            h = f.leaky_relu(self.h_to_h_2(h))
            h = f.leaky_relu(self.h_to_h_3(h))
            
            x = torch.sigmoid(self.h_to_x_1(h))
            x = torch.sigmoid(self.h_to_x_2(x))
            x = torch.sigmoid(self.h_to_x_3(x))
            
            points.append(x)
            
        recon_path = torch.stack(points, dim=1)
        return recon_path
            
            
        


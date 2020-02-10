import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary

APP_LSD = 2
DYN_LSD = 2
PROC_PATH_LENGTH = 19

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(APP_LSD * PROC_PATH_LENGTH, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 8)
        self.mu = nn.Linear(8, DYN_LSD)
        self.logvar = nn.Linear(8, DYN_LSD)
    
    def forward(self, p):
        h = p.flatten(start_dim=1)
        h = f.leaky_relu(self.lin1(h))
        h = f.leaky_relu(self.lin2(h))
        h = f.leaky_relu(self.lin3(h))
        mu = torch.sigmoid(h)
        logvar = torch.sigmoid(h)
        return mu, logvar
        
# ALSO MAKE AN RNN ENCODER LATER
    
if __name__ == "__main__":
    model = FeedForward()
    summary(model, (PROC_PATH_LENGTH, APP_LSD))
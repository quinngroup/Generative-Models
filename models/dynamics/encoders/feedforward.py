import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary
import const

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(const.APP_LSD * const.PROC_PATH_LENGTH, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 8)
        self.mu = nn.Linear(8, const.DYN_LSD)
        self.logvar = nn.Linear(8, const.DYN_LSD)
    
    def forward(self, p):
        h = p.flatten(start_dim=1)
        h = f.leaky_relu(self.lin1(h))
        h = f.leaky_relu(self.lin2(h))
        h = f.leaky_relu(self.lin3(h))
        mu = torch.sigmoid(h)
        logvar = torch.sigmoid(h)
        return mu, logvar
    
if __name__ == "__main__":
    model = FeedForward()
    summary(model, (const.PROC_PATH_LENGTH, const.APP_LSD))
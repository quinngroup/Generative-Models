import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary
import const

class HiddenState(nn.Module):
    def __init__(self):
        super(HiddenState, self).__init__()
        
        self.lin1 = nn.Linear(const.DYN_LSD, 100)
        self.lin2 = nn.Linear(100, 150)
        self.lin3 = nn.Linear(150, 80)
        self.lin4 = nn.Linear(80, const.PROC_PATH_LENGTH * const.APP_LSD)
        
    def forward(self, w):
        h = f.leaky_relu(self.lin1(w))
        h = f.leaky_relu(self.lin2(h))
        h = f.leaky_relu(self.lin3(h))
        h = torch.sigmoid(self.lin4(h))        
        h = h.view((-1, const.PROC_PATH_LENGTH, const.APP_LSD))
        return h
		
if __name__ == "__main__":
    decoder = HiddenState()
    summary(decoder, (const.DYN_LSD,))
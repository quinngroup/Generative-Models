import torch
from torch import nn

class PathProcessing(nn.Module):
    def __init__(self, type):
        super(PathProcessing, self).__init__()
        self.type = type
        
    def forward(self, z):
        if self.type == "subtract first":
            first_frame = torch.stack([z[:, 0, :] for _ in range (z.size()[1])], dim=1)
            return (z - first_frame)[:, 1:, :] # DO WE ACCOUNT FOR THIS LATER          
        elif self.type == "subtract previous":
            return z[:, :-1, :] -  z[:, 1:, :]
        else:
            print("Error: incorrect type argument passed")
        

mod1 = PathProcessing("subtract first")
mod2 = PathProcessing("subtract previous")
z = torch.randn((7, 20, 2))
print(mod1(z).size())
print(mod2(z).size())


        
    
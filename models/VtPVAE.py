import torch
from torch import nn
from vamps.NatVampPrior import NatVampPrior
import VtPVAE_D


class VtPVAE(nn.Module):
    def  __init__(self):
        super(VtPVAE, self).__init__()
        
        app_head = NatVampPrior() # NEEDS MORE ARGUMENTS I THINK
        dyn_head = VTP_D()
    
        
        
        


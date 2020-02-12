import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary


class VTP_D(nn.Module):
    def __init__(self):
        super(VTP_D, self).__init__()
     
    # sample using mu and logvar
    def sample_from(self, mu, logvar):
        eps = torch.randn_like(mu)
        std_dev = torch.exp(logvar / 2)
        return mu + eps * std_dev
           
    def forward(self, p, encoder_type="feedforward", decoder_type="sequential"):
        if encoder_type == "feedforward":
            return 
        mu, logvar = self.encoder(inp)
        w = self.sample_from(mu, logvar)
        if decoder_type == "sequential":
            return self.decoder_seq(w)
        elif decoder_type == "hidden state":
            return self.decoder_hid(w)
        elif decoder_type == "recurrent":
            return self.decoder_rec(w)
        elif decoder_type == "neural integrator":
            return self.decoder_ode(w)

#model = VTP_D()
#summary(model, (20, 2))
        
    
        
        
        
        
        
        
        
        
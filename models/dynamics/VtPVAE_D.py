import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary
import vtp_decoders

class VTP_D(nn.Module):
    def __init__(self):
        super(VTP_D, self).__init__()
        
        self.NUM_POINTS = 20
        
        # 3 linear layers for encoder  
        self.enc_lin_1 = nn.Linear(40, 40) 
        self.enc_lin_2 = nn.Linear(40, 20)
        self.enc_lin_3 = nn.Linear(20, 10)
        self.enc_mu = nn.Linear(10, 2)
        self.enc_logvar = nn.Linear(10, 2)
        
        self.decoder_seq = vtp_decoders.Sequential()
        self.decoder_hid = vtp_decoders.HiddenState()
        self.decoder_rec = vtp_decoders.Recurrent()
        self.decoder_ode = vtp_decoders.NeuralIntegrator()
        
    def encoder(self, input):
    
        # flatten input while keeping batch axis
        input = torch.flatten(input, start_dim=1)
        
        # pass input through encoder
        input = f.leaky_relu(self.enc_lin_1(input))
        input = f.leaky_relu(self.enc_lin_2(input))
        input = f.leaky_relu(self.enc_lin_3(input))
        mu = torch.sigmoid(self.enc_mu(input))
        logvar = torch.sigmoid(self.enc_logvar(input))
        
        # return mu and logvar of posterior
        return mu, logvar
     
    # sample using mu and logvar
    def sample_from(self, mu, logvar):
        eps = torch.randn_like(mu)
        std_dev = torch.exp(logvar / 2)
        return mu + eps * std_dev
        
     
    def forward(self, input, decoder_type="sequential"):
        mu, logvar = self.encoder(input)
        w = self.sample_from(mu, logvar)
        if decoder_type == "sequential":
            return self.decoder_seq(w)
        elif decoder_type == "hidden state":
            return self.decoder_hid(w)
        elif decoder_type == "recurrent":
            return self.decoder_rec(w)
        elif decoder_type == "neural integrator":
            return self.decoder_ode(w)

model = VTP_D()
summary(model, (20, 2))
#print(model(torch.randn((7, 20, 2))).size())
        
    
        
        
        
        
        
        
        
        
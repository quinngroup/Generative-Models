
import torch

import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kpl=1, kernel_size=3, padding=0):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kpl, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kpl, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
def spatial_broadcast_decoder(self,z):
    baseVector = z.view(-1, self.lsdim, 1, 1)
    base = baseVector.repeat(1,1,self.input_length,self.input_length)
    stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)
    xAxisVector = stepTensor.view(1,1,self.input_length,1)
    yAxisVector = stepTensor.view(1,1,1,self.input_length)
    xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
    yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)
    return torch.cat((xPlane, yPlane, base), 1)         

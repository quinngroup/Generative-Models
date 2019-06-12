from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import LeakyReLU

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kpl=1, kernel_size=3, padding=0):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kpl, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kpl, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class spatial_broadcast_decoder(nn.Module):
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder,self).__init__()
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        #Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels[3], 1, 1)

    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x
class true_spatial_broadcast_decoder(nn.Module):
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder,self).__init__()
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        #Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels[3], 1, kernel_size=kernel_size, padding=padding)

    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x

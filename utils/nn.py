import torch
from torch import nn, eye
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
        
'''
Spatial Broadcast Decoder with Batch Normalization

@author Quinn Wyner
'''
class spatial_broadcast_decoder_batchnorm(nn.Module):
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder_batchnorm,self).__init__()
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        
        #Size-Preserving Convolutions
        
        #(lsdim + 2, input_length, input_length) -> (channels[0], input_length, input_length)
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        
        #(channels[0], input_length, input_length) -> (channels[0], input_length, input_length)
        self.batch1 = nn.BatchNorm2d(channels[0])
        
        #(channels[0], input_length, input_length) -> (channels[1], input_length, input_length)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        
        #(channels[1], input_length, input_length) -> (channels[1], input_length, input_length)
        self.batch2 = nn.BatchNorm2d(channels[1])
        
        #(channels[1], input_length, input_length) -> (channels[2], input_length, input_length)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        
        #(channels[2], input_length, input_length) -> (channels[2], input_length, input_length)
        self.batch3 = nn.BatchNorm2d(channels[2])
        
        #(channels[2], input_length, input_length) -> (channels[3], input_length, input_length)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        
        #(channels[3], input_length, input_length) -> (channels[3], input_length, input_length)
        self.batch4 = nn.BatchNorm2d(channels[3])
        
        #(channels[3], input_length, input_length) -> (1, input_length, input_length)
        self.conv5 = nn.Conv2d(channels[3], 1, 1)
        
        #(1, input_length, input_length) -> (1, input_length, input_length)
        self.batch5 = nn.BatchNorm2d(1)

    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        #(lsdim+2, input_length, input_length) -> (channels[0], input_length, input_length)
        x = F.leaky_relu(self.batch1(self.conv1(base)))
        
        #(channels[0], input_length, input_length) -> (channels[1], input_length, input_length)
        x = F.leaky_relu(self.batch2(self.conv2(x)))
        
        #(channels[1], input_length, input_length) -> (channels[2], input_length, input_length)
        x = F.leaky_relu(self.batch3(self.conv3(x)))
        
        #(channels[2], input_length, input_length) -> (channels[3], input_length, input_length)
        x = F.leaky_relu(self.batch4(self.conv4(x)))
        
        #(channels[3], input_length, input_length) -> (1, input_length, input_length)
        x = F.leaky_relu(self.batch5(self.conv5(x)))

        return x
        
class true_spatial_broadcast_decoder(nn.Module):
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(true_spatial_broadcast_decoder,self).__init__()
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
        
class spatial_broadcast_decoder_asymmetric(nn.Module):
    def __init__(self,input_height,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder_asymmetric,self).__init__()
        self.input_height = input_height
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
        base = baseVector.repeat(1,1,self.input_height,self.input_length)

        heightStepTensor = torch.linspace(-1, 1, self.input_height).to(self.device)
        lengthStepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)
        heightAxisVector = heightStepTensor.view(1,1,self.input_height,1)
        lengthAxisVector = lengthStepTensor.view(1,1,1,self.input_length)

        xPlane = heightAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = lengthAxisVector.repeat(z.shape[0],1,self.input_height,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x

"""
ResNet-style block that preserves dimensionality of input

@author Quinn Wyner
"""
class ResNetBlock(nn.Module):
    
    '''
    Constructs a ResNetBlock
    
    @param channels number of filters
    @param kernel_size size of a kernel; either an int or tuple of 2 ints
    @param numLayers number of convolutions to perform
    @param activationFunction function to perform on layers; either a lambda function or a tuple of lambda functions
    @param shortcutInterval number of layers between each shortcut
    '''
    def __init__(self, channels, kernel_size, numLayers, activationFunction, shortcutInterval):
        super(ResNetBlock, self).__init__()
        if type(activationFunction) == tuple and len(activationFunction) != numLayers:
            raise Exception(f'length of activation function must be same as numLayers {numLayers}, but is instead {len(activationFunction)}')
        self.activationFunction = activationFunction
        self.shortcutInterval = shortcutInterval
        if type(kernel_size) == int:
            if kernel_size % 2 == 0:
                raise Exception(f'kernel_size must exclusively have odd values, but has value {kernel_size}')
                return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2) for i in range(numLayers)])

        else:
            for i in range(2):
                if kernel_size[i] % 2 == 0:
                    raise Exception(f'kernel_size must exclusively have odd values, but has value {kernel_size[0]}')
                    return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)) for i in range(numLayers)])
    def forward(self, x):
        shortcut = x
        z = x
        for i in range(len(self.layers)):
            shortcutLayer = ((i+1) % self.shortcutInterval == 0)
            z = self.layers[i](z)
            if shortcutLayer:
                z = z + shortcut
            if type(self.activationFunction) == tuple:
                z = self.activationFunction[i](z)
            else:
                z = self.activationFunction(z)
            if shortcutLayer:
                shortcut = z
        return z
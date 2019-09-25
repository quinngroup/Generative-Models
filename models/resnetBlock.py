from torch import eye, nn

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
    '''
    def __init__(self, channels, kernel_size, numLayers, activationFunction):
        super(ResNetBlock, self).__init__()
        self.activationFunction = activationFunction
        if type(kernel_size) == int:
            if kernel_size % 2 == 0:
                print(f'Error: kernel_size must exclusively have odd values, but has value {kernel_size}')
                return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2) for i in range(numLayers)])

        else:
            for i in range(2):
                if kernel_size[i] % 2 == 0:
                    print(f'Error: kernel_size must exclusively have odd values, but has value {kernel_size[0]}')
                    return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)) for i in range(numLayers)])
    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
            if type(self.activationFunction) == tuple:
                z = self.activationFunction[i](z)
            else:
                z = self.activationFunction(z)
        return z + x
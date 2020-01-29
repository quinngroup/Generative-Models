import torch

def process(input, type):
    # calcuate paths relative to first point
    # output vectors: x2 - x1, x3 - x1, ...
    if type == 'displacement':
        first_point = input[0]
        first_point_like = torch.stack(tuple(first_point for i in range(input.size()[0] - 1)))
        return input[1:] - first_point_like
    # calcuate paths relative to previous point
    # output vectors x2 - x1, x3 - x2, ...
    elif type == 'velocity':
        return input[1:] - input[:-1]
    else:
        print('Invalid processing step type!')
        
        
def unprocess(input, type):
    if type == 'displacement':
        pass
    elif type == 'velocity':
        pass
    else:
        print('Invalid processing step type!')



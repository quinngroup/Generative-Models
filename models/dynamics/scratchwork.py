import torch

t = torch.tensor([[1],[2]])
z = torch.tensor([[3,4],[5,6]])
w = torch.tensor([[7,8],[9,10]])
print(t.size(), z.size(), w.size())

inp = torch.cat((t,z,w), dim = 1)
print(inp)
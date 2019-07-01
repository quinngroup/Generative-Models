import torch
from datasetTemplate import nonOverlapDataset as Dataset

data=Dataset('../../bucket/grayscale',40)
print(len(data))

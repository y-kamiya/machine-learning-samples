import torch

paramA = torch.load('modelA')
print(paramA.keys())

paramB = torch.load('modelB')
print(paramB.keys())

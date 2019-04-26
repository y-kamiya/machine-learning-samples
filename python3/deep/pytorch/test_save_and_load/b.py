import torch
import torch.nn as nn

class ModelB(nn.Module):
  def __init__(self):
    super(ModelB,self).__init__()
    self.fc1_b = nn.Linear(11,100)
    self.fc2_b = nn.Linear(100,10)

  def forward(self,x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x

model = ModelB()

param = torch.load('modelA')
param['fc1_b.weight'] = param.pop('fc1.weight')
param['fc1_b.bias'] = param.pop('fc1.bias')
param['fc2_b.weight'] = param.pop('fc2.weight')
param['fc2_b.bias'] = param.pop('fc2.bias')
model.load_state_dict(param)
# print(model)
# torch.save(model.state_dict(), 'modelB')

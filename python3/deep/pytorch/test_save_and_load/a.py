import torch
import torch.nn as nn

class ModelA(nn.Module):
  def __init__(self):
    super(ModelA,self).__init__()
    self.fc1 = nn.Linear(10,100)
    self.fc2 = nn.Linear(100,10)

  def forward(self,x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x

model = ModelA()
print(model)
torch.save(model.state_dict(), 'modelA')

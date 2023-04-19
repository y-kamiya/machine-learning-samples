import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import batchnorm
from accelerate import Accelerator


class BNOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, inputs):
        return self.bn(inputs)


def cpu():
    print("--- cpu ---")
    model = BNOnlyModel()
    x = torch.arange(4, dtype=torch.float32).view(4, 1)** 2 # [0, 1, 4, 9]
    y = model(x)
    print(y)


def gpu():
    print("--- gpu ---")
    model = BNOnlyModel().cuda()
    x = torch.arange(4, dtype=torch.float32).cuda().view(4, 1)** 2
    y = model(x)
    print(y)


def data_parallel_wrong():
    print("--- (NG) DataParallel ---")
    model = BNOnlyModel().cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    x = torch.arange(4, dtype=torch.float32).cuda().view(4, 1)** 2
    y = model(x)
    print(y)


def data_parallel():
    print("--- DataParallel ---")
    model = BNOnlyModel().cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    model = batchnorm.convert_model(model).cuda()
    x = torch.arange(4, dtype=torch.float32).cuda().view(4, 1)** 2
    y = model(x)
    print(y)


def ddp():
    print("--- DDP ---")
    model = BNOnlyModel().cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    accelerator = Accelerator()
    x = torch.arange(4, dtype=torch.float32).cuda().view(4, 1)** 2
    dataloader = DataLoader(TensorDataset(x), batch_size=2)
    model, dataloader = accelerator.prepare(model, dataloader)

    for (x,) in dataloader:
        print(x)
        y = model(x)

    print(y)
    accelerator.wait_for_everyone()


def ddp_with_thirdparty_wrong():
    print("--- (NG) DDP with third party SyncBatchNorm ---")
    model = BNOnlyModel().cuda()
    model = batchnorm.convert_model(model)
    # you can get correct output by converting below
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    accelerator = Accelerator()
    x = torch.arange(4, dtype=torch.float32).cuda().view(4, 1)** 2
    dataloader = DataLoader(TensorDataset(x), batch_size=2)
    model, dataloader = accelerator.prepare(model, dataloader)

    for (x,) in dataloader:
        print(x)
        y = model(x)

    print(y)
    accelerator.wait_for_everyone()


# execute with python
# $ python main.py
cpu()
gpu()
data_parallel_wrong()
data_parallel()

# execute with accelerate
# $ accelerate launch main.py
# ddp()
# ddp_with_thirdparty_wrong()

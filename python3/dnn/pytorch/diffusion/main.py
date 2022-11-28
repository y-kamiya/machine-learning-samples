from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from logzero import setup_logger
from torch import Tensor, nn, optim, utils
from torch.utils import tensorboard
from torchvision import datasets
from torchvision.transforms import ToTensor


class ToyDataset(utils.data.Dataset):
    def __init__(self, dim=4, num_max=10):
        self.dim = dim
        self.num_max = num_max

        data = []
        for i in range(num_max - (dim - 1) + 1):
            data.append(list(range(i, i + dim)))

        self.data = self.normalize(torch.Tensor(data)).repeat(1000, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, x):
        return (x - self.num_max / 2) / self.num_max

    def denormalize(self, x):
        return x * self.num_max + self.num_max / 2


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, dim)

    def forward(self, x, t):
        x = F.relu(self.fc1(x.view(-1, self.dim)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Trainer(object):
    def __init__(self, config, logger, input_dim):
        self.config = config
        self.logger = logger

        self.input_dim = input_dim
        self.model = Model(self.input_dim)
        self.model.apply(self._weights_init)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8
        )

        beta0 = 1e-4
        betaT = 2e-2
        self.beta = torch.linspace(beta0, betaT, config.T)

        self.a_cum = torch.cumprod(1 - self.beta, dim=0)
        a_cum_prev = torch.cat((torch.Tensor([0]), self.a_cum[:-1]))
        self.sigma = (1 - a_cum_prev) / (1 - self.a_cum) * self.beta

        self.steps = 0
        self.writer = tensorboard.SummaryWriter(log_dir=config.tensorboard_log_dir)

        # e = torch.randn((2,4))
        # x = torch.tensor([[1,2,3,4],[5,6,7,8]])
        # print(self.q_sample(x, 10, e))
        # print(self.q_sample(x, 50, e))
        # print(self.q_sample(x, 100, e))
        # print(self.q_sample(x, 199, e))
        # sys.exit()

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
        elif classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self, dataloader, epoch: int):
        self.model.train()
        self.logger.info(f"start training epoch {epoch}")
        for x in dataloader:
            x = x.to(self.config.device)
            self.step(x)
            self.step_end()

    def step(self, x):
        e = torch.randn(x.shape)
        t = torch.randint(config.T, (1,)).item()

        t = 0
        input = self.q_sample(x, t, e)
        output = self.model(torch.flatten(input), t)

        # print(f"noise: {e}")
        # print(f"x: {x}")
        # print(f"input : {input}")
        # print(f"output: {output}")
        # print(list(self.model.parameters()))

        # loss = nn.functional.smooth_l1_loss(output, e)
        loss = nn.functional.mse_loss(output, e)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.steps % self.config.log_interval == 0:
            logger.info(f"[train] step: {self.steps}, loss: {loss:.3f}")
        self.writer.add_scalar("loss/train", loss, self.steps, time.time())

    def step_end(self):
        self.steps += 1

    @torch.no_grad()
    def evaluate(self, dataloader, epoch: int):
        self.model.eval()
        self.logger.info(f"evaluate epoch {epoch}")

        samples = self.sample(2)
        print("samples")
        print(dataloader.dataset.denormalize(samples))

    def q_sample(self, x0, t, e=None):
        if e is None:
            e = torch.rand_like(x0)
        return torch.sqrt(self.a_cum[t]) * x0 + torch.sqrt(1 - self.a_cum[t]) * e

    def sample(self, n: int):
        # xt = torch.randn((n, self.input_dim))
        # ts = range(self.config.T - 1, 1, -1)
        xt = torch.Tensor(
            [
                [-0.5000, -0.4000, -0.3000, -0.2000],
                [0.2000, 0.3000, 0.4000, 0.5000],
            ]
        )
        xt = self.q_sample(xt, 0)
        ts = [0]
        for t in ts:
            z = torch.randn((n, self.input_dim)) if t > 1 else 0
            e = self.beta[t] / torch.sqrt(1 - self.a_cum[t]) * self.model(xt, t)
            xt = (xt - e) / torch.sqrt(1 - self.beta[t]) + self.sigma[t] * z

        return xt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--dataroot", default="data", help="path to data")
    parser.add_argument(
        "--name",
        default="default",
        help="name of training, used to model name, log dir name etc",
    )
    parser.add_argument("--epochs", type=int, default=10, help="epoch count")
    parser.add_argument("--batch_size", type=int, default=2, help="size of batch")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="step num to display log"
    )
    parser.add_argument("--model_path", default=None, help="model path")
    parser.add_argument(
        "--eval_interval_epochs",
        type=int,
        default=5,
        help="evaluate by every this epochs ",
    )
    parser.add_argument(
        "--T", type=int, default=20, help="time step of diffusion process"
    )
    config = parser.parse_args()

    is_cpu = config.cpu or not torch.cuda.is_available()
    config.device_name = "cpu" if is_cpu else "cuda:0"
    config.device = torch.device(config.device_name)

    config.tensorboard_log_dir = f"{config.dataroot}/runs/{config.name}"
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)

    logger = setup_logger(name=__name__)

    input_dim = 4
    trainer = Trainer(config, logger, input_dim=input_dim)

    # dataset = datasets.MNIST(f"{config.dataroot}/train", download=True, transform=ToTensor())
    dataset = ToyDataset(input_dim, 2 * input_dim)
    dataloader = utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    for epoch in range(config.epochs):
        trainer.train(dataloader, epoch)

        if epoch % config.eval_interval_epochs == 0:
            trainer.evaluate(dataloader, epoch)

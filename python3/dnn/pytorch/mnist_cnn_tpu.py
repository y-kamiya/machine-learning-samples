from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torchvision import datasets, transforms, utils


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.linear1 = nn.Linear(7 * 7 * 64, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view([-1, 7 * 7 * 64])
        x = self.linear1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.linear2(x)
        return x


class Trainer():
    def __init__(self, config):
        self.config = config

    def get_dataset(self):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(self.config.dataroot, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(self.config.dataroot, train=False, download=True, transform=transform)

        return train_dataset, test_dataset


    def train(self):
        train_dataset, test_dataset = SERIAL_EXECUTOR.run(self.get_dataset)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.n_workers,
            drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.n_workers,
            drop_last=True)

        lr = self.config.lr * xm.xrt_world_size()

        device = xm.xla_device()
        model = WRAPPED_MODEL.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # if args.model != None:
        #     model.load_state_dict(torch.load(args.model, map_location=device_name), strict=False)

        start_time = time.time()

        def train_loop_fn(loader, config):
            tracker = xm.RateTracker()
            model.train()

            for batch_index, (data, target) in enumerate(loader):
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                xm.optimizer_step(optimizer)

                tracker.add(config.batch_size)
                if batch_index % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print("[xla:{}]({}) Elapsed: {:0.2f} sec, Loss: {:0.3f}, Rate: {:.2f}, GlobalRate: {:.2f}, AscTime: {}".format(
                        xm.get_ordinal(), batch_index, elapsed_time, loss.item(),
                        tracker.rate(), tracker.global_rate(), time.asctime()), flush=True)

        def test_loop_fn(loader):
            total_samples = 0
            correct = 0
            model.eval()
            data, pred, target = None, None, None
            for data, target in loader:
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size()[0]

            accuracy = 100.0 * correct / total_samples
            print('[xla:{}] Accuracy={:.2f}%'.format(
                xm.get_ordinal(), accuracy), flush=True)
            return accuracy, data, pred, target

        for epoch in range(1, args.epochs + 1):
            para_loader = pl.ParallelLoader(train_loader, [device])
            train_loop_fn(para_loader.per_device_loader(device), self.config)
            xm.master_print('Finished training epoch {}'.format(epoch))

            para_loader = pl.ParallelLoader(test_loader, [device])
            accuracy, data, pred, target = test_loop_fn(para_loader.per_device_loader(device))

            if self.config.metrics_debug:
                xm.master_print(met.metrics_report(), flush=True)

        return accuracy, data, pred, target


def mp_fn(rank, config):
    torch.set_default_tensor_type('torch.FloatTensor')
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--model', help='model to load')
    parser.add_argument('--estimate', help='image file path to estimate class')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_cores', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--metrics_debug', action='store_true')
    args = parser.parse_args()
    print(args)

    SERIAL_EXECUTOR = xmp.MpSerialExecutor()
    WRAPPED_MODEL = xmp.MpModelWrapper(Cnn())

    xmp.spawn(mp_fn, args=(args,), nprocs=args.n_cores, start_method='fork')

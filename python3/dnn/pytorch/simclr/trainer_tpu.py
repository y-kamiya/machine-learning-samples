import sys
import os
import time
import uuid
import math
import argparse
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser
import PIL
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from logzero import setup_logger
from sklearn import metrics
from tabulate import tabulate


class SimCLRDataset(Dataset):
    def __init__(self, config, split):
        super(SimCLRDataset, self).__init__()
        self.config = config

        self.image_size = 224
        self.blur_kernel_size = 21
        if config.pretrain_with_cifar10:
            self.image_size = 32
            self.blur_kernel_size = 0

        self.transform = self.__transform()
        train = True if split == 'train' else False
        self.data = datasets.CIFAR10(config.dataroot, train=train, download=True)

    def __transform(self):
        s = 0.5 if self.config.pretrain_with_cifar10 else 1.0

        list = [
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            self.__transform_color_distortion(s),
        ]

        if not self.config.pretrain_with_cifar10:
            list.append(transforms.GaussianBlur(self.blur_kernel_size))

        list.append(transforms.ToTensor())

        return transforms.Compose(list)

    def __transform_color_distortion(self, s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)

        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        return transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
        ])

    def __getitem__(self, index):
        image, _ = self.data[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.data)


class LinearEvaluationDataset(Dataset):
    def __init__(self, config, split):
        super(LinearEvaluationDataset, self).__init__()
        self.config = config

        self.image_size = 224
        self.blur_kernel_size = 21
        if config.pretrain_with_cifar10:
            self.image_size = 32
            self.blur_kernel_size = 0

        train = True if split == 'train' else False
        self.data = datasets.CIFAR10(config.dataroot, train=train, download=True,
                                     transform=self.__transform())

    def __transform(self):
        list = [
            transforms.Resize(self.image_size, PIL.Image.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ]
        return transforms.Compose(list)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super(ProjectionHead, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config.n_classes_encoder, config.projection_hidden_dim, bias=False)
        self.fc2 = nn.Linear(config.projection_hidden_dim, config.projection_output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(config.projection_hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.projection_output_dim)

        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.encoder = torchvision.models.resnet50(num_classes=config.n_classes_encoder)
        if config.pretrain_with_cifar10:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
        self.projection_head = ProjectionHead(config)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z, h


class LinearEvaluationHead(nn.Module):
    def __init__(self, config):
        super(LinearEvaluationHead, self).__init__()
        self.config = config

        self.fc = nn.Linear(config.n_classes_encoder, config.n_classes_eval, bias=True)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        return self.fc(x)


class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = xm.xla_device()
        self.model = WRAPPED_MODEL.to(self.device)

        lr = 0.075 * math.sqrt(config.batch_size) * xm.xrt_world_size()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-6)

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs - config.warmup_epochs)

        self.warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / config.warmup_epochs))

        train_dataset = SERIAL_EXECUTOR.run(lambda: SimCLRDataset(config, 'train'))
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

        self.dataloader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.n_workers,
            drop_last=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.start_time = time.time()

        self.start_epoch = self.load(config.model_path)

    def train(self):
        for epoch in range(self.start_epoch, args.epochs + 1):
            para_loader = pl.ParallelLoader(self.dataloader_train, [self.device], loader_prefetch_size=2, device_prefetch_size=2)
            self.train_loop_fn(para_loader.per_device_loader(self.device), epoch)

            if epoch % self.config.save_interval == 0:
                self.save(self.config.model_path, epoch)

            if self.config.warmup_epochs < epoch:
                self.lr_scheduler.step()
            else:
                self.warmup_scheduler.step()

            xm.master_print('Finished training epoch {}'.format(epoch))
            if self.config.metrics_debug:
                xm.master_print(met.metrics_report(), flush=True)

    def train_loop_fn(self, loader, epoch):
        self.model.train()
        tracker = xm.RateTracker()

        for i, (x1, x2) in enumerate(loader):
            self.optimizer.zero_grad()
            loss = self.__loss(x1, x2)
            loss.backward()
            xm.optimizer_step(self.optimizer)

            tracker.add(self.config.batch_size)
            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                self.config.logger.info("[xla:{}](train epoch: {}, step: {}) Elapsed: {:0.2f} sec, Loss: {:0.3f}, LR: {:.2e} {:.2e}, Rate: {:.2f}, GlobalRate: {:.2f}".format(
                    xm.get_ordinal(), epoch, i, elapsed_time, loss.item(), self.warmup_scheduler.get_last_lr()[0], self.lr_scheduler.get_last_lr()[0],
                    tracker.rate(), tracker.global_rate()))

            self.writer.add_scalar('train/loss', loss, epoch, time.time())

    def __loss(self, x1, x2):
        batch_size = x1.shape[0]

        x = torch.cat([x1, x2])
        z, _ = self.model(x)
        z1, z2 = torch.split(F.normalize(z, p=2), batch_size)

        labels = torch.eye(batch_size, batch_size * 2, device=self.device)
        masks = torch.eye(batch_size, device=self.device)

        temperature = self.config.temperature
        logits_aa = torch.matmul(z1, z1.T) / temperature
        logits_aa = logits_aa - masks * 1e9
        logits_bb = torch.matmul(z2, z2.T) / temperature
        logits_bb = logits_bb - masks * 1e9
        logits_ab = torch.matmul(z1, z2.T) / temperature
        logits_ba = torch.matmul(z2, z1.T) / temperature

        target = torch.argmax(labels, dim=1)
        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), target, reduction='none')
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), target, reduction='none')

        return torch.mean(loss_a + loss_b)

    def save(self, model_path, epoch):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        xm.save(data, model_path)

        self.config.logger.info(f'save model to {model_path}')

    def load(self, model_path):
        if not os.path.isfile(model_path):
            return 1

        data = torch.load(model_path)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])

        self.config.logger.info(f'load model from {model_path}')

        return data['epoch'] + 1


class LinearEvaluationTrainer():
    def __init__(self, config):
        self.config = config

        self.device = xm.xla_device()
        self.base_model = WRAPPED_MODEL.to(self.device)
        self.head = WRAPPED_MODEL_EVAL_HEAD.to(self.device)

        for param in self.base_model.parameters():
            param.requires_grad = False

        lr = 0.1 * config.batch_size / 256 * xm.xrt_world_size()
        self.optimizer = optim.SGD(self.head.parameters(), lr=lr, weight_decay=0.1)
        # self.optimizer = optim.LBFGS(self.head.parameters(), lr=lr)

        train_dataset, eval_dataset = SERIAL_EXECUTOR.run(lambda: 
            (LinearEvaluationDataset(config, 'train'), LinearEvaluationDataset(config, 'test')))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

        self.dataloader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.eval_batch_size,
            sampler=train_sampler,
            num_workers=self.config.n_workers,
            drop_last=True)

        self.dataloader_eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.n_workers,
            drop_last=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.start_time = time.time()

        self.start_epoch = 1
        if config.base_model_path is not None:
            self.start_epoch = self.load(config.base_model_path)

    def train(self):
        for epoch in range(self.start_epoch, args.eval_epochs + 1):
            para_loader = pl.ParallelLoader(self.dataloader_train, [self.device])
            self.train_loop_fn(para_loader.per_device_loader(self.device), epoch)

            xm.master_print('Finished training epoch {}'.format(epoch))
            if self.config.metrics_debug:
                xm.master_print(met.metrics_report(), flush=True)

    def train_loop_fn(self, loader, epoch):
        tracker = xm.RateTracker()
        self.base_model.eval()
        self.head.train()

        for i, (data, labels) in enumerate(loader):
            self.optimizer.zero_grad()
            loss, _ = self.__loss(data, labels)
            loss.backward()
            xm.optimizer_step(self.optimizer)

            tracker.add(self.config.batch_size)
            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                template = "[xla:{}](eval train epoch: {}, step: {}) Elapsed: {:0.2f} sec, Loss: {:0.3f}, Rate: {:.2f}, GlobalRate: {:.2f}"
                self.config.logger.info(template.format(
                    xm.get_ordinal(), epoch, i, elapsed_time, loss.item(),
                    tracker.rate(), tracker.global_rate()))

            self.writer.add_scalar('eval_train/loss', loss, epoch, time.time())

    def __loss(self, data, labels):
        with torch.no_grad():
            _, h = self.base_model(data)

        output = self.head(h)

        return F.cross_entropy(output, labels), output

    @torch.no_grad()
    def eval(self):
        para_loader = pl.ParallelLoader(self.dataloader_eval, [self.device])
        self.eval_loop_fn(para_loader.per_device_loader(self.device))

    @torch.no_grad()
    def eval_loop_fn(self, loader):
        self.base_model.eval()
        self.head.eval()

        all_labels = torch.empty(0)
        all_preds = torch.empty(0)
        losses = []

        for i, (data, labels) in enumerate(loader):
            loss, logits = self.__loss(data, labels)
            losses.append(loss)
            preds = torch.argmax(logits, dim=1)

            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

        elapsed_time = time.time() - self.start_time
        average_loss = sum(losses)/len(losses)
        self.config.logger.info('[xla:{}] loss: {:.2f}%, time: {:.2f}'.format(
            xm.get_ordinal(), average_loss, elapsed_time))

        df = pd.DataFrame(metrics.classification_report(all_labels, all_preds, output_dict=True))
        print(tabulate(df, headers='keys', tablefmt="github", floatfmt='.2f'))

    def save(self, model_path, epoch):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'eval_epoch': epoch,
        }
        xm.save(data, model_path)

        self.config.logger.info(f'save model to {model_path}')

    def load(self, model_path):
        if not os.path.isfile(model_path):
            return 1

        self.config.logger.info(f'load model from {model_path}')

        data = torch.load(model_path)
        self.base_model.load_state_dict(data['model'])
        if 'eval_epoch' not in data:
            return 1

        self.optimizer.load_state_dict(data['optimizer'])

        return data['eval_epoch'] + 1


def mp_train_fn(rank, config):
    torch.set_default_tensor_type('torch.FloatTensor')
    trainer = Trainer(config)
    trainer.train()


def mp_eval_fn(rank, config):
    torch.set_default_tensor_type('torch.FloatTensor')
    trainer = LinearEvaluationTrainer(config)
    trainer.train()
    trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--dataroot', default='data', help='path to data directory')
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', type=int, default=100, help='epochs for simclr training')
    parser.add_argument('--eval_epochs', type=int, default=90, help='epoch count')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch for simclr training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='size of batch for eval')
    parser.add_argument('--n_classes_encoder', type=int, default=2048, help='dimension of encoder output (h)')
    parser.add_argument('--n_classes_eval', type=int, default=10, help='dimension of evaluation output')
    parser.add_argument('--projection_hidden_dim', type=int, default=2048, help='dimension of projection hidden layer')
    parser.add_argument('--projection_output_dim', type=int, default=128, help='dimension of projection output (z)')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature parameter of loss function')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_cores', type=int, default=8)
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--eval', default=None, choices=['linear'])
    parser.add_argument('--base_model_path', default=None)
    parser.add_argument('--pretrain_with_cifar10', action='store_true')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    if args.name is None:
        args.name = str(uuid.uuid4())[:8]

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'
    args.model_path = f'{args.dataroot}/{args.name}.pth'

    if args.pretrain_with_cifar10:
        args.temperature = 0.5

    SERIAL_EXECUTOR = xmp.MpSerialExecutor()
    WRAPPED_MODEL = xmp.MpModelWrapper(Model(args))

    if args.eval is None:
        xmp.spawn(mp_train_fn, args=(args,), nprocs=args.n_cores, start_method='fork')
        sys.exit()

    WRAPPED_MODEL_EVAL_HEAD = xmp.MpModelWrapper(LinearEvaluationHead(args))

    xmp.spawn(mp_eval_fn, args=(args,), nprocs=args.n_cores, start_method='fork')


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
    IMAGE_SIZE = 224
    BLUR_KERNEL_SIZE = 21

    def __init__(self, config, split):
        super(SimCLRDataset, self).__init__()
        self.config = config

        self.transform = self.__transform()
        train = True if split == 'train' else False
        self.data = datasets.CIFAR10(config.dataroot, train=train, download=True)

    def __transform(self):
        list = [
            transforms.RandomResizedCrop(self.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            self.__transform_color_distortion(),
            transforms.GaussianBlur(self.BLUR_KERNEL_SIZE),
            transforms.ToTensor(),
        ]
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

        train = True if split == 'train' else False
        self.data = datasets.CIFAR10(config.dataroot, train=train, download=True,
                                     transform=self.__transform())

    def __transform(self):
        list = [
            transforms.Resize(SimCLRDataset.IMAGE_SIZE, PIL.Image.BICUBIC),
            transforms.CenterCrop(SimCLRDataset.IMAGE_SIZE),
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

        self.model = Model(config)
        self.eval_trainer = LinearEvaluationTrainer(config, self.model)

        lr = 0.075 * math.sqrt(config.batch_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-6)

        self.dataloader_train = DataLoader(SimCLRDataset(config, 'train'), batch_size=config.batch_size, shuffle=True)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

    def train(self, epoch):
        self.model.train()

        for i, (x_odd, x_even) in enumerate(self.dataloader_train):
            start_time = time.time()
            self.optimizer.zero_grad()
            loss = self.__loss(x_odd, x_even)
            loss.backward()

            self.optimizer.step()

            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.config.logger.info('train epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, i, loss, elapsed_time))

            self.writer.add_scalar('train/loss', loss, epoch, start_time)

    def __loss(self, x_odd, x_even):
        z_odd, h_odd = self.model(x_odd)
        z_even, h_even = self.model(x_even)

        z_odd_slided = torch.cat([z_odd[1:], z_odd[0].unsqueeze(0)])

        s_positive = self.__similarity(z_odd, z_even)
        s_negative = self.__similarity(z_odd_slided, z_even)

        batch_size = len(s_negative)
        return (s_positive + s_negative).sum() / (2 * batch_size)

    def __similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2)

    def eval(self, epoch):
        for epoch in range(args.eval_epochs):
            self.eval_trainer.train(epoch)

        self.eval_trainer.eval(epoch)


class LinearEvaluationTrainer():
    def __init__(self, config, base_model):
        self.config = config

        self.base_model = base_model
        self.head = LinearEvaluationHead(config)

        lr = 0.1 * config.batch_size / 256
        self.optimizer = optim.AdamW(self.head.parameters(), lr=lr)
        # self.optimizer = optim.LBFGS(self.head.parameters(), lr=lr)

        self.dataloader_train = DataLoader(LinearEvaluationDataset(config, 'train'),
                                           batch_size=config.eval_batch_size, shuffle=True)
        self.dataloader_eval = DataLoader(LinearEvaluationDataset(config, 'test'),
                                          batch_size=config.eval_batch_size, shuffle=False)

    def train(self, epoch):
        self.base_model.train()
        self.head.train()

        for i, (data, labels) in enumerate(self.dataloader_train):
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.__loss(data, labels)
            loss.backward()
            self.optimizer.step()

            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.config.logger.info('eval epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, i, loss, elapsed_time))

    def __loss(self, data, labels):
        with torch.no_grad():
            _, h = self.base_model(data)

        output = self.head(h)

        return F.cross_entropy(output, labels), output

    @torch.no_grad()
    def eval(self, epoch):
        self.base_model.eval()
        self.head.eval()

        all_labels = torch.empty(0)
        all_preds = torch.empty(0)
        losses = []
        start_time = time.time()

        for i, (data, labels) in enumerate(self.dataloader_eval):
            loss, logits = self.__loss(data, labels)
            losses.append(loss)
            preds = torch.argmax(logits, dim=1)

            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

        elapsed_time = time.time() - start_time
        average_loss = sum(losses)/len(losses)
        self.config.logger.info('eval epoch: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, average_loss, elapsed_time))

        df = pd.DataFrame(metrics.classification_report(all_labels, all_preds, output_dict=True))
        print(tabulate(df, headers='keys', tablefmt="github", floatfmt='.2f'))

        # f1_score = df.loc['f1-score']
        # micro = f1_score['micro avg'] if 'micro avg' in f1_score else f1_score['accuracy']
        # macro = f1_score['macro avg']
        # self.writer.add_scalar('eval/loss', average_loss, epoch, start_time)
        # self.writer.add_scalar('eval/f1_score_micro(accuracy)', micro, epoch, start_time)
        # self.writer.add_scalar('eval/f1_score_macro', macro, epoch, start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--dataroot', default='data', help='path to data directory')
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', type=int, default=100, help='epochs for simclr training')
    parser.add_argument('--eval_epochs', type=int, default=90, help='epoch count')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch for simclr training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='size of batch for eval')
    parser.add_argument('--n_classes_encoder', type=int, default=2048, help='dimension of encoder output (h)')
    parser.add_argument('--n_classes_eval', type=int, default=10, help='dimension of evaluation output')
    parser.add_argument('--projection_hidden_dim', type=int, default=2048, help='dimension of projection hidden layer')
    parser.add_argument('--projection_output_dim', type=int, default=128, help='dimension of projection output (z)')
    args = parser.parse_args()

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda:0"
    args.device = torch.device(args.device_name)

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    if args.name is None:
        args.name = str(uuid.uuid4())[:8]

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'

    trainer = Trainer(args)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        if epoch % args.eval_interval == 0:
            trainer.eval(epoch)

    trainer.eval(epoch)
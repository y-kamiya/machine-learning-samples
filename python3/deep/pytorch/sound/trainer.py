import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logzero import setup_logger
import numpy as np
import os
import argparse
import time
import sys

import model
import dataset

class Trainer():
    def __init__(self, config):
        self.config = config

        self.model = self.__create_model()
        self.optimizer = self.__create_optimizer(self.model)
        self.scheduler = self.__create_scheduler(self.optimizer)
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

        self.model.apply(self.__weights_init)

    def __weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def update_epoch(self):
        self.scheduler.step()

    def train(self, dataloader, epoch):
        self.model.train()
        device = self.config.device

        for batch_idx, (data, target, _) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.__loss(output, target)
            loss.backward()
            self.optimizer.step()

            n_processed_data = (epoch-1) * len(dataloader.dataset) + (batch_idx+1) * self.config.batch_size
            self.writer.add_scalar('loss/train', loss, n_processed_data, time.time())

            if batch_idx % self.config.log_interval == 0: #print training stats
                self.config.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.config.batch_size, len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss))

    def __loss(self, output, target):
        if self.config.model_type == 'envnet':
            return F.cross_entropy(output, target)

        return F.nll_loss(output, target)

    def __create_model(self):
        device = self.config.device
        if self.config.model_type == 'envnet':
            return model.EnvNet(self.config).to(device)

        if self.config.model_type == 'escconv':
            return model.EscConv(self.config).to(device)

        return model.M5().to(device)

    def __create_optimizer(self, model):
        if self.config.model_type == 'envnet':
            if self.config.use_adam:
                return optim.Adam(model.parameters(), lr=1.0)

            return optim.SGD(model.parameters(), momentum=0.9, lr=1.0, weight_decay=0.001, nesterov=True)

        if self.config.model_type == 'escconv':
            if self.config.use_adam:
                return optim.Adam(model.parameters(), lr=self.config.lr)

            return optim.SGD(model.parameters(), momentum=0.9, lr=self.config.lr, weight_decay=0.001, nesterov=True)

        return optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=0.0001)

    def __create_scheduler(self, optimizer):
        if self.config.model_type == 'envnet':
            def func(epoch):
                if epoch < 80:
                    return 1e-2
                elif epoch < 100:
                    return 1e-3
                elif epoch < 120:
                    return 1e-4
                else:
                    return 1e-5
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

        if self.config.model_type == 'escconv':
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda _: 1.0)

        return optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.1)

    @torch.no_grad()
    def eval(self, dataloader, epoch):
        self.model.eval()
        device = self.config.device

        correct = 0
        total_loss = 0
        n_data = dataloader.dataset.n_files()

        probability_sum = torch.zeros(n_data, self.config.n_class).to(device)
        file_labels = torch.zeros(n_data).to(device)

        for data, target, file_ids in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = self.model(data)

            for i, entry in enumerate(output):
                file_id = file_ids[i]
                probability_sum[file_id] += entry
                file_labels[file_id] = target[i]

        pred = probability_sum.max(1)[1]
        correct += pred.eq(file_labels).cpu().sum().item()

        accuracy = 100. * correct / n_data

        self.writer.add_scalar('loss/acc', accuracy, epoch, time.time())

        self.config.logger.info('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, n_data, accuracy))

        return accuracy

def train(args, train_folds, eval_folds):
    args.logger.debug(f'train folds: {train_folds}')
    args.logger.debug(f'eval folds: {eval_folds}')

    csv_path = f'{args.dataroot}/meta/esc50.csv'
    audio_dir = f'{args.dataroot}/audio'

    trainer = Trainer(args)

    if args.model_type == 'escconv':
        train_dataset = dataset.LogmelDataset(args, csv_path, audio_dir, train_folds, args.use_augment)
        eval_dataset = dataset.LogmelDataset(args, csv_path, audio_dir, eval_folds, False)
    elif args.model_type == 'envnet':
        train_dataset = dataset.EnvNetDataset(args, csv_path, audio_dir, train_folds)
        eval_dataset = dataset.EnvNetEvalDataset(args, csv_path, audio_dir, eval_folds)
    else:
        train_dataset = dataset.WaveDataset(args, csv_path, audio_dir, train_folds)
        eval_dataset = dataset.WaveDataset(args, csv_path, audio_dir, eval_folds)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        trainer.train(train_loader, epoch)
        accuracy = trainer.eval(eval_loader, epoch)
        trainer.update_epoch()

    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--dataroot', default='data', help='path to data')
    parser.add_argument('--name', default='default', help='name of training, used to model name, log dir name etc')
    parser.add_argument('--batch_size', type=int, default=8, help='epoch count')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--log_interval', type=int, default=1, help='log interval epochs')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--epochs', type=int, default=40, help='epoch count')
    parser.add_argument('--model_type', default=None, choices=['escconv', 'envnet', 'm5'], help='model type')
    parser.add_argument('--segmented', action='store_true')
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--use_augment', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--augment_mel_width_max', type=int, default=22)
    parser.add_argument('--augment_time_width_max', type=int, default=30)
    parser.add_argument('--n_class', type=int, default=50)
    parser.add_argument('--batchnorm', action='store_true')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda"
    args.device = torch.device(args.device_name)

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    if not args.cross_validation:
        train_folds = [1,2,3,4]
        eval_folds = [5]
        accuracy = train(args, train_folds, eval_folds)
        args.logger.info(f'accuracy: {accuracy}')
        sys.exit()

    accuracy_list = []
    n_folds = 5 # for dataset of ecs-50
    for fold_head in range(n_folds):
        train_folds = [(fold_head + i) % n_folds + 1 for i in range(n_folds - 1)]
        eval_folds = [(fold_head + n_folds - 1) % n_folds + 1]

        accuracy = train(args, train_folds, eval_folds)

        accuracy_list.append(accuracy)

    args.logger.info(f'accuracy list: {accuracy_list}')
    args.logger.info('accuracy average: {}'.format(sum(accuracy_list) / n_folds))

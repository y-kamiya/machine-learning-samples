import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from logzero import setup_logger
import torchaudio
import pandas as pd
import numpy as np
import os
import argparse
import time
import sys
import librosa
# import matplotlib.pyplot as plt
# import librosa.display

class Trainer():
    def __init__(self, config):
        self.config = config

        self.model = self.__create_model()
        self.optimizer = self.__create_optimizer(self.model)
        self.scheduler = self.__create_scheduler(self.optimizer)
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

        self.model.apply(self.__weights_init)

    def __weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def update_epoch(self):
        self.scheduler.step()

    def train(self, dataloader, epoch):
        self.model.train()
        device = self.config.device

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)

            data = data.requires_grad_()
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            n_processed_data = epoch * (batch_idx+1) * self.config.batch_size
            self.writer.add_scalar('loss/train', loss, n_processed_data, time.time())

            if batch_idx % self.config.log_interval == 0: #print training stats
                self.config.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.config.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))

    def __create_model(self):
        device = self.config.device
        if self.config.model_type == 'escconv':
            return Model_EscConv().to(device)
        
        return Model_M5().to(device)

    def __create_optimizer(self, model):
        if self.config.model_type == 'escconv':
            return optim.Adam(model.parameters(), lr = 0.0001)
        #     return optim.SGD(model.parameters(), momentum=0.9, lr=0.002, weight_decay=0.001, nesterov=True)
        
        return optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)

    def __create_scheduler(self, optimizer):
        if self.config.model_type == 'escconv':
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda _: 1.0)

        return optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.1)

    def eval(self, dataloader, epoch):
        self.model.eval()
        device = self.config.device

        correct = 0
        total_loss = 0
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = self.model(data)
            pred = output.max(1)[1]
            correct += pred.eq(target).cpu().sum().item()
            total_loss += F.nll_loss(output, target, reduction='sum')

        n_data = len(dataloader.dataset)
        self.writer.add_scalar('loss/eval', total_loss / n_data, epoch, time.time())
        self.writer.add_scalar('loss/acc', correct, epoch, time.time())

        correct_rate = 100. * correct / n_data
        self.config.logger.info('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, n_data, correct_rate))

        return correct_rate

class Model_EscConv(nn.Module):
    def __init__(self):
        super(Model_EscConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 80, kernel_size=(57,6), stride=1),
            nn.MaxPool2d((4,3), stride=(1,3)),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(1,3), stride=1),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.ReLU(True),
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(240, 5000),
            nn.Linear(3680, 5000),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(5000, 50),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Model_M5(nn.Module):
    def __init__(self):
        super(Model_M5, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 50)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2).permute(1, 0, 2).squeeze()

class BaseDataset(Dataset):
    def __init__(self, csv_path, audio_dir, folderList):
        super(BaseDataset, self).__init__()

        csvData = pd.read_csv(csv_path)

        self.file_names = []
        self.labels = []

        for i in range(0,len(csvData)):
            if csvData.iloc[i, 1] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 2])
                
        self.audio_dir = audio_dir

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.file_names)

class LogmelDataset(BaseDataset):
    def __init__(self, csv_path, audio_dir, folderList):
        super(LogmelDataset, self).__init__(csv_path, audio_dir, folderList)

        self.data = []

        frame_size = 512
        window_size = 1024
        frame_per_segment = 40
        segment_size = frame_size * frame_per_segment
        step_size = segment_size // 2

        transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, win_length=window_size, n_fft=window_size, hop_length=frame_size, n_mels=60)

        for index, name in enumerate(self.file_names):
            # if index > 5:
            #     break
            path = os.path.join(self.audio_dir, name)
            soundData, _ = librosa.load(path)
            tensor = torch.tensor(soundData)

            self.data.append((transform(tensor), self.labels[index]))

            # start = 0
            # clip = soundData[:, start:(start+segment_size)]
            # while clip.shape[1] == segment_size:
            #     mel = transform(clip)
            #     self.data.append((mel, self.labels[index]))
            #     start += step_size
            #     clip = soundData[:, start:(start+segment_size)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class WaveDataset(BaseDataset):
    def __getitem__(self, index):
        path = os.path.join(self.audio_dir, self.file_names[index])
        sound = torchaudio.load(path, out = None, normalization = True)
        soundData = sound[0].permute(1, 0)

        tempData = torch.zeros([160000, 1])
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]
        
        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--dataroot', default='data', help='path to data')
    parser.add_argument('--name', default='default', help='name of training, used to model name, log dir name etc')
    parser.add_argument('--batch_size', type=int, default=8, help='epoch count')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval epochs')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--epochs', type=int, default=40, help='epoch count')
    parser.add_argument('--model_type', default=None, choices=['escconv', 'm5'], help='model type')
    args = parser.parse_args()

    logger = setup_logger(name=__name__, level=args.loglevel)
    logger.info(args)
    args.logger = logger

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda"
    args.device = torch.device(args.device_name)

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    csv_path = f'{args.dataroot}/meta/esc50.csv'
    audio_dir = f'{args.dataroot}/audio'

    correct_rates = []
    n_folds = 5 # for dataset of ecs-50
    for fold_head in range(n_folds):
        train_folds = [(fold_head + i) % n_folds + 1 for i in range(n_folds - 1)]
        eval_folds = [(fold_head + n_folds - 1) % n_folds + 1]
        args.logger.debug(f'train folds: {train_folds}')
        args.logger.debug(f'eval folds: {eval_folds}')

        if args.model_type == 'escconv':
            train_dataset = LogmelDataset(csv_path, audio_dir, train_folds)
            eval_dataset = LogmelDataset(csv_path, audio_dir, eval_folds)
        else:
            train_dataset = WaveDataset(csv_path, audio_dir, train_folds)
            eval_dataset = WaveDataset(csv_path, audio_dir, eval_folds)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

        trainer = Trainer(args)

        rate = 0.0
        for epoch in range(1, args.epochs + 1):
            trainer.train(train_loader, epoch)
            rate = trainer.eval(eval_loader, epoch)
            trainer.update_epoch()
        correct_rates.append(rate)

    args.logger.info(f'correct rates: {correct_rates}')
    args.logger.info('correct rates average: {}'.format(sum(correct_rates) / n_folds))

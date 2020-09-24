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
import random
# import matplotlib.pyplot as plt
# import librosa
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

            self.optimizer.zero_grad()
            # plt.figure()
            # print(data[0].shape)
            # librosa.display.specshow(data[0][0].detach().numpy(), sr=22050, x_axis='time', y_axis='mel')
            # plt.imshow(data.log2()[0][0,:,:].detach().numpy()) 
            # plt.show()

            output = self.model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            n_processed_data = (epoch-1) * len(dataloader.dataset) + (batch_idx+1) * self.config.batch_size
            self.writer.add_scalar('loss/train', loss, n_processed_data, time.time())

            if batch_idx % self.config.log_interval == 0: #print training stats
                self.config.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.config.batch_size, len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss))

    def __create_model(self):
        device = self.config.device
        if self.config.model_type == 'escconv':
            return Model_EscConv(self.config).to(device)

        return Model_M5().to(device)

    def __create_optimizer(self, model):
        if self.config.model_type == 'escconv':
            if self.config.use_adam:
                return optim.Adam(model.parameters(), lr=self.config.lr)

            return optim.SGD(model.parameters(), momentum=0.9, lr=self.config.lr, weight_decay=0.001, nesterov=True)

        return optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=0.0001)

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
        accuracy = 100. * correct / n_data
        self.writer.add_scalar('loss/eval', total_loss / n_data, epoch, time.time())
        self.writer.add_scalar('loss/acc', accuracy, epoch, time.time())

        self.config.logger.info('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, n_data, accuracy))

        return accuracy

class Model_EscConv(nn.Module):
    def __init__(self, config):
        super(Model_EscConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 80, kernel_size=(57,6), stride=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d((4,3), stride=(1,3)),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=(1,3), stride=1),
            nn.BatchNorm2d(80),
            nn.ReLU(True),
            nn.MaxPool2d((1,3), stride=(1,3)),
        )

        n_features = 240 if config.segmented else 3680
        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(5000, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            # nn.Dropout(0.5),
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

        self.filenames = []
        self.labels = []

        for i in range(0,len(csvData)):
            if csvData.iloc[i, 1] in folderList:
                self.filenames.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 2])

        self.audio_dir = audio_dir

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.filenames)

class LogmelDataset(BaseDataset):
    def __init__(self, config, csv_path, audio_dir, folderList, apply_augment=True):
        super(LogmelDataset, self).__init__(csv_path, audio_dir, folderList)
        self.config = config
        self.apply_augment = apply_augment

        data_cache_path = os.path.join(self.config.dataroot, self.__data_filename(folderList))
        if not os.path.exists(data_cache_path):
            frame_size = 512
            window_size = 1024
            frame_per_segment = 41
            segment_size = frame_size * frame_per_segment
            step_size = segment_size // 2

            torchaudio.set_audio_backend('sox_io')
            transforms_mel = transforms.Compose([
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=22050, win_length=window_size, n_fft=window_size, hop_length=frame_size, n_mels=60, normalized=True),
                torchaudio.transforms.AmplitudeToDB(top_db=80.0),
            ])

            self.data = torch.empty(0)
            self.segment_labels = []
            for index, file in enumerate(self.filenames):
                path = os.path.join(self.audio_dir, file)
                tensor, _ = torchaudio.load(path)
                if not self.config.segmented:
                    data, label = self.__create_data(index, tensor, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                    continue

                start = 0
                clip = tensor[:, start:(start+segment_size-1)]
                while clip.shape[1] == segment_size-1:
                    data, label = self.__create_data(index, clip, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                    start += step_size
                    clip = tensor[:, start:(start+segment_size-1)]

            torch.save({
                'data': self.data,
                'label': self.segment_labels,
            }, data_cache_path)

        loaded = torch.load(data_cache_path, map_location=torch.device(self.config.device_name))
        self.data = loaded['data']
        self.segment_labels = loaded['label']

        mean = self.data.mean()
        std = self.data.std()

        self.transforms_norm = transforms.Compose([
            transforms.Normalize(mean, std),
        ])

    def __data_filename(self, folderList):
        folder_str = ''.join([str(n) for n in folderList])
        if self.config.segmented:
            return 'data{}.segmented.pth'.format(folder_str)
        return 'data{}.pth'.format(folder_str)

    def __create_data(self, index, wave, transforms):
        mel = transforms(wave)
        if torch.mean(mel) < -70.0:
            return None, None
        return mel, self.labels[index]

    def __augment(self, data):
        if not self.apply_augment:
            return data

        _, n_mel, n_time = data.shape

        mel_width = random.randint(0, self.config.augment_mel_width_max)
        mel_start = random.randint(0, n_mel - mel_width)
        mel_end = mel_start + mel_width

        time_width = random.randint(0, self.config.augment_time_width_max)
        time_start = random.randint(0, n_time - time_width)
        time_end = time_start + time_width

        # fig = plt.figure()
        # ax1 = fig.add_subplot(2, 1, 1)
        # ax1.imshow(data[0].detach().numpy(), cmap='gray')
        data[0][mel_start:mel_end, :] = 0
        data[0][:, time_start:time_end] = 0
        # ax2 = fig.add_subplot(2, 1, 2)
        # ax2.imshow(data[0].detach().numpy(), cmap='gray')
        # plt.show()
        return data

    def __getitem__(self, index):
        data = self.data[index]
        if self.config.normalized:
            data = self.transforms_norm(data)

        data = self.__augment(data)
        label = self.segment_labels[index]

        deltas = torchaudio.functional.compute_deltas(data)
        return torch.cat((data, deltas), dim=0), label

    def __len__(self):
        return len(self.data)

class WaveDataset(BaseDataset):
    def __getitem__(self, index):
        path = os.path.join(self.audio_dir, self.filenames[index])
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

def train(args, train_folds, eval_folds):
    args.logger.debug(f'train folds: {train_folds}')
    args.logger.debug(f'eval folds: {eval_folds}')

    csv_path = f'{args.dataroot}/meta/esc50.csv'
    audio_dir = f'{args.dataroot}/audio'

    trainer = Trainer(args)

    if args.model_type == 'escconv':
        train_dataset = LogmelDataset(args, csv_path, audio_dir, train_folds, args.use_augment)
        print(len(train_dataset))
        eval_dataset = LogmelDataset(args, csv_path, audio_dir, eval_folds, False)
        print(len(eval_dataset))
    else:
        train_dataset = WaveDataset(csv_path, audio_dir, train_folds)
        eval_dataset = WaveDataset(csv_path, audio_dir, eval_folds)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

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
    parser.add_argument('--model_type', default=None, choices=['escconv', 'm5'], help='model type')
    parser.add_argument('--segmented', action='store_true')
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--use_augment', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--augment_mel_width_max', type=int, default=22)
    parser.add_argument('--augment_time_width_max', type=int, default=30)
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

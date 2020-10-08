import torch
import torchaudio
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
import os
import random

class BaseDataset(Dataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(BaseDataset, self).__init__()
        self.config = config

        csvData = pd.read_csv(csv_path)

        self.filenames = []
        self.labels = []

        for i in range(0,len(csvData)):
            if csvData.iloc[i, 1] in folderList:
                self.filenames.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 2])

        self.audio_dir = audio_dir

    def n_files(self):
        return len(self.filenames)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.filenames)

class LogmelDataset(BaseDataset):
    def __init__(self, config, csv_path, audio_dir, folderList, apply_augment=True):
        super(LogmelDataset, self).__init__(config, csv_path, audio_dir, folderList)
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
            self.file_ids = []
            for index, file in enumerate(self.filenames):
                path = os.path.join(self.audio_dir, file)
                tensor, _ = torchaudio.load(path)
                if not self.config.segmented:
                    data, label = self.__create_data(index, tensor, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                        self.file_ids.append(index)
                    continue

                start = 0
                clip = tensor[:, start:(start+segment_size-1)]
                while clip.shape[1] == segment_size-1:
                    data, label = self.__create_data(index, clip, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                        self.file_ids.append(index)
                    start += step_size
                    clip = tensor[:, start:(start+segment_size-1)]

            torch.save({
                'data': self.data,
                'label': self.segment_labels,
                'file_ids': self.file_ids,
            }, data_cache_path)

        loaded = torch.load(data_cache_path, map_location=torch.device(self.config.device_name))
        self.data = loaded['data']
        self.segment_labels = loaded['label']
        self.file_ids = loaded['file_ids']

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

        data[0][mel_start:mel_end, :] = 0
        data[0][:, time_start:time_end] = 0
        return data

    def __getitem__(self, index):
        data = self.data[index]
        if self.config.normalized:
            data = self.transforms_norm(data)

        data = self.__augment(data)
        label = self.segment_labels[index]

        deltas = torchaudio.functional.compute_deltas(data)
        return torch.cat((data, deltas), dim=0), label, self.file_ids[index]

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
        return soundFormatted, self.labels[index], index

class EnvNetDataset(BaseDataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(EnvNetDataset, self).__init__(config, csv_path, audio_dir, folderList)

        trans = transforms.Compose([
            torchaudio.transforms.Resample(44100, 16000)
        ])
        self.segment_size = int(16000 * 1.5)

        self.sounds = []
        for i, file in enumerate(self.filenames):
            path = os.path.join(self.audio_dir, file)
            sound = torchaudio.load(path, out = None, normalization = True)
            resampled = trans(sound[0].squeeze())
            resampled /= torch.max(torch.abs(resampled))
            self.sounds.append(resampled)

    def is_enough_amplitude(self, data):
        return 0.2 < torch.max(torch.abs(data))

    def __getitem__(self, index):
        resampled = self.sounds[index]

        max_iter = 10000
        for i in range(max_iter):
            start = random.randint(0, len(resampled) - self.segment_size)
            data = resampled[start : start + self.segment_size]
            if self.is_enough_amplitude(data):
                break

        if i == max_iter - 1:
            self.config.logger.warning("valid section is not found: {}".format(path))

        return data.unsqueeze(0), self.labels[index], index

class EnvNetEvalDataset(EnvNetDataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(EnvNetEvalDataset, self).__init__(config, csv_path, audio_dir, folderList)

        step_size = int(16000 * 0.2)
        self.sounds_segmented = []
        self.labels_segmented = []
        self.file_ids = []

        for index, sound in enumerate(self.sounds):
            start = 0
            clip = sound[start:(start+self.segment_size)]
            while clip.shape[0] == self.segment_size:
                if self.is_enough_amplitude(clip):
                    self.sounds_segmented.append(clip.unsqueeze(0))
                    self.labels_segmented.append(self.labels[index])
                    self.file_ids.append(index)

                start += step_size
                clip = sound[start:(start+self.segment_size)]

    def __getitem__(self, index):
        return self.sounds_segmented[index], self.labels_segmented[index], self.file_ids[index]

    def __len__(self):
        return len(self.sounds_segmented)


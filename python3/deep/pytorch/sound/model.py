import torch
import torch.nn as nn
import torch.nn.functional as F

class EscConv(nn.Module):
    def __init__(self, config):
        super(EscConv, self).__init__()

        dropout = 1.0 if config.batchnorm else 0.5
        self.conv1 = self.__conv(2, 80, (57,6), (4,3), dropout)
        self.conv2 = self.__conv(80, 80, (1,3), (1,3))

        n_features = 240 if config.segmented else 3680
        self.fc1 = self.__linear(n_features, 5000, dropout)
        self.fc2 = self.__linear(5000, 50, dropout)

    def __conv(self, n_input, n_output, conv_kernel, pool_kernel, dropout=1.0):
        list = [
            nn.Conv2d(n_input, n_output, kernel_size=conv_kernel, stride=1),
        ]
        if dropout == 1.0:
            list.append(nn.BatchNorm2d(n_output))

        list.append(nn.ReLU(True))
        list.append(nn.MaxPool2d(pool_kernel, stride=(1,3)))

        if dropout != 1.0:
            list.append(nn.Dropout(dropout))

        return nn.Sequential(*list)

    def __linear(self, n_input, n_output, dropout=1.0):
        list = [
            nn.Linear(n_input, n_output),
        ]
        if dropout == 1.0:
            list.append(nn.BatchNorm1d(n_output))

        list.append(nn.ReLU(True))

        if dropout != 1.0:
            list.append(nn.Dropout(dropout))

        return nn.Sequential(*list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
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

class EnvNet(nn.Module):
    def __init__(self, config):
        super(EnvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(40)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(40, 40, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(40)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=160)

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(8, 13), stride=(1, 1)),
            nn.BatchNorm2d(50)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(50)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3))

        self.fc5 = nn.Sequential(
            nn.Linear(50 * 11 * 14, 4096),
            nn.Dropout(0.5)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(4096, config.n_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.output(x)


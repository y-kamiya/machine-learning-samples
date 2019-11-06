import torch
from torch import nn
from torch.nn import functional as F

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, z):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def latent_feature(self, x):
        raise NotImplementedError()

class VAE(Base):
    def __init__(self, config):
        super(VAE, self).__init__()

        dim = config.dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, dim)
        self.fc22 = nn.Linear(400, dim)
        self.fc3 = nn.Linear(dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def latent_feature(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        return self.reparameterize(mu, logvar)

class AE(Base):
    def __init__(self, config):
        super(AE, self).__init__()

        dim = config.dim

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, dim)
        self.fc3 = nn.Linear(dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), None

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        z, _ = self.encode(x.view(-1, 784))
        return self.decode(z), None, None

    def latent_feature(self, x):
        z, _ = self.encode(x.view(-1, 784))
        return z

class AE_CNN(Base):
    def __init__(self, config):
        super(AE_CNN, self).__init__()

        self.config = config
        dim = config.dim
        outer_dim = 26
        inner_dim = 36
        hidden_dim = 250

        self.encoder = nn.Sequential(
            nn.Conv2d(config.channel_size, outer_dim, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(outer_dim, inner_dim, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fcEnc1 = nn.Linear(576, hidden_dim)
        self.fcEnc2 = nn.Linear(hidden_dim, dim)

        self.fcDec = nn.Linear(dim, hidden_dim)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, inner_dim, kernel_size=4, stride=1)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(inner_dim, outer_dim, kernel_size=5, stride=1)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv0 = nn.ConvTranspose2d(outer_dim, config.channel_size, kernel_size=5, stride=1)

    def encode(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = F.relu(self.fcEnc1(x.view(batch_size, -1)))
        x = self.fcEnc2(x)
        return x, None

    def decode(self, z):
        x = F.relu(self.fcDec(z))
        x = x.unsqueeze(-1).unsqueeze(-1) # use hidden dim as channel num in deconvolution
        x = F.relu(self.deconv2(x))
        x = self.unpool2(x, self.__unpool_indices(2, x.shape))
        x = F.relu(self.deconv1(x))
        x = self.unpool1(x, self.__unpool_indices(2, x.shape))
        x = self.deconv0(x)
        return torch.sigmoid(x)

    def __unpool_indices(self, kernel_size, input_shape):
        dim = 2 * input_shape[-1]
        indices = torch.arange(0, dim*dim, step=kernel_size)
        indices = indices.view(dim, -1)[0::2]
        return indices.expand(input_shape).to(self.config.device)

    def forward(self, x):
        z, _ = self.encode(x)
        return self.decode(z), None, None

    def latent_feature(self, x):
        z, _ = self.encode(x)
        return z

class AE_VGG(Base):
    def __init__(self, config):
        super(AE_VGG, self).__init__()

        self.config = config
        dim = config.dim

        self.encoder = nn.Sequential(
            nn.Conv2d(config.channel_size, 16, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        hidden_dim = 1024
        self.fcEnc1 = nn.Linear(28 * 128, hidden_dim)
        self.fcEnc2 = nn.Linear(hidden_dim, dim)

        self.fcDec2 = nn.Linear(dim, hidden_dim)
        self.fcDec1 = nn.Linear(hidden_dim, 28 * 128)
        self.deconv5a = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv5b = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv5c = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv4a = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv4b = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.deconv4c = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv3a = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.deconv3b = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.deconv3c = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv2a = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.deconv2b = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.deconv1a = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)
        self.deconv1b = nn.ConvTranspose2d(16, self.config.channel_size, kernel_size=3, padding=1)

    def encode(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = F.relu(self.fcEnc1(x.view(batch_size, -1)))
        x = self.fcEnc2(x)
        return x, None

    def decode(self, z):
        batch_size = z.shape[0]
        x = F.relu(self.fcDec2(z))
        x = F.relu(self.fcDec1(x))
        x = x.view(batch_size, 128, 7, 4)
        x = F.max_unpool2d(x, self.__unpool_indices(2, x.shape), 2)
        x = self.deconv5c(self.deconv5b(self.deconv5a(x)))
        x = F.max_unpool2d(x, self.__unpool_indices(2, x.shape), 2)
        x = self.deconv4c(self.deconv4b(self.deconv4a(x)))
        x = F.max_unpool2d(x, self.__unpool_indices(2, x.shape), 2)
        x = self.deconv3c(self.deconv3b(self.deconv3a(x)))
        x = F.max_unpool2d(x, self.__unpool_indices(2, x.shape), 2)
        x = self.deconv2b(self.deconv2a(x))
        x = F.max_unpool2d(x, self.__unpool_indices(2, x.shape), 2)
        x = self.deconv1b(self.deconv1a(x))
        return torch.sigmoid(x)

    def __unpool_indices(self, kernel_size, input_shape):
        w = kernel_size * input_shape[-1]
        h = kernel_size * input_shape[-2]
        indices = torch.arange(0, w*h, step=kernel_size)
        indices = indices.view(h, -1)[0::kernel_size]
        return indices.expand(input_shape).to(self.config.device)

    def forward(self, x):
        z, _ = self.encode(x)
        return self.decode(z), None, None

    def latent_feature(self, x):
        z, _ = self.encode(x)
        return z




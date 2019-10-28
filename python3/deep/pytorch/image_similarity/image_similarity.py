from __future__ import print_function
import argparse
import sys
import os
import pickle
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import tabulate
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, dim):
        super(VAE, self).__init__()

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

class AE(nn.Module):
    def __init__(self, dim):
        super(AE, self).__init__()

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


class Trainer():
    def __init__(self, config):
        self.config = config

        self.model = self.__create_model()
        if config.model != None:
            self.model.load_state_dict(torch.load(config.model, map_location=config.device_name), strict=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config.dataroot, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=config.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config.dataroot, train=False, transform=transforms.ToTensor()),
            batch_size=config.batch_size, shuffle=True, **kwargs)

    def __create_model(self):
        dim = self.config.dim
        device = self.config.device
        if self.config.model_type == 'vae':
            return VAE(dim).to(device)

        return AE(dim).to(device)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        KLD = 0
        if mu != None and logvar != None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.config.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.__loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

        model_dir = '{}/model'.format(self.config.output_dir)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/epoch{}.pth'.format(model_dir, epoch))


    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.config.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.__loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(self.config.batch_size, 1, 28, 28)[:n]])
                    output_file = '{}/reconstruction_{}.png'.format(self.config.output_dir, str(epoch))
                    save_image(comparison.cpu(), output_file, nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def latent_feature(self):
        self.model.eval()
        with torch.no_grad():
            pickle_path = '{}/latent_feature.pickle'.format(self.config.output_dir)
            data = []
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as fp:
                    data = pickle.load(fp)

            filename = self.config.latent_feature.split('/')[-1]
            label = filename.split('.')[0]

            image = Image.open('{}/{}'.format(self.config.dataroot, self.config.latent_feature))
            x = transforms.functional.to_tensor(image).unsqueeze(0).to(self.config.device)

            z = self.model.latent_feature(x).squeeze()
            data.append({
                'path': self.config.latent_feature,
                'label': label,
                'feature': z,
            })

            with open(pickle_path, 'wb') as fp:
                pickle.dump(data, fp)

    def __cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def __img_tag(self, src):
        path = os.path.abspath('{}/{}'.format(self.config.dataroot, src))
        return '<img src="{}" width="50" height="50">'.format(path)

    def analyze(self):
        pickle_path = '{}/latent_feature.pickle'.format(self.config.output_dir)
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as fp:
                data = pickle.load(fp)

        table = []
        for row in data:
            d = []
            for col in data:
                d.append(self.__cos_sim(row['feature'].numpy(), col['feature'].numpy()))
            table.append(d)

        for i in range(len(data)):
            table[i].insert(0, self.__img_tag(data[i]['path']))

        headers = [self.__img_tag(entry['path']) for entry in data]
        headers.insert(0, "")

        html = tabulate.tabulate(table, headers, tablefmt='html', floatfmt='.3f', numalign='right')
        print(html)

    def sample_image(self):
        with torch.no_grad():
            sample = torch.randn(64, self.config.dim).to(self.config.device)
            sample = self.model.decode(sample).cpu()
            output_file = '{}/sample_{}.png'.format(self.config.output_dir, str(epoch))
            save_image(sample.view(64, 1, 28, 28), output_file)

    def plot(self):
        assert self.config.dim == 2, 'dimension of latent feature is wrong. use --dim 2'
        assert self.config.batch_size == 1, 'batch_size should be 1 to plot'

        self.model.eval()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        plotData = {}
        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                if i == 1000:
                    break
                data = data.to(self.config.device)
                z = self.model.latent_feature(data).squeeze()

                label = label.item()
                if label not in plotData:
                    plotData[label] = {'x':[], 'y':[]}

                plotData[label]['x'].append(z[0].item())
                plotData[label]['y'].append(z[1].item())

        for label, item in sorted(plotData.items(), key=lambda x:x[0]):
            plt.scatter(item['x'], item['y'], c=colors[label], label=label)

        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dim', type=int, default=20, metavar='N',
                        help='dimension of latent feature vector')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-type', default='ae',
                        help='model type')
    parser.add_argument('--model', default=None,
                        help='model path to load')
    parser.add_argument('--dataroot', default='./data',
                        help='where the data directory exists')
    parser.add_argument('--output-dir-name', default=None,
                        help='output directory name')
    parser.add_argument('--latent-feature', default='',
                        help='image file path to get latent feature')
    parser.add_argument('--analyze', action='store_true',
                        help='compare cosine similarity of images')
    parser.add_argument('--plot', action='store_true',
                        help='plot latent features as 2-dimensional graph')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device_name = "cuda" if args.cuda else "cpu"
    args.device = torch.device(args.device_name)

    args.output_dir = '{}/image_similarity'.format(args.dataroot)
    if args.output_dir_name != None:
        args.output_dir = '{}/{}'.format(args.output_dir, args.output_dir_name)

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    trainer = Trainer(args)

    if args.latent_feature:
        trainer.latent_feature()
        sys.exit()

    if args.analyze:
        trainer.analyze()
        sys.exit()

    if args.plot:
        trainer.plot()
        sys.exit()

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.sample_image()

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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default=None,
                    help='model path to load')
parser.add_argument('--dataroot', default='./data',
                    help='where the data directory exists')
parser.add_argument('--latent-feature', default='',
                    help='image file path to get latent feature')
parser.add_argument('--analyze', action='store_true',
                    help='compare cosine similarity of images')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device_name = "cuda" if args.cuda else "cpu"
device = torch.device(device_name)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.dataroot, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.dataroot, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
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

model = VAE().to(device)
if args.model != None:
    model.load_state_dict(torch.load(args.model, map_location=device_name), strict=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    model_dir = '{}/model'.format(args.output_dir)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), '{}/epoch{}.pth'.format(model_dir, epoch))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                output_file = '{}/reconstruction_{}.png'.format(args.output_dir, str(epoch))
                save_image(comparison.cpu(), output_file, nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def latent_feature():
    model.eval()
    with torch.no_grad():
        pickle_path = '{}/latent_feature.pickle'.format(args.output_dir)
        data = []
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as fp:
                data = pickle.load(fp)

        filename = args.latent_feature.split('/')[-1]
        label = filename.split('.')[0]

        image = Image.open('{}/{}'.format(args.dataroot, args.latent_feature))
        x = transforms.functional.to_tensor(image).unsqueeze(0).to(device)

        z = model.latent_feature(x).squeeze()
        data.append({
            'path': args.latent_feature,
            'label': label,
            'feature': z,
        })

        with open(pickle_path, 'wb') as fp:
            pickle.dump(data, fp)

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def img_tag(src):
    path = os.path.abspath('{}/{}'.format(args.dataroot, src))
    return '<img src="{}" width="50" height="50">'.format(path)

def analyze():
    pickle_path = '{}/latent_feature.pickle'.format(args.output_dir)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as fp:
            data = pickle.load(fp)

    table = []
    for row in data:
        d = []
        for col in data:
            d.append(cos_sim(row['feature'].numpy(), col['feature'].numpy()))
        table.append(d)

    for i in range(len(data)):
        table[i].insert(0, img_tag(data[i]['path']))

    headers = [img_tag(entry['path']) for entry in data]
    headers.insert(0, "")

    html = tabulate.tabulate(table, headers, tablefmt='html', floatfmt='.3f', numalign='right')
    print(html)

if __name__ == "__main__":
    args.output_dir = '{}/vae'.format(args.dataroot)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.latent_feature:
        latent_feature()
        sys.exit()

    if args.analyze:
        analyze()
        sys.exit()

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            output_file = '{}/sample_{}.png'.format(args.output_dir, str(epoch))
            save_image(sample.view(64, 1, 28, 28), output_file)

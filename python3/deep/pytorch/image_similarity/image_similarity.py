from __future__ import print_function
import argparse
import sys
import os
import time
import random
import pickle
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import tabulate
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import model

class MyDataset(Dataset):
    IMG_EXTENSIONS = ['.png']

    def __init__(self, config, phase=None, path=None):
        self.config = config

        if phase != None:
            dir = os.path.join(config.dataroot, phase)
            self.images = sorted(self.__make_dataset(dir))

        if path != None:
            self.images = [path]

    @classmethod
    def is_image_file(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __transform(self, param):
        list = []

        (x, y) = param['crop_pos']
        crop_width = self.config.crop_width
        crop_height = self.config.crop_height
        list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_width, y + crop_height))))

        list += [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))]

        return transforms.Compose(list)

    def __transform_param(self, image):
        load_w, load_h = image.size
        x = random.randint(0, np.maximum(0, load_w - self.config.crop_width))
        y = random.randint(0, np.maximum(0, load_h - self.config.crop_height))

        return {'crop_pos': (x, y)}

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path)

        param = self.__transform_param(image)
        transform = self.__transform(param)

        label = os.path.basename(path).split('_')[0]
        return (transform(image), label)

    def __len__(self):
        return len(self.images)

class Trainer():
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

        self.model = self.__create_model()
        self.model.apply(self.__weights_init)
        if config.model != None:
            self.model.load_state_dict(torch.load(config.model, map_location=config.device_name), strict=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.train_loader = self.__create_loader('train')
        self.test_loader = self.__create_loader('test')

    def __create_loader(self, phase):
        config = self.config
        if not config.use_mnist:
            dataset = MyDataset(self.config, phase)
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        is_train = True if phase == 'train' else False
        kwargs = {'num_workers': 1, 'pin_memory': True} if config.cuda else {}

        list = [
            transforms.Resize((config.crop_width, config.crop_height)),
            transforms.ToTensor(),
        ]
        return torch.utils.data.DataLoader(
            datasets.MNIST(config.dataroot, train=is_train, download=True, transform=transforms.Compose(list)),
            batch_size=config.batch_size, shuffle=True, **kwargs)

    def __create_model(self):
        device = self.config.device
        if self.config.model_type == 'vae':
            return model.VAE(self.config).to(device)

        if self.config.model_type == 'ae_cnn':
            if use_mnist:
                return model.AE_CNN_MNIST(self.config).to(device)
            return model.AE_CNN(self.config).to(device)

        if self.config.model_type == 'ae_vgg':
            return model.AE_VGG(self.config).to(device)

        return model.AE(self.config).to(device)

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __loss_function(self, recon_x, x, mu, logvar):
        dim = self.config.crop_height * self.config.crop_width
        BCE = F.binary_cross_entropy(recon_x.view(-1, dim), x.view(-1, dim), reduction='sum')

        KLD = 0
        if mu != None and logvar != None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def __loss_mse(self, recon_x, x):
        dim = self.config.crop_height * self.config.crop_width
        return F.mse_loss(recon_x.view(-1, dim), x.view(-1, dim), reduction='sum')

    def train(self, epoch):
        start_time = time.time()

        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        train_loss = 0
        train_loss_mse = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.config.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)

            loss = self.__loss_function(recon_batch, data, mu, logvar)
            loss_mse = self.__loss_mse(recon_batch, data)
            (loss + loss_mse).backward()
            train_loss += loss.item()
            train_loss_mse += loss_mse.item()

            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss MSE: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_dataset,
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data), loss_mse.item() / len(data)))

        time_epoch = time.time() - start_time
        time_all = time.time() - self.start_time
        print('====> Epoch: {} Average loss: {:.4f}\tAverage loss MSE: {:.4f}\tTime epoch: {:.3f}\tTime all: {:.3f}'.format(
              epoch, train_loss / n_dataset, train_loss_mse / n_dataset, time_epoch, time_all))
        self.writer.add_scalar('LossBCE/train', loss, epoch, time_all)
        self.writer.add_scalar('LossMSE/train', loss_mse, epoch, time_all)

        if epoch % self.config.save_interval == 0:
            self.save_model()

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        test_loss_mse = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.config.device)
                recon_batch, mu, logvar = self.model(data)

                test_loss += self.__loss_function(recon_batch, data, mu, logvar).item()
                test_loss_mse += self.__loss_mse(recon_batch, data).item()

                if i == 0 and epoch % self.config.save_interval == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(-1, self.config.channel_size,
                                                           self.config.crop_height, self.config.crop_width)[:n]])
                    output_file = '{}/reconstruction_{}.png'.format(self.config.output_dir, str(epoch))
                    save_image(comparison.cpu(), output_file, nrow=n)

        n_dataset = len(self.test_loader.dataset)
        test_loss /= n_dataset
        test_loss_mse /= n_dataset
        print('====> Test set loss: {:.4f}\tMSE: {:.4f}'.format(test_loss, test_loss_mse))
        self.writer.add_scalar('LossBCE/test', test_loss, epoch)
        self.writer.add_scalar('LossMSE/test', test_loss_mse, epoch)

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

            # image = Image.open('{}/{}'.format(self.config.dataroot, self.config.latent_feature))
            path = '{}/{}'.format(self.config.dataroot, self.config.latent_feature)
            dataset = MyDataset(self.config, None, path)
            loader = DataLoader(dataset, batch_size=1)

            _, (x, _) = next(enumerate(loader))

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

    def sample_image(self, epoch):
        with torch.no_grad():
            sample = torch.randn(16, self.config.dim).to(self.config.device)
            sample = self.model.decode(sample).cpu()
            output_file = '{}/sample_{}.png'.format(self.config.output_dir, str(epoch))
            save_image(sample.view(16, self.config.channel_size, self.config.crop_height, self.config.crop_width), output_file)
            print('save image to {}'.format(output_file))

    def save_model(self):
        model_dir = '{}/model'.format(self.config.output_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_path = '{}/epoch{}.pth'.format(model_dir, epoch)
        torch.save(self.model.state_dict(), model_path)
        print('save model to {}'.format(model_path))

    def __plot_with_sne(self):
        plotData = {'data':[], 'label':[]}
        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                if i == 1000:
                    break
                z = self.model.latent_feature(data).squeeze()

                plotData['data'].append(z.numpy())
                # plotData['label'].append(label.item())

        reduced = TSNE(n_components=2, random_state=0).fit_transform(plotData['data'])

        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, cmap='rainbow')
        # plt.scatter(reduced[:, 0], reduced[:, 1], c=plotData['label'], alpha=0.5, cmap='rainbow')
        plt.colorbar()
        plt.show()

    def plot(self):
        assert self.config.batch_size == 1, 'batch_size should be 1 to plot'

        self.model.eval()

        if self.config.dim != 2:
            self.__plot_with_sne()
            return

        plotData = {}
        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                if i == 1000:
                    break
                z = self.model.latent_feature(data).squeeze()

                label = label.item()
                if label not in plotData:
                    plotData[label] = {'x':[], 'y':[]}

                plotData[label]['x'].append(z[0].item())
                plotData[label]['y'].append(z[1].item())

        for label, item in sorted(plotData.items(), key=lambda x:x[0]):
            plt.scatter(item['x'], item['y'], label=label, alpha=0.5)

        plt.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--channel-size', type=int, default=1, help='input and output channel size')
    parser.add_argument('--dim', type=int, default=20, metavar='N', help='dimension of latent feature vector')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--save-interval', type=int, default=5, metavar='N', help='how many epochs to save model and sample image')
    parser.add_argument('--model-type', default='ae', help='model type')
    parser.add_argument('--model', default=None, help='model path to load')
    parser.add_argument('--dataroot', default='./data', help='where the data directory exists')
    parser.add_argument('--output-dir-name', default=None, help='output directory name')
    parser.add_argument('--crop-width', type=int, default=0, help='crop size')
    parser.add_argument('--crop-height', type=int, default=0, help='crop size, 0 means no crop')
    parser.add_argument('--latent-feature', default='', help='image file path to get latent feature')
    parser.add_argument('--analyze', action='store_true', help='compare cosine similarity of images')
    parser.add_argument('--plot', action='store_true', help='plot latent features as 2-dimensional graph')
    parser.add_argument('--use-mnist', action='store_true', help='use mnist dataset')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device_name = "cuda" if args.cuda else "cpu"
    args.device = torch.device(args.device_name)

    if args.output_dir_name == None:
        args.output_dir_name = '{}_dim{}'.format(args.model_type, args.dim)

    args.output_dir = '{}/output/{}'.format(args.dataroot, args.output_dir_name)
    args.tensorboard_log_dir = '{}/output/runs/{}'.format(args.dataroot, args.output_dir_name)

    # assert not os.path.exists(args.output_dir), 'output dir has already existed, change --output-dir-name'

    if args.use_mnist:
        args.crop_height = args.crop_width = 28
        if args.model_type == 'ae_vgg':
            args.crop_height = args.crop_width = 32

    if args.plot:
        args.batch_size = 1

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

        if epoch % args.save_interval == 0:
            trainer.sample_image(epoch)

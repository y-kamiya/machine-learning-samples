from __future__ import print_function
import argparse
import sys
import os
import shutil
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
import logzero
from logzero import logger
from tqdm import tqdm
from datetime import datetime

import model

class MyDataset(Dataset):
    IMG_EXTENSIONS = ['.png']

    def __init__(self, config, dir):
        self.config = config

        self.paths = []
        self.labels = []
        self.label_map = {}

        self.__make_dataset(dir)

    @classmethod
    def is_image_file(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    def __make_dataset(self, dir):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.paths.append(path)

                    label_str = fname.split('_')[0]
                    if label_str not in self.label_map:
                        self.label_map[label_str] = len(self.label_map)

                    self.labels.append(self.label_map[label_str])

    def __transform(self, param):
        list = []

        # list.append(transforms.Grayscale())
        # list.append(transforms.Resize((144, 256)))

        (x, y) = param['crop_pos']
        crop_width = self.config.crop_width
        crop_height = self.config.crop_height
        list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_width, y + crop_height))))

        list += [transforms.ToTensor()]
        # list += [transforms.ToTensor(),
        #          transforms.Normalize((0.5,), (0.5,))]

        return transforms.Compose(list)

    def __transform_param(self, image):
        load_w, load_h = image.size
        diff_w = load_w - self.config.crop_width
        diff_h = load_h - self.config.crop_height
        x = diff_w // 2
        y = diff_h // 2
        if not self.config.crop_center:
            x = random.randint(0, np.maximum(0, diff_w))
            y = random.randint(0, np.maximum(0, diff_h))

        return {'crop_pos': (x, y)}

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)

        param = self.__transform_param(image)
        transform = self.__transform(param)

        return {'data':transform(image), 'label':self.labels[index], 'path':self.paths[index]}

    def __len__(self):
        return len(self.paths)

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

    def __enumerate_loader(self, loader):
        if self.config.use_mnist:
            for i, (data, label) in enumerate(loader):
                yield (i, {'data':data, 'label':label})
        else:
            for i, dict in enumerate(loader):
                yield (i, dict)

    def __create_loader(self, phase):
        config = self.config
        if not config.use_mnist:
            dir = os.path.join(config.dataroot, phase)
            dataset = MyDataset(self.config, dir)
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
            if self.config.use_mnist:
                return model.AE_CNN_MNIST(self.config).to(device)
            return model.AE_CNN(self.config).to(device)

        if self.config.model_type == 'ae_vgg':
            return model.AE_VGG(self.config).to(device)

        if self.config.model_type == 'vae_vgg':
            return model.VAE_VGG(self.config).to(device)

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
        if mu is not None and logvar is not None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def __loss_mse(self, recon_x, x):
        dim = self.config.crop_height * self.config.crop_width
        return F.mse_loss(recon_x.view(-1, dim), x.view(-1, dim), reduction='sum')

    def __create_input(self, data):
        factor = self.config.noise_factor 
        if factor == 0:
            return data

        dev = self.config.noise_dev
        return (data + torch.empty_like(data).normal_(0, dev) * factor).clamp_(0,1)

    def train(self, epoch):
        start_time = time.time()

        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        train_loss = 0
        train_loss_mse = 0
        for batch_idx, batch in self.__enumerate_loader(self.train_loader):
            data = batch['data'].to(self.config.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(self.__create_input(data))

            loss = self.__loss_function(recon_batch, data, mu, logvar)
            loss_mse = self.__loss_mse(recon_batch, data)
            (loss + loss_mse).backward()
            train_loss += loss.item()
            train_loss_mse += loss_mse.item()

            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss MSE: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_dataset,
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data), loss_mse.item() / len(data)))

        time_epoch = time.time() - start_time
        time_all = time.time() - self.start_time
        logger.info('====> Epoch: {} Average loss: {:.4f}\tAverage loss MSE: {:.4f}\tTime epoch: {:.3f}\tTime all: {:.3f}'.format(
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
            for i, batch in self.__enumerate_loader(self.test_loader):
                data = batch['data'].to(self.config.device)
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
        logger.info('====> Test set loss: {:.4f}\tMSE: {:.4f}'.format(test_loss, test_loss_mse))
        self.writer.add_scalar('LossBCE/test', test_loss, epoch)
        self.writer.add_scalar('LossMSE/test', test_loss_mse, epoch)

    def __cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def __img_tag(self, src):
        path = os.path.abspath('{}'.format(src))
        return '<img src="{}" width="50" height="50">'.format(path)

    def analyze(self):
        self.model.eval()
        with torch.no_grad():
            dataset = MyDataset(self.config, self.config.analyze)
            loader = DataLoader(dataset, batch_size=1)

            data = []
            for batch in loader:
                z = self.model.latent_feature(batch['data']).squeeze()
                data.append({
                    'path': batch['path'][0],
                    'feature': z,
                })

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

            html = tabulate.tabulate(table, headers, tablefmt='html', floatfmt='.2f', numalign='right')
            print(html)

    def sample_image(self, epoch):
        with torch.no_grad():
            sample = torch.randn(16, self.config.dim).to(self.config.device)
            sample = self.model.decode(sample).cpu()
            output_file = '{}/sample_{}.png'.format(self.config.output_dir, str(epoch))
            save_image(sample.view(16, self.config.channel_size, self.config.crop_height, self.config.crop_width), output_file)
            logger.info('save image to {}'.format(output_file))

    def save_model(self):
        model_dir = '{}/model'.format(self.config.output_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_path = '{}/epoch{}.pth'.format(model_dir, epoch)
        torch.save(self.model.state_dict(), model_path)
        logger.info('save model to {}'.format(model_path))

    def __plot_with_sne(self):
        plotData = {'data':[], 'label':[]}
        with torch.no_grad():
            for i, batch in self.__enumerate_loader(self.test_loader):
                data = batch['data']
                label = batch['label']
                if i == 1000:
                    break
                z = self.model.latent_feature(data).squeeze()

                plotData['data'].append(z.numpy())
                plotData['label'].append(label.item())

        reduced = TSNE(n_components=2, random_state=0).fit_transform(plotData['data'])

        by_label = {}
        for i, label in enumerate(plotData['label']):
            if label not in by_label:
                by_label[label] = {'x':[], 'y':[]}
            by_label[label]['x'].append(reduced[i,0])
            by_label[label]['y'].append(reduced[i,1])

        by_label = sorted(by_label.items(), key=lambda x: x[0])

        label_map = {v:k for k, v in self.test_loader.dataset.label_map.items()}
        for label, data in by_label:
            plt.scatter(data['x'], data['y'], label=label_map[label], alpha=0.5, cmap='rainbow')

        plt.legend()
        plt.show()

    def plot(self):
        assert self.config.batch_size == 1, 'batch_size should be 1 to plot'

        self.model.eval()

        if self.config.dim != 2:
            self.__plot_with_sne()
            return

        plotData = {}
        with torch.no_grad():
            for i, batch in self.__enumerate_loader(self.test_loader):
                data = batch['data']
                label = batch['label']
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

    def save_latent_feature(self):
        self.model.eval()
        with torch.no_grad():
            data = {}

            for dir in self.config.latent_feature:
                dataset = MyDataset(self.config, dir)
                loader = DataLoader(dataset, batch_size=self.config.batch_size)

                logger.info(dir)
                for _, batch in tqdm(self.__enumerate_loader(loader)):
                    z = self.model.latent_feature(batch['data']).squeeze()

                    paths = batch['path']
                    for i in range(len(paths)):
                        data[paths[i]] = z[i]

            path = '{}/latent_feature.pickle'.format(args.output_dir)
            if os.path.exist(path):
                timestamp = datetime.now().strftime('%s')
                path = '{}/latent_feature_{}.pickle'.format(args.output_dir, timestamp)

            logger.info('save pickle to {}'.format(path))
            with open(path, 'wb') as fp:
                pickle.dump(data, fp)

    @classmethod
    def categorize_images(self, config):
        file = config.categorize
        assert os.path.exists(file), "latent_feature.pickle does not exist, please execute --latent-feature before"

        with open(file, 'rb') as fp:
            data = pickle.load(fp)

        processed = {}
        groups = []
        for src_i, (src_path, src_z) in tqdm(enumerate(data.items())):
            if src_i in processed:
                continue
            processed[src_i] = True
            groups.append([src_path])
            for tgt_i, (tgt_path, tgt_z) in enumerate(data.items()):
                if tgt_i in processed:
                    continue
                similarity = self.__cos_sim(self, src_z.numpy(), tgt_z.numpy())
                if config.categorize_threshold < similarity:
                    processed[tgt_i] = True
                    groups[-1].append(tgt_path)
                
        groups = sorted(groups, key=lambda x:-len(x))

        dir = './categorized'
        if os.path.exists(dir):
            shutil.rmtree(dir)

        os.mkdir(dir)

        for i, images in enumerate(groups):
            if not os.path.exists('{}/{}'.format(dir, i)):
                os.mkdir('{}/{}'.format(dir, i))

            for path in images:
                src = os.path.realpath(path)
                dst = '{}/{}/{}'.format(dir, i, os.path.basename(path))
                os.symlink(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check image similarity')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--channel-size', type=int, default=1, help='input and output channel size')
    parser.add_argument('--dim', type=int, default=20, metavar='N', help='dimension of latent feature vector')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--log-file', action='store_true', help='print log to file')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--save-interval', type=int, default=5, metavar='N', help='how many epochs to save model and sample image')
    parser.add_argument('--model-type', default='ae', help='model type')
    parser.add_argument('--model', default=None, help='model path to load')
    parser.add_argument('--dataroot', default='./data', help='where the data directory exists')
    parser.add_argument('--noise-dev', type=float, default=0.2, help='noise deviations for DAE')
    parser.add_argument('--noise-factor', type=float, default=0.0, help='noise factor for DAE')
    parser.add_argument('--output-dir-name', default=None, help='output directory name')
    parser.add_argument('--crop-center', action='store_true', help='crop center of the image')
    parser.add_argument('--crop-width', type=int, default=0, help='crop size')
    parser.add_argument('--crop-height', type=int, default=0, help='crop size, 0 means no crop')
    parser.add_argument('--analyze', default='', help='image dir to get latent feature')
    parser.add_argument('--plot', action='store_true', help='plot latent features as 2-dimensional graph')
    parser.add_argument('--use-mnist', action='store_true', help='use mnist dataset')
    parser.add_argument('--latent-feature', nargs='+', metavar='dirs...', help='get latent features of all images in dirs')
    parser.add_argument('--categorize', default=None, help='categorize images')
    parser.add_argument('--categorize_threshold', type=float, default=0.8, help='similarity threshold')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device_name = "cuda" if args.cuda else "cpu"
    args.device = torch.device(args.device_name)

    if args.output_dir_name == None:
        args.output_dir_name = '{}_dim{}'.format(args.model_type, args.dim)

    args.output_dir = '{}/output/{}'.format(args.dataroot, args.output_dir_name)
    args.tensorboard_log_dir = '{}/output/runs/{}'.format(args.dataroot, args.output_dir_name)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_file:
        filepath = '{}/default.log'.format(args.output_dir)
        logzero.logfile(filepath, disableStderrLogger=True)

    # assert not os.path.exists(args.output_dir), 'output dir has already existed, change --output-dir-name'

    if args.use_mnist:
        args.crop_height = args.crop_width = 28
        if args.model_type == 'ae_vgg':
            args.crop_height = args.crop_width = 32

    if args.plot:
        args.batch_size = 1

    if args.analyze:
        args.crop_center = True

    torch.manual_seed(args.seed)

    if args.categorize is not None:
        Trainer.categorize_images(args)
        sys.exit()

    trainer = Trainer(args)

    if args.analyze:
        trainer.analyze()
        sys.exit()

    if args.latent_feature:
        trainer.save_latent_feature()
        sys.exit()

    if args.plot:
        trainer.plot()
        sys.exit()

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)

        if epoch % args.save_interval == 0:
            trainer.sample_image(epoch)

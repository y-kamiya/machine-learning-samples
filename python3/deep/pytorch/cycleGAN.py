import sys
import argparse
import os.path
import random
import time
import numpy as np
import torch
import itertools
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down0 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)

        self.down1 = self.__down(64, 128)
        self.down2 = self.__down(128, 256)
        self.down3 = self.__down(256, 512)
        self.down4 = self.__down(512, 512)
        self.down5 = self.__down(512, 512)
        self.down6 = self.__down(512, 512)
        self.down7 = self.__down(512, 512, use_norm=False)

        self.up7 = self.__up(512, 512)
        self.up6 = self.__up(1024, 512)
        self.up5 = self.__up(1024, 512)
        self.up4 = self.__up(1024, 512)
        self.up3 = self.__up(1024, 256)
        self.up2 = self.__up(512, 128)
        self.up1 = self.__up(256, 64)

        self.up0 = nn.Sequential(
            self.__up(128, 3, use_norm=False),
            nn.Tanh(),
        )

    def __down(self, input, output, use_norm=True):
        layer = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1),
        ]
        if use_norm:
            layer.append(nn.BatchNorm2d(output))

        return nn.Sequential(*layer)

    def __up(self, input, output, use_norm=True):
        layer = [
            nn.ReLU(True),
            nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1),
        ]
        if use_norm:
            layer.append(nn.BatchNorm2d(output))

        return nn.Sequential(*layer)

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        y7 = self.up7(x7)
        y6 = self.up6(self.concat(x6, y7))
        y5 = self.up5(self.concat(x5, y6))
        y4 = self.up4(self.concat(x4, y5))
        y3 = self.up3(self.concat(x3, y4))
        y2 = self.up2(self.concat(x2, y3))
        y1 = self.up1(self.concat(x1, y2))
        y0 = self.up0(self.concat(x0, y1))

        return y0

    def concat(self, x, y):
        return torch.cat([x, y], dim=1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            self.__layer(64, 128),
            self.__layer(128, 256),
            self.__layer(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def __layer(self, input, output, stride=2):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return self.loss(prediction, target_tensor.expand_as(prediction))

class CycleGAN():
    def __init__(self, config):
        self.config = config
        self.netG_A = Generator().to(self.config.device)
        self.netG_B = Generator().to(self.config.device)
        self.netG_A.apply(self.__weights_init)
        self.netG_B.apply(self.__weights_init)
        if self.config.generator != None:
            self.netG_A.load_state_dict(torch.load(self.config.generator, map_location=self.config.device_name), strict=False)
            self.netG_B.load_state_dict(torch.load(self.config.generator, map_location=self.config.device_name), strict=False)

        self.netD_A = Discriminator().to(self.config.device)
        self.netD_B = Discriminator().to(self.config.device)
        self.netD_A.apply(self.__weights_init)
        self.netD_B.apply(self.__weights_init)
        if self.config.discriminator != None:
            self.netD_A.load_state_dict(torch.load(self.config.discriminator, map_location=self.config.device_name), strict=False)
            self.netD_B.load_state_dict(torch.load(self.config.discriminator, map_location=self.config.device_name), strict=False)

        self.optimizerG = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.criterionGAN = GANLoss().to(self.config.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionL1 = nn.L1Loss()

        self.training_start_time = time.time()
        self.append_log(config)
        self.append_log(self.netG_A)
        self.append_log(self.netD_A)

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def calc_lossD(self, netD, real, fake):
        pred_fake = netD(fake)
        lossD_fake = self.criterionGAN(pred_fake, False)

        pred_real = netD(real)
        lossD_real = self.criterionGAN(pred_real, True)

        return (lossD_fake + lossD_real) * 0.5

    def calc_lossG(self, netD, real, fake, rec, alpha):
        with torch.no_grad():
            pred_fake = netD(fake)

        lossG_GAN = self.criterionGAN(pred_fake, True)
        lossG_cycle = self.criterionCycle(rec, real)

        return lossG_GAN + lossG_cycle * alpha

    def train(self, data):
        self.realA = data['A'].to(self.config.device)
        self.realB = data['B'].to(self.config.device)

        fakeB = self.netG_A(self.realA)
        fakeA = self.netG_B(self.realB)
        recA = self.netG_B(fakeB)
        recB = self.netG_A(fakeA)

        # Discriminator
        lossD_A = self.calc_lossD(self.netD_A, self.realA, fakeA.detach())
        lossD_B = self.calc_lossD(self.netD_B, self.realB, fakeB.detach())

        self.optimizerD.zero_grad()
        lossD_A.backward()
        lossD_B.backward()
        self.optimizerD.step()

        # Generator
        lossG_A = self.calc_lossG(self.netD_A, self.realA, fakeA, recA, self.config.lambdaA)
        lossG_B = self.calc_lossG(self.netD_B, self.realB, fakeB, recB, self.config.lambdaB)

        self.optimizerG.zero_grad()
        lossG_A.backward()
        lossG_B.backward()
        self.optimizerG.step()

        # for log
        self.fakeB = fakeB
        self.lossG_A = lossG_A
        self.lossG_B = lossG_B
        self.lossD_A = lossD_A
        self.lossD_B = lossD_B

    def save(self, epoch):
        output_dir = self.config.output_dir
        torch.save(self.netG.state_dict(), '{}/cycle_G_epoch_{}'.format(output_dir, epoch))
        torch.save(self.netD.state_dict(), '{}/cycle_D_epoch_{}'.format(output_dir, epoch))

    def save_image(self, epoch):
        output_image = torch.cat([self.realA, self.fakeB, self.realB], dim=3)
        vutils.save_image(output_image,
                '{}/cycle_epoch_{}.png'.format(self.config.output_dir, epoch),
                normalize=True)

    def print_loss(self, epoch):
        elapsed_time = time.time() - self.training_start_time
        message = '(epoch: {}, time: {:.3f}, lossG_A: {:.3f}, lossG_B: {:.3f}, lossD_A: {:.3f}, lossD_B: {:.3f}) '.format(epoch, elapsed_time, self.lossG_A, self.lossG_B, self.lossD_A, self.lossD_B)

        self.append_log(message)

    def append_log(self, message):
        log_file = '{}/cycle.log'.format(self.config.output_dir)
        with open(log_file, "a") as log_file:
            log_file.write('{}\n'.format(message))  # save the message


class AlignedDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']

    def __init__(self, config):
        self.config = config
        
        dir = os.path.join(config.dataroot, config.phase)
        self.AB_paths = sorted(self.__make_dataset(dir))

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

        load_size = self.config.load_size
        list.append(transforms.Resize([load_size, load_size], Image.BICUBIC))

        (x, y) = param['crop_pos']
        crop_size = self.config.crop_size
        list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))))

        if param['flip']:
            list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))

        list += [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)

    def __transform_param(self):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        param = self.__transform_param()
        w, h = AB.size
        w2 = int(w / 2)
        transform = self.__transform(param)
        A = transform(AB.crop((0, 0, w2, h)))
        B = transform(AB.crop((w2, 0, w, h)))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=200, help='epoch count')
    parser.add_argument('--save_data_interval', type=int, default=10, help='save data interval epochs')
    parser.add_argument('--save_image_interval', type=int, default=10, help='save image interval epochs')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='epoch count')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--dataroot', default='data', help='path to the data directory')
    parser.add_argument('--output_dir', default='data', help='output directory')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--lambdaA', type=float, default=100.0, help='weight for cycle loss of A')
    parser.add_argument('--lambdaB', type=float, default=100.0, help='weight for cycle loss of B')
    parser.add_argument('--generator', help='file path to data for generator')
    parser.add_argument('--discriminator', help='file path to data for discriminator')
    args = parser.parse_args()
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda:0"
    args.device = torch.device(args.device_name)

    model = CycleGAN(args)
    dataset = AlignedDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        for i, data in enumerate(dataloader):
            model.train(data)

        if epoch % args.save_data_interval == 0:
            model.save(epoch)

        if epoch % args.save_image_interval == 0:
            model.save_image(epoch)

        if epoch % args.log_interval == 0:
            model.print_loss(epoch)

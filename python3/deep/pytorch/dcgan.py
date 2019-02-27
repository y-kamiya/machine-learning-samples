import sys
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.channels = 256
        self.rows = 3
        self.fc1 = nn.Linear(100, self.rows * self.rows * self.channels)

        self.conv1 = nn.ConvTranspose2d(self.channels,     self.channels // 2, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(self.channels // 2, self.channels // 4, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(self.channels // 4, self.channels // 8, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(self.channels // 8, 1, kernel_size=3, stride=3, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(self.channels)
        self.batch_norm2 = nn.BatchNorm2d(self.channels // 2)
        self.batch_norm3 = nn.BatchNorm2d(self.channels // 4)
        self.batch_norm4 = nn.BatchNorm2d(self.channels // 8)

    def forward(self, x):
        x = self.fc1(x).view(-1, self.channels, self.rows, self.rows)
        x = F.relu(self.batch_norm1(x))
        x = F.relu(self.batch_norm2(self.conv1(x)))
        x = F.relu(self.batch_norm3(self.conv2(x)))
        x = F.relu(self.batch_norm4(self.conv3(x)))
        return F.tanh(self.conv4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.channels = 256
        self.rows = 3

        self.conv1 = nn.Conv2d(1, self.channels // 8, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(self.channels // 8, self.channels // 4, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.channels // 4, self.channels // 2, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.channels // 2, self.channels, kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(self.channels, 1, kernel_size=self.rows)

        # self.fc = nn.Linear(self.rows * self.rows * self.channels, 1)

        # self.batch_norm1 = nn.BatchNorm2d(self.channels // 8)
        self.batch_norm2 = nn.BatchNorm2d(self.channels // 4)
        self.batch_norm3 = nn.BatchNorm2d(self.channels // 2)
        self.batch_norm4 = nn.BatchNorm2d(self.channels)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm3(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm4(self.conv4(x)))
        return F.sigmoid(self.conv5(x))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--save_interval', type=int, default=5, help='save interval epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='epoch count')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate for Adam')
    parser.add_argument('--dataroot', default='data', help='path to the data directory')
    parser.add_argument('--generator', help='file path to data for generator')
    parser.add_argument('--discriminator', help='file path to data for discriminator')
    parser.add_argument('--no_training', action='store_true', help='no training')
    args = parser.parse_args()
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu" if is_cpu else "cuda:0")

    dataset = dset.MNIST(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator().to(device)
    generator.apply(weights_init)
    if args.generator != None:
        generator.load_state_dict(torch.load(args.generator))

    if args.no_training:
        z = torch.randn(args.batch_size, 100, device=device)
        output = generator(z)
        vutils.save_image(output, 'data/generated.png', normalize=True)
        sys.exit()

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    if args.discriminator != None:
        discriminator.load_state_dict(torch.load(args.discriminator))

    loss = nn.BCELoss()

    optimizerG = optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.learning_rate)


    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader, 0):
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            discriminator.zero_grad()
            label = torch.full((batch_size,), 1, device=device)
            output = discriminator(real_data)
            loss_with_real = loss(output, label)
            loss_with_real.backward()
            D_x = output.mean().item()

            z = torch.randn(batch_size, 100, device=device)
            fake_data = generator(z)
            label.fill_(0)
            output = discriminator(fake_data.detach())
            loss_with_fake = loss(output, label)
            loss_with_fake.backward()

            optimizerD.step()
            D_G_z1 = output.mean().item()
            loss_discriminator = loss_with_real + loss_with_fake


            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake_data)
            loss_generator = loss(output, label)
            loss_generator.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()

            print("[{}/{}][{}/{}] Loss_D: {} Loss_G: {} D(x): {} D(G(x)): {} / {}".format(epoch, args.epochs, i, len(dataloader), loss_discriminator.item(), loss_generator.item(), D_x, D_G_z1, D_G_z2))

        if epoch % args.save_interval == 0:
            torch.save(generator.state_dict(), 'data/generator_epoch_{}.dat'.format(epoch))
            torch.save(discriminator.state_dict(), 'data/discriminator_epoch_{}.dat'.format(epoch))




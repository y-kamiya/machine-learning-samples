from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import apex
from torchvision import datasets, transforms, utils

BATCH_SIZE=32

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.linear1 = nn.Linear(7 * 7 * 64, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view([-1, 7 * 7 * 64])
        x = self.linear1(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.linear2(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--model', help='model to load')
    parser.add_argument('--estimate', help='image file path to estimate class')
    parser.add_argument('--fp16', action='store_true', help='run model with float16')
    args = parser.parse_args()
    print(args)

    is_cpu = args.cpu or not torch.cuda.is_available()
    device_name = 'cpu' if is_cpu else 'cuda'
    device = torch.device(device_name)

    model = Cnn().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if args.model != None:
        model.load_state_dict(torch.load(args.model, map_location=device_name), strict=False)

    if args.fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, 'O1')

    transform = transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.estimate != None:
        model.eval()
        input = transform(Image.open(args.estimate)).unsqueeze(0)
        output = F.softmax(model(input))
        print('estimated class: {}'.format(np.around(output.data.numpy(), decimals=3)))
        sys.exit()

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=transform),
            batch_size=BATCH_SIZE, shuffle=True)


    start = time.time()
    for i in range(args.epochs):
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            if args.fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            if batch_index % 100 == 0:
                print("elapsed: {:0.2f} sec, step {}, loss {:0.3f}".format(time.time() - start, batch_index * len(data), loss.item()))

        torch.save(model.state_dict(), 'data/mnist_cnn_{}.model'.format(i))


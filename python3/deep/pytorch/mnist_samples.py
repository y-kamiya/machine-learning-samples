import os
from PIL import Image
from torchvision import datasets

output_dir = './mnist_samples'
os.makedirs(output_dir, exist_ok=True)

sets = datasets.MNIST(output_dir, train=False, download=True)
for i, (data, label) in enumerate(sets):
    data.save('{}/{}.png'.format(output_dir, label))

    if 100 <= i:
        break



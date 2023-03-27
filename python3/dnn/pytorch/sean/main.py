import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.utils import save_image
from PIL import Image
from dataclasses import dataclass
from argparse_dataclass import ArgumentParser

from batchnorm import SynchronizedBatchNorm2d


@dataclass
class Config:
    device_name: str = "cuda"
    device: torch.device = torch.device("cuda")
    label_nc: int = 19
    model_path: str = "/tmp/models/latest_net_G.pth"


class SEAN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.generator = Generator(config).to(config.device)
        self.transform_image = self.build_transform()
        self.transform_label = self.build_transform(Image.Resampling.NEAREST, False)

        if config.model_path is not None:
            data = torch.load(config.model_path, map_location=config.device)
            for key in list(data.keys()):
                if "Spade.param_free_norm" in key:
                    data.pop(key)
                else:
                    data[key.replace("fc_mu", "per_style_convs.")] = data.pop(key)
            self.generator.load_state_dict(data)

    @torch.no_grad()
    def forward(self, image, label, style_codes):
        return self.generator(image, self.build_label(label), style_codes)

    @torch.no_grad()
    def encode(self, image, label):
        return self.generator.encode(image, self.build_label(label))

    def build_label(self, label):
        b, _, h, w = label.size()
        label = label.to(dtype=torch.int64)
        input_label = torch.zeros((b, self.config.label_nc, h, w)).to(device=self.config.device, dtype=torch.float)
        return input_label.scatter_(1, label, 1.0)

    def preprocess(self, image_pil, label_pil):
        image = self.transform_image(image_pil).unsqueeze(0).to(self.config.device)
        label = self.transform_label(label_pil).unsqueeze(0).to(self.config.device)
        label = label * 255.0
        label[label == 255] = 182
        return image, label

    def postprocess(self, tensor):
        image = tensor.detach().cpu().float().numpy()
        image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
        image = np.clip(image, 0, 255)
        return Image.fromarray(image.astype(np.uint8))

    def build_transform(self, method=Image.Resampling.BICUBIC, normalize=True):
        transform_list = [
            transforms.Lambda(lambda img: self.scale_width(img, 512, method)),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]

        if normalize:
            base = (0.5, 0.5, 0.5)
            transform_list.append(transforms.Normalize(base, base))

        return transforms.Compose(transform_list)

    def scale_width(self, img, width, method):
        w, h = img.size
        if w == width:
            return img
        h = int(width * h / w)
        return img.resize((width, h), method)


class Generator(nn.Module):
    CROP_SIZE = 256
    N_LAYERS = 5

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = self.CROP_SIZE // (2 ** self.N_LAYERS)

        self.Zencoder = Zencoder().to(config.device)

        self.fc = nn.Conv2d(config.label_nc, 1024, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.head_0 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.G_middle_0 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024, config.label_nc)
        self.up_0 = SPADEResnetBlock(1024, 512, config.label_nc)
        self.up_1 = SPADEResnetBlock(512, 256, config.label_nc)
        self.up_2 = SPADEResnetBlock(256, 128, config.label_nc)
        self.up_3 = SPADEResnetBlock(128, 64, config.label_nc, apply_style=False)
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, image, seg, style_codes):
        x = F.interpolate(seg, size=(self.dim, self.dim))
        x = self.fc(x)

        x = self.head_0(x, seg, style_codes)
        x = self.up(x)
        x = self.G_middle_0(x, seg, style_codes)
        x = self.G_middle_1(x, seg, style_codes)
        x = self.up(x)
        x = self.up_0(x, seg, style_codes)
        x = self.up(x)
        x = self.up_1(x, seg, style_codes)
        x = self.up(x)
        x = self.up_2(x, seg, style_codes)
        x = self.up(x)
        x = self.up_3(x, seg, style_codes)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

    def encode(self, image, seg):
        return self.Zencoder(image, seg)


class Zencoder(nn.Module):
    def __init__(self, fin=3, fout=512, n_hidden=32, n_kernel=3):
        super().__init__()

        sequence = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(fin, n_hidden, kernel_size=n_kernel, padding=0),
            nn.InstanceNorm2d(n_hidden),
            nn.LeakyReLU(0.2, False),
        ]

        for i in [0, 1]:
            n_in = n_hidden * (2 ** i)
            n_out = n_hidden * (2 ** (i+1))
            sequence += [
                nn.Conv2d(n_in, n_out, kernel_size=n_kernel, stride=2, padding=1),
                nn.InstanceNorm2d(n_out),
                nn.LeakyReLU(0.2, False),
            ]

        for i in [2]:
            n_in = n_hidden * (2 ** i)
            n_out = n_hidden * (2 ** (i+1))
            sequence += [
                nn.ConvTranspose2d(n_in, n_out, kernel_size=n_kernel, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(n_in / 2)),
                nn.LeakyReLU(0.2, False),
            ]

        sequence += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(fout // 2, fout, kernel_size=n_kernel, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, image, seg):
        x = self.model(image)

        b, image_nc, h, w = x.shape
        label_nc = seg.shape[1]
        seg = F.interpolate(seg, size=(h, w), mode="nearest")
        style_codes = torch.zeros((b, label_nc, image_nc), dtype=x.dtype, device=x.device)

        for i in range(b):
            for j in range(label_nc):
                mask = seg.bool()[i, j]
                mask_count = torch.sum(mask)
                if mask_count:
                    style_codes[i][j] = x[i].masked_select(mask).reshape(image_nc, mask_count).mean(1)

        return style_codes


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc, apply_style=True):
        super().__init__()
        self.learned_shortcut = fin != fout

        fmiddle = min(fin, fout)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.ace_0 = ACE(fin, label_nc, apply_style)
        self.ace_1 = ACE(fmiddle, label_nc, apply_style)
        if self.learned_shortcut:
            self.ace_s = ACE(fin, label_nc, apply_style)

    def forward(self, x, seg, style_codes):
        xs = x
        if self.learned_shortcut:
            xs = self.ace_s(xs, seg, style_codes)
            xs = self.conv_s(xs)

        x = self.ace_0(x, seg, style_codes)
        x = self.conv_0(self.activate(x))
        x = self.ace_1(x, seg, style_codes)
        x = self.conv_1(self.activate(x))

        return xs + x

    def activate(self, x):
        return F.leaky_relu(x, 2e-1)


class ACE(nn.Module):
    N_STYLES = 512

    def __init__(self, fin, label_nc, apply_style=True):
        super().__init__()
        self.apply_style = apply_style

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(fin), requires_grad=True)

        self.param_free_norm = SynchronizedBatchNorm2d(fin, affine=False)
        self.Spade = SPADE(label_nc, fin)
        if apply_style:
            self.per_style_convs = nn.ModuleList([nn.Linear(self.N_STYLES, self.N_STYLES) for _ in range(label_nc)])
            self.conv_gamma = nn.Conv2d(self.N_STYLES, fin, kernel_size=3, padding=1)
            self.conv_beta = nn.Conv2d(self.N_STYLES, fin, kernel_size=3, padding=1)

    def forward(self, x, seg, style_codes):
        noise = self.generate_noise(x)
        x_norm = self.param_free_norm(x + noise)

        b, _, h, w = x.shape
        seg = F.interpolate(seg, size=(h, w), mode="nearest")

        spade_gamma, spade_beta = self.Spade(seg)

        if not self.apply_style:
            return x_norm * (1 + spade_gamma) + spade_beta

        middle_avg = torch.zeros((b, self.N_STYLES, h, w), device=x_norm.device)
        for i in range(b):
            for j in range(seg.shape[1]):
                mask = seg.bool()[i, j]
                mask_count = torch.sum(mask)
                if mask_count:
                    mu = F.relu(self.per_style_convs[j](style_codes[i][j]))
                    mu = mu.reshape(self.N_STYLES, 1).expand(self.N_STYLES, mask_count)
                    middle_avg[i].masked_scatter_(mask, mu)

        style_gamma = self.conv_gamma(middle_avg)
        style_beta = self.conv_beta(middle_avg)

        a_gamma = torch.sigmoid(self.blending_gamma)
        a_beta = torch.sigmoid(self.blending_beta)

        gamma = a_gamma * style_gamma + (1 - a_gamma) * spade_gamma
        beta = a_beta * style_beta + (1 - a_beta) * spade_beta
        return x_norm * (1 + gamma) + beta

    def generate_noise(self, x):
        b, _, h, w = x.shape
        noise = torch.randn(b, w, h, 1).to(device=x.device)
        return (noise * self.noise_var).transpose(1, 3)


class SPADE(nn.Module):
    def __init__(self, fin, fout, n_hidden=128, n_kernel=3, n_padding=1):
        super().__init__()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(fin, n_hidden, kernel_size=n_kernel, padding=n_padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(n_hidden, fout, kernel_size=n_kernel, padding=n_padding)
        self.mlp_beta = nn.Conv2d(n_hidden, fout, kernel_size=n_kernel, padding=n_padding)

    def forward(self, x):
        x = self.mlp_shared(x)
        return self.mlp_gamma(x), self.mlp_beta(x)


if __name__ == "__main__":
    parser = ArgumentParser(Config)
    args = parser.parse_args()
    args.device = torch.device(args.device_name)
    print(args)

    model = SEAN(args)
    model.eval()

    image_pil = Image.open("image.png")
    label_pil = Image.open("label.png")

    image, label = model.preprocess(image_pil, label_pil)
    style_codes = model.encode(image, label)

    image_backhair_pil = Image.open("image_backhair.png")
    label_backhair_pil = Image.open("label_backhair.png")

    seg = np.where(np.array(label_backhair_pil) > 0, 7, label_pil)
    image_backhair, seg_image = model.preprocess(image_backhair_pil, Image.fromarray(np.uint8(seg)))
    output = model(image_backhair, seg_image, style_codes=style_codes)
    save_image(output[0], "test.png")

    image_np = output.squeeze(0).detach().cpu().float().numpy()
    image_np = (np.transpose(image_np, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save("test_normalized.png")


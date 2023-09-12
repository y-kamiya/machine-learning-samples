import os
import sys
import time
import json
import glob
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.transforms import so3_exp_map
from pytorch3d.io import load_objs_as_meshes
import pytorch3d.renderer as renderer

from plot_image_grid import image_grid


@dataclass
class Config():
    device: str = "cpu"
    dataroot: str = "data/mesh_tex"
    n_steps: int = 3
    plot_interval_steps: int = 1
    n_views: int = 40
    n_views_train: int = 2
    volume_size: int = 128
    volume_extent_world: float = 3.0


class Dataset():
    @classmethod
    def _camera(cls, i, R, T, d):
        return renderer.FoVPerspectiveCameras(device=d, R=R[None, i, ...], T=T[None, i, ...])

    def __init__(self, config: Config, mesh):
        self.config = config

        elev = torch.linspace(0, 360, config.n_views)
        azim = torch.linspace(-180, 180, config.n_views)
        R, T = renderer.look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        camera = self._camera(1, R, T, config.device)

        lights = renderer.PointLights(device=config.device, location=[[0.0, 0.0, -3.0]])
        raster_settings = renderer.RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        mesh_renderer = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=renderer.SoftPhongShader(
                device=config.device,
                cameras=camera,
                lights=lights
            )
        )

        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        meshes = mesh.extend(config.n_views)
        cameras = renderer.FoVPerspectiveCameras(device=config.device, R=R, T=T)
        tgt_images = mesh_renderer(meshes, cameras=cameras, lights=lights)

        # image_grid(tgt_images.cpu().numpy(), rows=4, cols=5, rgb=True)
        # plt.show()

        self.default_camera = camera
        self.default_lights = lights
        self.tgt_mesh = mesh
        self.tgt_images = tgt_images
        self.cameras = cameras
        self.center = center
        self.scale = scale


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()
        self.renderer = renderer
        self.voxel_size = voxel_size

        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))

    def forward(self, cameras):
        bs = cameras.R.shape[0]

        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)

        volumes = Volumes(
            densities=densities[None].expand(bs, *self.log_densities.shape),
            features=colors[None].expand(bs, *self.log_colors.shape),
            voxel_size=self.voxel_size,
        )
        return self.renderer(cameras=cameras, volumes=volumes)[0]


class Trainer():
    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        render_size = dataset.tgt_images.shape[1]

        self.volume_renderer = renderer.VolumeRenderer(
            raysampler=renderer.NDCMultinomialRaysampler(
                image_width=render_size,
                image_height=render_size,
                n_pts_per_ray=150,
                min_depth=0.1,
                max_depth=config.volume_extent_world,
            ),
            raymarcher=renderer.EmissionAbsorptionRaymarcher(),
        )
        self.volume_model = VolumeModel(
            self.volume_renderer,
            volume_size=[config.volume_size] * 3,
            voxel_size=config.volume_extent_world / config.volume_size,
        ).to(config.device)

        self.huber_loss = torch.nn.HuberLoss(delta=0.1)
        self.optimizer = torch.optim.Adam(self.volume_model.parameters(), lr=0.1)
        self.writer = SummaryWriter()

    def train(self):
        for i in tqdm(range(self.config.n_steps)):
            self.optimizer.zero_grad()

            loss = self.loss(i)
            loss.backward()
            self.optimizer.step()

        self.log_rotating_volume(n_frames=7*4)

    def loss(self, steps):
        n_views_train = config.n_views_train
        ds = self.dataset
        tgt_images = ds.tgt_images

        idxs = torch.randperm(self.config.n_views)[:n_views_train]
        cameras = renderer.FoVPerspectiveCameras(
            device=self.config.device,
            R=ds.cameras.R[idxs],
            T=ds.cameras.T[idxs],
            znear=ds.cameras.znear[idxs],
            zfar=ds.cameras.zfar[idxs],
            aspect_ratio=ds.cameras.aspect_ratio[idxs],
            fov=ds.cameras.fov[idxs],
        )

        pred_images = self.volume_model(cameras)

        silhouette = self.huber_loss(pred_images[..., 3], tgt_images[idxs, ..., 3])
        self.writer.add_scalar("loss/silhouette", silhouette, steps, time.time())

        rgb = self.huber_loss(pred_images[..., :3], tgt_images[idxs, ..., :3])
        self.writer.add_scalar("loss/rgb", rgb, steps, time.time())

        loss = silhouette + rgb
        self.writer.add_scalar("loss/all", loss, steps, time.time())

        if steps % self.config.plot_interval_steps == 0:
            self.log_image(steps, pred_images, tgt_images, idxs[0])

        return loss

    def log_image(self, steps, pred_images, tgt_images, idx):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.ravel()
        ax[0].imshow(self.clamp_and_detach(pred_images[0, ..., :3]))
        ax[1].imshow(self.clamp_and_detach(tgt_images[idx, ..., :3]))
        ax[2].imshow(self.clamp_and_detach(pred_images[0, ..., 3]))
        ax[3].imshow(self.clamp_and_detach(tgt_images[idx, ..., 3]))
        for ax_, title_ in zip(ax, ("pred rgb", "target rgb", "pred silhouette", "target silhouette")):
            ax_.grid("off")
            ax_.axis("off")
            ax_.set_title(title_)

        fig.canvas.draw()
        self.writer.add_figure("preds", fig, steps, True, time.time())

    def clamp_and_detach(self, x):
        return x.clamp(0.0, 1.0).cpu().detach().numpy()

    @torch.no_grad()
    def log_rotating_volume(self, n_frames=50):
        logRs = torch.zeros(n_frames, 3, device=self.config.device)
        logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=self.config.device)
        Rs = so3_exp_map(logRs)
        Ts = torch.zeros(n_frames, 3, device=self.config.device)
        Ts[:, 2] = 2.7
        cameras = self.dataset.cameras

        print('Generating rotating volume ...')
        frames = []
        for R, T in zip(tqdm(Rs), Ts):
            camera = FoVPerspectiveCameras(
                R=R[None],
                T=T[None],
                znear=cameras.znear[0],
                zfar=cameras.zfar[0],
                aspect_ratio=cameras.aspect_ratio[0],
                fov=cameras.fov[0],
                device=self.config.device,
            )
            frames.append(self.volume_model(camera)[..., :3].clamp(0.0, 1.0))

        rotating_volume_frames = torch.cat(frames)
        image_grid(rotating_volume_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=7, rgb=True, fill=True)
        self.writer.add_figure("final", plt.gcf(), config.n_steps, True, time.time())


if __name__ == "__main__":
    config = Config()

    mesh = load_objs_as_meshes([f"{config.dataroot}/cow.obj"], device=config.device)
    dataset = Dataset(config, mesh)

    trainer = Trainer(config, dataset)
    trainer.train()

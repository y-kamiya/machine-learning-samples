import os
import sys
import time
import json
import glob
import torch
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pytorch3d.utils import ico_sphere
import numpy as np
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.transforms import so3_exp_map
from pytorch3d.structures import Meshes
import pytorch3d.renderer as renderer

from plot_image_grid import image_grid


@dataclass
class Config():
    device: str = "cpu"
    dataroot: str = "data/mesh_tex"
    n_steps: int = 3000
    plot_interval_steps: int = 100
    n_views: int = 40
    n_views_train: int = 6
    volume_extent_world: float = 3.0
    render_size: int = 256
    n_pts_per_ray: int = 128
    n_rays_per_image: int = 750


class Dataset():
    def __init__(self, config: Config, mesh):
        self.config = config

        elev = torch.linspace(0, 360, config.n_views)
        azim = torch.linspace(-180, 180, config.n_views)
        R, T = renderer.look_at_view_transform(dist=2.7, elev=elev, azim=azim)

        lights = renderer.PointLights(device=config.device, location=[[0.0, 0.0, -3.0]])
        raster_settings = renderer.RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        meshes = mesh.extend(config.n_views)
        cameras = renderer.FoVPerspectiveCameras(device=config.device, R=R, T=T)
        mesh_renderer = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=renderer.SoftPhongShader(
                device=config.device,
                cameras=cameras,
                lights=lights
            )
        )
        tgt_images = mesh_renderer(meshes, cameras=cameras, lights=lights)

        # image_grid(tgt_images.cpu().numpy(), rows=4, cols=5, rgb=True)
        # plt.show()

        raster_settings_silhouette = renderer.RasterizationSettings(
            image_size=128, blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-4, faces_per_pixel=50
        )
        renderer_silhouette = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings_silhouette
            ),
            shader=renderer.SoftSilhouetteShader(),
        )
        silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

        self.R = R
        self.T = T
        self.tgt_mesh = mesh
        self.tgt_images = tgt_images
        self.tgt_silhouettes = (silhouette_images[..., 3] > 1e-4).float()
        self.cameras = cameras
        self.center = center
        self.scale = scale


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        super().__init__()
        self.register_buffer("frequencies", omega0 * (2.0 ** torch.arange(n_harmonic_functions)))

    def forward(self, x):
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden=256):
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        embedding_dim = n_harmonic_functions * 2 * 3

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Softplus(beta=10.0),
        )

        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden + embedding_dim, n_hidden),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden, 3),
            torch.nn.Sigmoid(),
        )

        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 1),
            torch.nn.Softplus(beta=10.0),
        )

        self.density_layer[0].bias.data[0] = -1.5

    def densities(self, features):
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def colors(self, features, rays_directions):
        spacial_size = features.shape[:-1]

        rays_directions_normed = F.normalize(rays_directions, dim=-1)
        rays_embedding = self.harmonic_embedding(rays_directions_normed)
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spacial_size, rays_embedding.shape[-1]
        )

        color_layer_input = torch.cat((features, rays_embedding_expand), dim=-1)
        return self.color_layer(color_layer_input)

    def forward(self, ray_bundle: renderer.RayBundle, **kwargs):
        rays_points_world = renderer.ray_bundle_to_ray_points(ray_bundle)
        embeds = self.harmonic_embedding(rays_points_world)
        features = self.mlp(embeds)
        rays_densities = self.densities(features)
        rays_colors = self.colors(features, ray_bundle.directions)
        return rays_densities, rays_colors

    def batch_forward(self, ray_bundle: renderer.RayBundle, n_batches: int = 16, **kwargs):
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        origins = ray_bundle.origins
        spatial_size = [*origins.shape[:-1], n_pts_per_ray]

        tot_samples = origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        batch_outputs = [
            self.forward(renderer.RayBundle(
                origins=origins.view(-1, 3)[idx],
                directions=ray_bundle.directions.view(-1, 3)[idx],
                lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[idx],
                xys=None,
            )) for idx in batches
        ]
        rays_densities, rays_colors = [
            torch.cat(
                [output[i] for output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for i in (0, 1)
        ]
        return rays_densities, rays_colors


class Trainer():
    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        raysampler_grid = renderer.NDCMultinomialRaysampler(
            image_height=config.render_size,
            image_width=config.render_size,
            n_pts_per_ray=config.n_pts_per_ray,
            min_depth=0.1,
            max_depth=config.volume_extent_world,
        )
        raysampler_mc = renderer.MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=config.n_rays_per_image,
            n_pts_per_ray=config.n_pts_per_ray,
            min_depth=0.1,
            max_depth=config.volume_extent_world,
        )
        raymarcher = renderer.EmissionAbsorptionRaymarcher()

        self.renderer_grid = renderer.ImplicitRenderer(
            raysampler=raysampler_grid,
            raymarcher=raymarcher,
        ).to(config.device)

        self.renderer_mc = renderer.ImplicitRenderer(
            raysampler=raysampler_mc,
            raymarcher=raymarcher,
        ).to(config.device)

        self.tgt_cameras = self.dataset.cameras.to(config.device)
        self.tgt_images = self.dataset.tgt_images.to(config.device)
        self.tgt_silhouettes = self.dataset.tgt_silhouettes.to(config.device)

        self.huber_loss = torch.nn.HuberLoss(delta=0.1)
        self.nerf = NeuralRadianceField().to(config.device)
        self.optimizer = torch.optim.Adam(self.nerf.parameters(), lr=1e-3)
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

        idxs = torch.randperm(self.config.n_views)[:n_views_train]
        cameras = renderer.FoVPerspectiveCameras(
            device=self.config.device,
            R=self.tgt_cameras.R[idxs],
            T=self.tgt_cameras.T[idxs],
            znear=self.tgt_cameras.znear[idxs],
            zfar=self.tgt_cameras.zfar[idxs],
            aspect_ratio=self.tgt_cameras.aspect_ratio[idxs],
            fov=self.tgt_cameras.fov[idxs],
        )

        pred_images, sampled_rays = self.renderer_mc(
            cameras=cameras,
            volumetric_function=self.nerf,
        )

        rgb_at_rays = self.sample_images_at_mc_locs(
            self.tgt_images[idxs, ..., :3],
            sampled_rays.xys,
        )
        rgb = self.huber_loss(pred_images[..., :3], rgb_at_rays)
        self.writer.add_scalar("loss/rgb", rgb, steps, time.time())

        silhouette_at_rays = self.sample_images_at_mc_locs(
            self.tgt_silhouettes[idxs, ..., None],
            sampled_rays.xys,
        )
        silhouette = self.huber_loss(pred_images[..., 3].unsqueeze(-1), silhouette_at_rays)
        self.writer.add_scalar("loss/silhouette", silhouette, steps, time.time())

        loss = silhouette + rgb
        self.writer.add_scalar("loss/all", loss, steps, time.time())

        if steps % self.config.plot_interval_steps == 0:
            self.log_image(steps)

        return loss

    def sample_images_at_mc_locs(self, images, sampled_rays):
        bs = images.shape[0]
        dim = images.shape[-1]
        spatial_size = sampled_rays.shape[1:-1]

        images_sampled = F.grid_sample(
            images.permute(0, 3, 1, 2),
            -sampled_rays.view(bs, -1, 1, 2),
            align_corners=True,
        )
        return images_sampled.permute(0, 2, 3, 1).view(bs, *spatial_size, dim)

    def log_image(self, steps):
        camera_idx = 0
        with torch.no_grad():
            pred_images, _ = self.renderer_grid(
                cameras=self.tgt_cameras[camera_idx],
                volumetric_function=self.nerf.batch_forward,
            )

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax = ax.ravel()
        ax[0].imshow(self.clamp_and_detach(pred_images[0, ..., :3]))
        ax[1].imshow(self.clamp_and_detach(self.tgt_images[camera_idx, ..., :3]))
        ax[2].imshow(self.clamp_and_detach(pred_images[0, ..., 3]))
        ax[3].imshow(self.clamp_and_detach(self.tgt_silhouettes[camera_idx, ...]))
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

        print('Generating rotating volume ...')
        frames = []
        for R, T in zip(tqdm(Rs), Ts):
            camera = renderer.FoVPerspectiveCameras(
                R=R[None],
                T=T[None],
                znear=self.tgt_cameras.znear[0],
                zfar=self.tgt_cameras.zfar[0],
                aspect_ratio=self.tgt_cameras.aspect_ratio[0],
                fov=self.tgt_cameras.fov[0],
                device=self.config.device,
            )
            frames.append(
                self.renderer_grid(
                    cameras=camera,
                    volumetric_function=self.nerf.batch_forward,
                )[0][..., :3]
            )

        rotating_volume_frames = torch.cat(frames)
        image_grid(rotating_volume_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=7, rgb=True, fill=True)
        self.writer.add_figure("final", plt.gcf(), config.n_steps, True, time.time())


if __name__ == "__main__":
    config = Config()
    torch.manual_seed(1)

    mesh = load_objs_as_meshes([f"{config.dataroot}/cow.obj"], device=config.device)
    dataset = Dataset(config, mesh)

    trainer = Trainer(config, dataset)
    trainer.train()

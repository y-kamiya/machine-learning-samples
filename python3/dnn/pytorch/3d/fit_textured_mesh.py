import os
import time
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
from pytorch3d.structures import Meshes
import pytorch3d.renderer as renderer

from plot_image_grid import image_grid


@dataclass
class Config():
    device: str = "cpu"
    dataroot: str = "data/mesh_tex"
    n_steps: int = 2000
    n_samples: int = 5000
    w_silhouette: float = 1.0
    w_rgb: float = 1.0
    w_edge: float = 1.0
    w_normal: float = 0.01
    w_laplacian: float = 1.0
    plot_interval_steps: int = 1
    output_file: str = "output.obj"

    n_views: int = 20
    n_views_train: int = 2
    blur_radius_sigma: float = 1e-4


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
        tgt_cameras = [self._camera(i, R, T, config.device) for i in range(config.n_views)]

        # image_grid(tgt_images.cpu().numpy(), rows=4, cols=5, rgb=True)
        # plt.show()

        self.R = R
        self.T = T
        self.default_camera = camera
        self.default_lights = lights
        self.tgt_mesh = mesh
        self.tgt_images = tgt_images
        self.tgt_cameras = tgt_cameras
        self.center = center
        self.scale = scale

    def pick_cameras(self, indexes):
        return [self.tgt_cameras[i] for i in indexes]



class Trainer():
    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        self.src_mesh = ico_sphere(4, self.config.device)
        verts_shape = self.src_mesh.verts_packed().shape
        self.deform_verts = torch.full(verts_shape, 0.0, device=config.device, requires_grad=True)
        self.verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=config.device, requires_grad=True)

        self.optimizer = torch.optim.SGD([self.deform_verts, self.verts_rgb], lr=1.0, momentum=0.9)
        self.writer = SummaryWriter()

        self.renderer_textured = renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(
                cameras=dataset.default_camera,
                raster_settings=renderer.RasterizationSettings(
                    image_size=128,
                    blur_radius=np.log(1. / 1e-4 - 1.) * config.blur_radius_sigma,
                    faces_per_pixel=50,
                    perspective_correct=False,
                )
            ),
            shader=renderer.SoftPhongShader(
                device=config.device,
                cameras=dataset.default_camera,
                lights=dataset.default_lights,
            )
        )

    def train(self):
        for i in tqdm(range(self.config.n_steps)):
            self.optimizer.zero_grad()

            mesh = self.src_mesh.offset_verts(self.deform_verts)
            mesh.textures = renderer.TexturesVertex(verts_features=self.verts_rgb)
            loss = self.loss(i, mesh)
            loss.backward()
            self.optimizer.step()

            if i % self.config.plot_interval_steps == 0:
                self.visualize_prediction(i, mesh, title=f"steps: {i}")

        self.visualize_prediction(i, mesh, title=f"steps: {i}")
        self.postprocess(mesh)

    def postprocess(self, mesh):
        verts, faces = mesh.get_mesh_verts_faces(0)
        verts = verts * self.dataset.scale + self.dataset.center
        save_obj(self.config.output_file, verts, faces)

    def loss(self, steps, mesh):
        n_views_train = config.n_views_train
        ds = self.dataset
        tgt_images = ds.tgt_images
        lights = ds.default_lights

        idxs = torch.randperm(self.config.n_views)[:n_views_train]
        cameras = renderer.FoVPerspectiveCameras(device=self.config.device, R=ds.R[idxs], T=ds.T[idxs])
        pred_images = self.renderer_textured(mesh.extend(n_views_train), cameras=cameras, lights=lights)

        silhouette = F.mse_loss(pred_images[..., 3], tgt_images[idxs, ..., 3])
        self.writer.add_scalar("loss/silhouette", silhouette, steps, time.time())

        rgb = F.mse_loss(pred_images[..., :3], tgt_images[idxs, ..., :3])
        self.writer.add_scalar("loss/rgb", rgb, steps, time.time())

        edge = mesh_edge_loss(mesh)
        self.writer.add_scalar("loss/edge", edge, steps, time.time())

        normal = mesh_normal_consistency(mesh)
        self.writer.add_scalar("loss/normal", normal, steps, time.time())

        laplacian = mesh_laplacian_smoothing(mesh, method="uniform")
        self.writer.add_scalar("loss/laplacian", laplacian, steps, time.time())

        cfg = self.config
        loss = cfg.w_silhouette * silhouette + cfg.w_rgb * rgb + cfg.w_edge * edge * cfg.w_normal * normal + cfg.w_laplacian * laplacian
        self.writer.add_scalar("loss/all", loss, steps, time.time())

        return loss

    def visualize_prediction(self, steps, mesh, title=''):
        with torch.no_grad():
            predicted_images = self.renderer_textured(mesh)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_images[0, ..., :3].cpu().detach().numpy())

        plt.subplot(1, 2, 2)
        plt.imshow(self.dataset.tgt_images[1, ..., :3].cpu().detach().numpy())
        plt.title(title)
        plt.axis("off")

        self.writer.add_figure("preds", plt.gcf(), steps, True, time.time())


if __name__ == "__main__":
    config = Config()

    mesh = load_objs_as_meshes([f"{config.dataroot}/cow.obj"], device=config.device)
    dataset = Dataset(config, mesh)

    trainer = Trainer(config, dataset)
    trainer.train()

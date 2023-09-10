import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


@dataclass
class Config():
    device: str = "cpu"
    n_steps: int = 2000
    n_samples: int = 5000
    w_chamfer: float = 1.0
    w_edge: float = 1.0
    w_normal: float = 0.01
    w_laplacian: float = 0.1
    plot_interval_steps: int = 250
    output_file: str = "output.obj"


class Trainer():
    def __init__(self, config: Config):
        self.config = config

        self.src_mesh = ico_sphere(4, self.config.device)
        self.deform_verts = torch.full(self.src_mesh.verts_packed().shape, 0.0, device=config.device, requires_grad=True)

        self.optimizer = torch.optim.SGD([self.deform_verts], lr=1.0, momentum=0.9)
        self.writer = SummaryWriter()

    def preprocess(self, tgt_verts, tgt_faces_idx):
        self.center = tgt_verts.mean(0)
        verts = tgt_verts - self.center
        self.scale = max(verts.abs().max(0)[0])
        verts = verts / self.scale
        return Meshes(verts=[verts], faces=[tgt_faces_idx])

    def postprocess(self, mesh):
        verts, faces = mesh.get_mesh_verts_faces(0)
        verts = verts * self.scale + self.center
        save_obj(self.config.output_file, verts, faces)

    def train(self, tgt_verts, tgt_faces):
        tgt_verts = tgt_verts.to(self.config.device)
        tgt_faces_idx = tgt_faces.verts_idx.to(self.config.device)
        tgt_mesh = self.preprocess(tgt_verts, tgt_faces_idx)

        for i in range(self.config.n_steps):
            self.optimizer.zero_grad()

            mesh = self.src_mesh.offset_verts(self.deform_verts)
            sample_src = sample_points_from_meshes(mesh, self.config.n_samples)
            sample_tgt = sample_points_from_meshes(tgt_mesh, self.config.n_samples)

            loss = self.loss(i, mesh, sample_src, sample_tgt)
            loss.backward()
            self.optimizer.step()

            if i % self.config.plot_interval_steps == 0:
                self.plot_pointcloud(i, mesh, title=f"steps: {i}")

        self.plot_pointcloud(self.config.n_steps, mesh, title=f"steps: {i}")
        self.postprocess(mesh)

    def loss(self, steps, mesh, sample_src, sample_tgt):
        chamfer, _ = chamfer_distance(sample_tgt, sample_src)
        self.writer.add_scalar("loss/chamfer", chamfer, steps, time.time())

        edge = mesh_edge_loss(mesh)
        self.writer.add_scalar("loss/edge", edge, steps, time.time())

        normal = mesh_normal_consistency(mesh)
        self.writer.add_scalar("loss/normal", normal, steps, time.time())

        laplacian = mesh_laplacian_smoothing(mesh, method="uniform")
        self.writer.add_scalar("loss/laplacian", laplacian, steps, time.time())

        cfg = self.config
        loss = cfg.w_chamfer * chamfer + cfg.w_edge * edge * cfg.w_normal * normal + cfg.w_laplacian * laplacian
        self.writer.add_scalar("loss/all", loss, steps, time.time())

        return loss

    def plot_pointcloud(self, steps, mesh, title=""):
        points = sample_points_from_meshes(mesh, 2000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        ax.view_init(190, 30)
        fig.add_axes(ax)
        self.writer.add_figure("mesh", plt.gcf(), steps, True, time.time())
        # plt.show()


if __name__ == "__main__":
    verts_org, faces_org, aux_org = load_obj("dolphin.obj")

    config = Config()
    trainer = Trainer(config)
    trainer.train(verts_org, faces_org)

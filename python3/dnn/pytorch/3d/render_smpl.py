import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.structures import Meshes
import pytorch3d.renderer as renderer
from pytorch3d.io import load_obj


device = torch.device("cpu")
    

def create_renderer():
    R, T = renderer.look_at_view_transform(1.0, 0, 0) 
    cameras = renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = renderer.RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    lights = renderer.PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    return renderer.MeshRenderer(
        rasterizer=renderer.MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=renderer.SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )


DATA_DIR = "./data"
tex_path = os.path.join(DATA_DIR, "smpl/smpl_uv_20200910.png")
obj_path = os.path.join(DATA_DIR, "smpl/smpl_uv.obj")


with Image.open(tex_path) as image:
    np_image = np.asarray(image.convert("RGB")).astype(np.float32)
tex = torch.from_numpy(np_image / 255.)[None].to(device)

verts, faces, aux = load_obj(obj_path)
texture = renderer.TexturesUV(maps=tex, faces_uvs=[faces.textures_idx], verts_uvs=[aux.verts_uvs])
mesh = Meshes([verts], [faces.verts_idx], texture)

renderer = create_renderer()
images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
plt.show()

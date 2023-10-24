import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# libraries for reading data from files
from scipy.io import loadmat
from PIL import Image
import pickle

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV
)


device = torch.device("cpu")
    
# Set paths
DATA_DIR = "./data"
data_filename = os.path.join(DATA_DIR, "densepose/UV_Processed.mat")
tex_filename = os.path.join(DATA_DIR,"densepose/texture_from_SURREAL.png")
verts_filename = os.path.join(DATA_DIR, "densepose/smpl_model.pkl")


ALP_UV = loadmat(data_filename)
verts = torch.from_numpy((ALP_UV["All_vertices"]).astype(int)).squeeze().to(device) # (7829,)
U = torch.Tensor(ALP_UV['All_U_norm']).to(device) # (7829, 1)
V = torch.Tensor(ALP_UV['All_V_norm']).to(device) # (7829, 1)
faces = torch.from_numpy((ALP_UV['All_Faces'] - 1).astype(int)).to(device)  # (13774, 3)
face_indices = torch.Tensor(ALP_UV['All_FaceIndices']).squeeze()  # (13774,)


def convert_denpose_verts_uv():
    # Map each face to a (u, v) offset
    offset_per_part = {}
    already_offset = set()
    cols, rows = 4, 6
    for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
            part = rows * i + j + 1  # parts are 1-indexed in face_indices
            offset_per_part[part] = (u, v)

    U_norm = U.clone()
    V_norm = V.clone()

    # iterate over faces and offset the corresponding vertex u and v values
    for i in range(len(faces)):
        face_vert_idxs = faces[i]
        part = face_indices[i]
        offset_u, offset_v = offset_per_part[int(part.item())]
        
        for vert_idx in face_vert_idxs:   
            # vertices are reused, but we don't want to offset multiple times
            if vert_idx.item() not in already_offset:
                # offset u value
                U_norm[vert_idx] = U[vert_idx] / cols + offset_u
                # offset v value
                # this also flips each part locally, as each part is upside down
                V_norm[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
                # add vertex to our set tracking offsetted vertices
                already_offset.add(vert_idx.item())

    # invert V values
    V_norm = 1 - V_norm

    # create our verts_uv values
    return torch.cat([U_norm[None],V_norm[None]], dim=2) # (1, 7829, 2)


def create_renderer():
    # Initialize a camera.
    # World coordinates +Y up, +X left and +Z in.
    R, T = look_at_view_transform(2.7, 0, 0) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the person. 
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )


# Load SMPL and texture data
with open(verts_filename, 'rb') as f:
    data = pickle.load(f, encoding='latin1') 
    v_template = torch.Tensor(data['v_template']).to(device) # (6890, 3)

with Image.open(tex_filename) as image:
    np_image = np.asarray(image.convert("RGB")).astype(np.float32)
tex = torch.from_numpy(np_image / 255.)[None].to(device)

# plt.figure(figsize=(10,10))
# plt.imshow(tex.squeeze(0).cpu())
# plt.axis("off")
# plt.show()


# There are 6890 xyz vertex coordinates but 7829 vertex uv coordinates. 
# This is because the same vertex can be shared by multiple faces where each face may correspond to a different body part.  
# Therefore when initializing the Meshes class,
# we need to map each of the vertices referenced by the DensePose faces (in verts, which is the "All_vertices" field)
# to the correct xyz coordinate in the SMPL template mesh.
v_template_extended = v_template[verts-1][None] # (1, 7829, 3)

verts_uv = convert_denpose_verts_uv()
texture = TexturesUV(maps=tex, faces_uvs=faces[None], verts_uvs=verts_uv)
mesh = Meshes(v_template_extended, faces[None], texture)

renderer = create_renderer()
images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");
plt.show()

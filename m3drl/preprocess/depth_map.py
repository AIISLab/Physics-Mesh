# 128x128 grayscale depth map generated for OBJ model
import open3d as o3d
import numpy as np
import mesh_to_depth as m2d # TODO: fix version control issue
import matplotlib.pyplot as plt

mesh = o3d.io.read_triangle_mesh("model_normalized.obj")

# x, y, z = mesh.get_center()
camera_params = {
    "cam_pos": [1, 1, 1],
    "cam_lookat": [0, 0, 0],
    "cam_up": [0, 1, 0],
    "x_fov": 0.349,
    "near": 0.1,
    "far": 10,
    "height": 128,
    "width": 128,
    "is_depth": True
}

vertices = np.asarray(mesh.verticies, dtype=np.float32)
faces = np.asarray(mesh.triangles, dtype=np.float32)

depth_map = m2d.mesh_to_depth(vertices, faces, camera_params, empty_pixel_value=np.nan)

plt.imshow(depth_map[0], interpolation="none")
# plt.colorbar()
plt.show()
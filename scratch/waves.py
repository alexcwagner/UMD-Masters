import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# --- Geodesic mesh ---
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
verts0, faces = mesh.vertices, mesh.faces
N = len(verts0)

# --- Per-vertex area ---
areas = np.zeros(N)
for tri in faces:
    v0, v1, v2 = verts0[tri]
    A = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
    for i in tri:
        areas[i] += A / 3.0

# --- Cotangent Laplacian ---
L = np.zeros((N, N))


def angle(a, b):
    return np.arccos(
        np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
    )


for tri in faces:
    i, j, k = tri
    vi, vj, vk = verts0[i], verts0[j], verts0[k]
    ai = angle(vj - vi, vk - vi)
    aj = angle(vi - vj, vk - vj)
    ak = angle(vi - vk, vj - vk)
    cot_i, cot_j, cot_k = 1 / np.tan(ai), 1 / np.tan(aj), 1 / np.tan(ak)
    L[i, j] += -0.5 * cot_k
    L[j, i] += -0.5 * cot_k
    L[i, k] += -0.5 * cot_j
    L[k, i] += -0.5 * cot_j
    L[j, k] += -0.5 * cot_i
    L[k, j] += -0.5 * cot_i
for i in range(N):
    L[i, i] = -np.sum(L[i])

M_inv = 1.0 / areas

# --- Simulation params ---
c0 = 10.0
h = np.min(np.linalg.norm(verts0[faces[:, 0]] - verts0[faces[:, 1]], axis=1))
dt = 0.4 * h / c0
gamma = 0.02

# --- Initial displacement: off-center bump ---
center = np.array([0.5, 0.0, np.sqrt(1 - 0.5**2)])
dot = verts0 @ center
u = np.exp(-15 * (1 - dot) ** 2)
v = -2.0 * (L @ u) * M_inv

# --- Frame buffer ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
frames = []
num_frames = 400

# --- Simulation loop ---
for n in range(num_frames):
    print(
        n,
    )
    for _ in range(2):
        a = -(c0**2) * (L @ u) * M_inv - gamma * v
        v += dt * a
        u += dt * v

    # Physically deform vertices along normals
    verts = verts0 * (1.0 + 0.15 * u[:, None])  # 0.15 = exaggeration scale

    ax.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        cmap="coolwarm",
        linewidth=0.1,
        antialiased=False,
        shade=True,
        facecolors=plt.cm.coolwarm((u - u.min()) / (u.max() - u.min())),
    )
    ax.view_init(elev=20, azim=30)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(Image.fromarray(img))

plt.close(fig)

frames[0].save("wave.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
print("✅ Saved wave.gif with", len(frames), "frames")

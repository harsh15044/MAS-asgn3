import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2


#params
NAME = "HARSH"
IMAGE_FOLDER = "letters"
N = 20
P = 0.35
DT = 0.04
T_TRANSITION = 3.0
T_PAUSE = 1.0
K_CONSENSUS = 1.2
K_ANCHOR = 1.8

np.random.seed(42)


#Letter point extraction
def extract_points_from_image(char, folder=IMAGE_FOLDER, n_points=20, scale=6.0):
    """
    Loads <folder>/<char>.png, thresholds the letter strokes, and clusters the foreground pixels into exactly n_points.
    """
    path = os.path.join(folder, f"{char}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    #load and convert to grayscale
    img = Image.open(path).convert('L')
    img_arr = np.array(img, dtype=float) / 255.0
    h, w = img_arr.shape

    #assuming dark background(png image)
    mask = img_arr > 0.5

    rows, cols = np.where(mask)

    #converting pixel coords to Cartesian
    xs = cols.astype(float)
    ys = (h - 1 - rows).astype(float)
    pixel_coords = np.column_stack([xs, ys])

    #K-Means clustering to uniformly distribute n_points along the stroke
    centroids, _ = kmeans2(pixel_coords, k=n_points, minit='points', iter=25)

    #centering target points at (0, 0) and normalizing
    centered = centroids - np.mean(centroids, axis=0)
    max_extent = np.max(np.abs(centered))
    normalized = (centered / max_extent) * (scale / 2.0)

    return normalized

# Erdos-Renyi Graph generation helper
def gen_connected_er_graph(n, p):
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G

G = gen_connected_er_graph(N, P)
L = nx.laplacian_matrix(G).toarray()
edges = list(G.edges())

#Formation Control Simulation
pos = np.random.uniform(-6.0, 6.0, size=(N, 2))
frames = []
stage_labels = []

steps_transition = int(T_TRANSITION / DT)
steps_pause = int(T_PAUSE / DT)

for char in NAME:
    target_pts = extract_points_from_image(char, folder=IMAGE_FOLDER, n_points=N)

    
    cost_matrix = np.linalg.norm(pos[:, None, :] - target_pts[None, :, :], axis=2)
    _, col_ind = linear_sum_assignment(cost_matrix)
    R = target_pts[col_ind]

    #dynamics
    for _ in range(steps_transition):
        err = pos - R
        pos_dot = - K_CONSENSUS * (L @ err) - K_ANCHOR * err
        pos += pos_dot * DT
        frames.append(pos.copy())
        stage_labels.append(char)

    for _ in range(steps_pause):
        err = pos - R
        pos_dot = - K_CONSENSUS * (L @ err) - K_ANCHOR * err
        pos += pos_dot * DT
        frames.append(pos.copy())
        stage_labels.append(char)


#plotting and exporting
fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

edge_lines = [ax.plot([], [], color='#b0b0b0', lw=0.9, alpha=0.7)[0] for _ in edges]
scatter = ax.scatter([], [], s=55, color='#2b5c8f', edgecolors='#14324f', lw=0.8, zorder=3)
title = ax.set_title('', fontsize=12, pad=12)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.35)
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update(frame_idx):
    current_pos = frames[frame_idx]
    scatter.set_offsets(current_pos)

    for line, (u, v) in zip(edge_lines, edges):
        line.set_data([current_pos[u, 0], current_pos[v, 0]], 
                      [current_pos[u, 1], current_pos[v, 1]])

    title.set_text(f"Formation: Letter '{stage_labels[frame_idx]}'")
    return [scatter, title] + edge_lines

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=int(DT * 1000), blit=True)

output_base = f"{NAME.lower()}_formation_{N}"
try:
    writer = animation.FFMpegWriter(fps=int(1 / DT), bitrate=1500)
    ani.save(f"{output_base}.mp4", writer=writer)
    print(f"Saved: {output_base}.mp4")
except Exception:
    writer = animation.PillowWriter(fps=25)
    ani.save(f"{output_base}.gif", writer=writer)
    print(f"Saved: {output_base}.gif")

plt.close()
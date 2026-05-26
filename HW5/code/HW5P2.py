import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P2", exist_ok=True)

A = np.array([[1, 1], [0, 1]])
b = np.array([1, 0])
A_inv = np.array([[1, -1], [0, 1]])

X_verts = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
Y_verts = (A @ X_verts.T).T + b
Y_ordered = Y_verts[ConvexHull(Y_verts).vertices]

# Preimage of S = {y : 0<=y1<=2, -1<=y2<=1}
S_verts = np.array([[0,-1],[2,-1],[2,1],[0,1]])
pre_verts = (A_inv @ (S_verts - b).T).T
pre_ordered = pre_verts[ConvexHull(pre_verts).vertices]

def annotate(ax, verts, color, offset=(5, 5)):
    for v in verts:
        ax.plot(*v, 'o', color=color, ms=6, zorder=5)
        ax.annotate(f"({v[0]:.0f},{v[1]:.0f})", v, xytext=offset,
                    textcoords='offset points', fontsize=10, color=color)

# Plot (a): X and image Y
fig, ax = plt.subplots(figsize=(4.5, 5.0))
ax.add_patch(Polygon(X_verts, closed=True, alpha=0.3, facecolor='steelblue', edgecolor='steelblue', label='X'))
ax.add_patch(Polygon(Y_ordered, closed=True, alpha=0.4, facecolor='tomato', edgecolor='tomato', label='Y = AX+b'))
for vx, vy in zip(X_verts, Y_verts):
    ax.annotate("", xy=vy, xytext=vx, arrowprops=dict(arrowstyle="->", color='gray', lw=1, linestyle='dashed'))
annotate(ax, X_verts, 'steelblue', offset=(-18, -14))
annotate(ax, Y_verts, 'tomato', offset=(5, 5))
ax.set_xlim(-3, 5); ax.set_ylim(-3, 3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_aspect('equal')
ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P2/image_Y_of_X_under_affine_map.png", dpi=150)
plt.close()

# Plot (b): S and its preimage in x-space
fig, ax = plt.subplots(figsize=(4.5, 5.0))
ax.add_patch(Polygon(S_verts, closed=True, alpha=0.2, facecolor='gray', edgecolor='gray', linestyle='--', label='S (target in y-space)'))
ax.add_patch(Polygon(pre_ordered, closed=True, alpha=0.45, facecolor='mediumpurple', edgecolor='mediumpurple', label='Preimage of S'))
annotate(ax, S_verts, 'gray', offset=(5, -14))
annotate(ax, pre_verts, 'mediumpurple', offset=(5, 5))
for vs, vp in zip(S_verts, pre_verts):
    ax.annotate("", xy=vp, xytext=vs, arrowprops=dict(arrowstyle="->", color='gray', lw=1, linestyle='dashed'))
ax.set_xlim(-3, 4); ax.set_ylim(-2, 2)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_aspect('equal')
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P2/preimage_of_S_in_x_space.png", dpi=150)
plt.close()

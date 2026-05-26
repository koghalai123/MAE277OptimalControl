import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P1", exist_ok=True)

P_verts = np.array([[-1,-1],[2,-1],[2,1],[-1,1]])
Q_verts = np.array([[1,0],[0,1],[-1,0],[0,-1]])

# Minkowski pairwise vertex sums
mink_pts = np.array([p + q for p in P_verts for q in Q_verts])
hull_verts = mink_pts[ConvexHull(mink_pts).vertices]
hull_verts = hull_verts[np.argsort(np.arctan2(hull_verts[:,1] - hull_verts[:,1].mean(),
                                               hull_verts[:,0] - hull_verts[:,0].mean()))]

def make_ax():
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    ax.set_xlim(-2.5, 4); ax.set_ylim(-3, 3)
    ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    return fig, ax

# (a) P and Q
fig, ax = make_ax()
ax.add_patch(Polygon(P_verts, closed=True, facecolor='steelblue', edgecolor='steelblue', linewidth=2, alpha=0.4, label='$P$'))
ax.add_patch(Polygon(Q_verts, closed=True, facecolor='tomato',    edgecolor='tomato',    linewidth=2, alpha=0.4, label='$Q$'))
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/P_and_Q.png", dpi=150)
plt.close()

# (b) Intersection P ∩ Q = Q
fig, ax = make_ax()
ax.add_patch(Polygon(P_verts, closed=True, facecolor='steelblue', edgecolor='steelblue', linewidth=2, alpha=0.4, label='$P$'))
ax.add_patch(Polygon(Q_verts, closed=True, facecolor='purple',    edgecolor='purple',    linewidth=2, alpha=0.5, label='$P \\cap Q = Q$'))
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/intersection_P_cap_Q.png", dpi=150)
plt.close()

# (c) Minkowski sum P ⊕ Q
fig, ax = make_ax()
ax.add_patch(Polygon(hull_verts, closed=True, facecolor='seagreen', edgecolor='seagreen', linewidth=2, alpha=0.4, label='$P \\oplus Q$'))
ax.scatter(mink_pts[:, 0], mink_pts[:, 1], color='dimgray', s=30, zorder=5, label='$P+$Q vertices')
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/minkowski_sum_P_plus_Q.png", dpi=150)
plt.close()

print("Figures saved to HW5P1/")

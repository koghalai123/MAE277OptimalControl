import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P1", exist_ok=True)

P_verts = np.array([[-1,-1],[2,-1],[2,1],[-1,1]])
Q_verts = np.array([[1,0],[0,1],[-1,0],[0,-1]])

# P ∩ Q: Q is contained in P, so the intersection is just Q
inter_verts = Q_verts

# Minkowski sum P ⊕ Q = {p+q : p in P, q in Q}
mink_pts = np.array([p + q for p in P_verts for q in Q_verts])
hull = ConvexHull(mink_pts)
hull_verts = mink_pts[hull.vertices]
hull_verts = hull_verts[np.argsort(np.arctan2(
    hull_verts[:,1] - hull_verts[:,1].mean(),
    hull_verts[:,0] - hull_verts[:,0].mean()))]

def make_ax():
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    return fig, ax

def annotate_verts(ax, verts, color, offset=(5, 5)):
    for v in verts:
        ax.plot(*v, 'o', color=color, ms=5, zorder=5)
        ax.annotate(f"({v[0]},{v[1]})", v, xytext=offset,
                    textcoords='offset points', fontsize=9, color=color)

# (a) P and Q
fig, ax = make_ax()
ax.add_patch(Polygon(P_verts, closed=True, facecolor='steelblue', edgecolor='steelblue', lw=2, alpha=0.35, label='$P$'))
ax.add_patch(Polygon(Q_verts, closed=True, facecolor='tomato', edgecolor='tomato', lw=2, alpha=0.5, label='$Q$'))
annotate_verts(ax, P_verts, 'steelblue', offset=(5, 5))
annotate_verts(ax, Q_verts, 'tomato', offset=(5, -14))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/P_and_Q.png", dpi=150)
plt.close()

# (b) Intersection P ∩ Q
fig, ax = make_ax()
ax.add_patch(Polygon(P_verts, closed=True, facecolor='steelblue', edgecolor='steelblue', lw=2, alpha=0.2, label='$P$'))
ax.add_patch(Polygon(inter_verts, closed=True, facecolor='purple', edgecolor='purple', lw=2, alpha=0.6, label='$P \\cap Q$'))
annotate_verts(ax, inter_verts, 'purple')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/intersection_P_cap_Q.png", dpi=150)
plt.close()

# (c) Minkowski sum P ⊕ Q
fig, ax = make_ax()
ax.add_patch(Polygon(P_verts, closed=True, facecolor='steelblue', edgecolor='steelblue', lw=1, alpha=0.2, label='$P$'))
ax.add_patch(Polygon(Q_verts, closed=True, facecolor='tomato', edgecolor='tomato', lw=1, alpha=0.2, label='$Q$'))
ax.add_patch(Polygon(hull_verts, closed=True, facecolor='seagreen', edgecolor='seagreen', lw=2, alpha=0.45, label='$P \\oplus Q$'))
ax.scatter(mink_pts[:,0], mink_pts[:,1], color='dimgray', s=25, zorder=5, label='vertex sums')
annotate_verts(ax, hull_verts, 'seagreen', offset=(5, 5))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P1/minkowski_sum_P_plus_Q.png", dpi=150)
plt.close()

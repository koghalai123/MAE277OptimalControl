import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P2", exist_ok=True)

# y = Ax + b, X = unit square
A = np.array([[1, 1], [0, 1]])
b = np.array([1, 0])

X_verts = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
Y_verts = (A @ X_verts.T).T + b
Y_ordered = Y_verts[ConvexHull(Y_verts).vertices]

# --- Part (a): image Y ---
print("=" * 50)
print("Part (a): Image Y = {Ax + b : x in X}")
print("Vertices of Y (mapped from corners of X):")
for vx, vy in zip(X_verts, Y_verts):
    print(f"  x = {vx}  -->  y = {vy}")
print("""
Image as linear inequalities (derived from X = [-1,1]^2):
  y2 = x2                =>  -1 <= y2 <= 1
  y1 = x1 + x2 + 1,  x1 in [-1,1]  =>  y2 <= y1 <= y2 + 2
""")

# --- Part (b): preimage of S = {y : 0<=y1<=2, -1<=y2<=1} ---
print("\n" + "=" * 50)
print("Part (b): Preimage {x : Ax + b in S}")
print("""
S = {y : c_lo <= y <= c_hi},  c_lo = [0,-1],  c_hi = [2,1]

Preimage: c_lo <= Ax + b <= c_hi  =>  c_lo - b <= Ax <= c_hi - b
  A^{-1}(c_lo - b) <= x <= A^{-1}(c_hi - b)  (A invertible, A^{-1} = [[1,-1],[0,1]])

  => -1 <= x1 + x2 <= 1,   -1 <= x2 <= 1
""")

S_corners = np.array([[0,-1],[2,-1],[2,1],[0,1]])
A_inv = np.array([[1,-1],[0,1]])
pre_corners = (A_inv @ (S_corners - b).T).T
print("Vertices of S  -->  preimage vertex (A^{-1}(y - b)):")
for vs, vp in zip(S_corners, pre_corners):
    print(f"  y = {vs}  -->  x = {vp}")
print()

# --- Plot (a): X and its image Y ---
fig, ax = plt.subplots(figsize=(4.5, 5.0))
ax.add_patch(Polygon(X_verts, closed=True, alpha=0.3,
                     facecolor='steelblue', edgecolor='steelblue', label='X'))
ax.add_patch(Polygon(Y_ordered, closed=True, alpha=0.4,
                     facecolor='tomato', edgecolor='tomato', label='Y = AX+b'))
for vx, vy in zip(X_verts, Y_verts):
    ax.plot(*vx, 'o', color='steelblue', ms=6)
    ax.annotate(f"({vx[0]},{vx[1]})", vx, xytext=(-18,-14), textcoords='offset points', fontsize=10, color='steelblue')
    ax.plot(*vy, 's', color='tomato', ms=6)
    ax.annotate(f"({vy[0]:.0f},{vy[1]:.0f})", vy, xytext=(5,5), textcoords='offset points', fontsize=10, color='tomato')
    ax.annotate("", xy=vy, xytext=vx, arrowprops=dict(arrowstyle="->", color='gray', lw=1, linestyle='dashed'))
ax.set_xlim(-3, 5); ax.set_ylim(-3, 3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_aspect('equal')
ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P2/image_Y_of_X_under_affine_map.png", dpi=150)
plt.close()

# --- Plot (b): preimage of S in x-space ---
pre_verts = np.array([[0,-1],[2,-1],[0,1],[-2,1]])
pre_ordered = pre_verts[ConvexHull(pre_verts).vertices]

fig, ax = plt.subplots(figsize=(4.5, 5.0))
ax.add_patch(Polygon(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]), closed=True,
                     alpha=0.15, facecolor='gray', edgecolor='gray',
                     linestyle='--', label='X (ref)'))
ax.add_patch(Polygon(pre_ordered, closed=True, alpha=0.4,
                     facecolor='mediumpurple', edgecolor='mediumpurple', label='Preimage of S'))
for v in np.array([[-1,-1],[1,-1],[1,1],[-1,1]]):
    ax.plot(*v, 'o', color='gray', ms=6, zorder=5)
    ax.annotate(f"({v[0]},{v[1]})", v, xytext=(5,-14), textcoords='offset points', fontsize=10, color='gray')
for v in pre_verts:
    ax.plot(*v, 's', color='mediumpurple', ms=6, zorder=5)
    ax.annotate(f"({v[0]},{v[1]})", v, xytext=(5,5), textcoords='offset points', fontsize=10, color='mediumpurple')
ax.set_xlim(-3, 4); ax.set_ylim(-2, 2)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_aspect('equal')
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.grid(True, alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P2/preimage_of_S_in_x_space.png", dpi=150)
plt.close()

print("Figures saved to HW5P2/")

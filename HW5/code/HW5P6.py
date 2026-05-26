import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polytope as pc
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P6", exist_ok=True)

A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])

X  = pc.box2poly([[-5, 5], [-2, 2]])
U  = pc.box2poly([[-1, 1]])
Xf = pc.box2poly([[-0.5, 0.5], [-0.5, 0.5]])

def pre(T):
    n, m = A.shape[0], B.shape[1]
    HAB = np.hstack([T.A @ A, T.A @ B])
    HX  = np.hstack([X.A, np.zeros((X.A.shape[0], m))])
    HU  = np.hstack([np.zeros((U.A.shape[0], n)), U.A])
    H   = np.vstack([HAB, HX, HU])
    h   = np.concatenate([T.b, X.b, U.b])
    return pc.projection(pc.Polytope(H, h), list(range(1, n + 1)))

K1 = pre(Xf)
K2 = pre(K1)
K3 = pre(K2)

def poly_area(p):
    v = pc.extreme(p)
    order = np.argsort(np.arctan2(v[:,1] - v[:,1].mean(), v[:,0] - v[:,0].mean()))
    v = v[order]
    # shoelace
    x, y = v[:,0], v[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

areas = {name: poly_area(p) for name, p in [('$X_f$', Xf), ('$K_1$', K1), ('$K_2$', K2), ('$K_3$', K3)]}

def plot_poly(ax, poly, color, alpha):
    verts = pc.extreme(poly)
    order = np.argsort(np.arctan2(verts[:,1] - verts[:,1].mean(),
                                   verts[:,0] - verts[:,0].mean()))
    verts = verts[order]
    ax.fill(verts[:,0], verts[:,1], color=color, alpha=alpha)
    ax.plot(np.append(verts[:,0], verts[0,0]),
            np.append(verts[:,1], verts[0,1]), color=color, lw=1)

patches = [mpatches.Patch(color='limegreen', label='$X_f$'),
           mpatches.Patch(color='steelblue', label='$K_1$'),
           mpatches.Patch(color='royalblue', label='$K_2$'),
           mpatches.Patch(color='navy',      label='$K_3$')]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))

# Left: nested controllable sets
ax = axes[0]
for poly, color, alpha in [(K3,'navy',0.45),(K2,'royalblue',0.45),(K1,'steelblue',0.55),(Xf,'limegreen',0.9)]:
    plot_poly(ax, poly, color, alpha)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_xlim(-5, 5); ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)
ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)

# Right: exact area growth
ax2 = axes[1]
labels, vals = list(areas.keys()), list(areas.values())
bar_colors = ['limegreen', 'steelblue', 'royalblue', 'navy']
ax2.bar(labels, vals, color=bar_colors, alpha=0.8, edgecolor='k', lw=0.8)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=10)
ax2.set_ylabel("Area (units²)")
ax2.set_xlabel("Set")
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)

plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P6/controllable_sets_K1_K2_K3.png", dpi=150)
plt.close()

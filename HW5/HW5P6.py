import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polytope as pc
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P6", exist_ok=True)

# x+ = A*x + B*u
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])

# Constraints as polytopes (H-rep: {x : Hx <= h})
X  = pc.box2poly([[-5, 5], [-2, 2]])
U  = pc.box2poly([[-1, 1]])
Xf = pc.box2poly([[-0.5, 0.5], [-0.5, 0.5]])

def pre(T):
    """Pre(T) = {x in X : exists u in U s.t. Ax+Bu in T}
    Lift to (x,u)-space, intersect with X x U, then project onto x."""
    n, m = A.shape[0], B.shape[1]
    # Dynamics constraint: H_T*(Ax+Bu) <= h_T  =>  [H_T A | H_T B][x;u] <= h_T
    HAB = np.hstack([T.A @ A, T.A @ B])
    # State and input constraints padded to (x,u)-space
    HX  = np.hstack([X.A, np.zeros((X.A.shape[0], m))])
    HU  = np.hstack([np.zeros((U.A.shape[0], n)), U.A])
    H   = np.vstack([HAB, HX, HU])
    h   = np.concatenate([T.b, X.b, U.b])
    xu  = pc.Polytope(H, h)
    return pc.projection(xu, list(range(1, n + 1)))   # 1-indexed: keep x1..xn, eliminate u

print("Computing K1 = Pre(Xf)...")
K1 = pre(Xf)
print("Computing K2 = Pre(K1)...")
K2 = pre(K1)
print("Computing K3 = Pre(K2)...")
K3 = pre(K2)

print("\n(c) K_N grows with N because each predecessor step adds states that can")
print("    reach Xf in one more step. As N increases, we include states that")
print("    need more time steps to be steered into Xf.")
print("\n(d) If K_{N+1} = K_N, the set has converged to the maximal control")
print("    invariant set C_inf. Every state outside it cannot be driven to Xf")
print("    regardless of how many steps are allowed.")

# Plot all sets on one figure
fig, ax = plt.subplots(figsize=(4.5, 5.0))

for poly, color, alpha in [(K3, 'navy', 0.45), (K2, 'royalblue', 0.45),
                            (K1, 'steelblue', 0.55), (Xf, 'limegreen', 0.9)]:
    verts = pc.extreme(poly)
    order = np.argsort(np.arctan2(verts[:,1] - verts[:,1].mean(),
                                   verts[:,0] - verts[:,0].mean()))
    verts = verts[order]
    ax.fill(verts[:,0], verts[:,1], color=color, alpha=alpha)
    ax.plot(np.append(verts[:,0], verts[0,0]),
            np.append(verts[:,1], verts[0,1]), color=color, lw=1)

patches = [mpatches.Patch(color='limegreen', label='$X_f$ (target)'),
           mpatches.Patch(color='steelblue', label='$K_1 = \\mathrm{Pre}(X_f)$'),
           mpatches.Patch(color='royalblue', label='$K_2 = \\mathrm{Pre}^2(X_f)$'),
           mpatches.Patch(color='navy',      label='$K_3 = \\mathrm{Pre}^3(X_f)$')]
ax.legend(handles=patches, fontsize=12, loc='lower center',
          bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_xlim(-5, 5); ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)

plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P6/controllable_sets_K1_K2_K3.png", dpi=150)
plt.close()
print("Figure saved to HW5P6/")

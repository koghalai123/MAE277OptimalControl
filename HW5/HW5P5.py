import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P5", exist_ok=True)

# ================================================================
# Problem 5: Control Invariant Sets
# x_{k+1} = 1.2*x + u,  X=[-2,2], U=[-0.3,0.3]
# ================================================================
print("=" * 60)
print("Problem 5: Control Invariant Sets")

# (a) Show X is not control invariant
print("\n(a) Is X = [-2, 2] control invariant?")
print("    Try x = 2: need 1.2(2) + u = 2.4 + u in [-2,2]")
print("    => u in [-4.4, -0.4], but U = [-0.3, 0.3] => empty intersection.")
print("    => X is NOT control invariant.")

# (b) Condition on alpha
print("\n(b) S = [-alpha, alpha] invariant iff for all x in S, exists u in U: 1.2x+u in S")
print("    Worst case x = alpha: need 1.2*alpha + u <= alpha for some u >= -0.3")
print("    => u <= -0.2*alpha  =>  -0.3 <= -0.2*alpha  =>  alpha <= 1.5")
print("    (Symmetric for x = -alpha.)  Condition: alpha <= 1.5")

# (c) Largest alpha
alpha_max = 0.3 / 0.2
print(f"\n(c) Largest invariant set: alpha_max = 0.3/0.2 = {alpha_max}")
print(f"    S* = [-{alpha_max}, {alpha_max}]")

# (d) Check x = 1.5
x = 1.5
xn_base = 1.2 * x          # = 1.8
u_star  = -0.3
xn = xn_base + u_star
print(f"\n(d) x = {x}: x_{{k+1}} = {xn_base} + u,  need x_{{k+1}} in [-1.5, 1.5]")
print(f"    u must be in [{-alpha_max - xn_base:.2f}, {alpha_max - xn_base:.2f}]")
print(f"    With u = {u_star}: x_{{k+1}} = {xn} in [-1.5, 1.5]? {abs(xn) <= alpha_max}")

# Plot
fig, ax = plt.subplots(figsize=(4.5, 5.0))
alphas = np.linspace(0, 2.1, 400)
ax.fill_between(alphas, -alphas, alphas, where=(alphas <= alpha_max),
                alpha=0.35, color='green', label='Control invariant S=[-a,a]')
ax.fill_between(alphas, -alphas, alphas, where=(alphas > alpha_max),
                alpha=0.25, color='red', label='Not invariant')
ax.axvline(alpha_max, color='k', ls='--', lw=1.5, label=f'alpha_max={alpha_max}')
ax.set_xlabel("alpha"); ax.set_ylabel("x")
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=1); ax.grid(True, alpha=0.3)

plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P5/control_invariant_set_alpha.png", dpi=150)
plt.close()
print("Figure saved to HW5P5/")

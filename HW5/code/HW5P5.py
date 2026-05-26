import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P5", exist_ok=True)

a = 1.2
X_lim = 2.0
u_max = 0.3
alpha_max = u_max / (a - 1)   # = 0.3/0.2 = 1.5

alphas = np.linspace(0, X_lim, 500)

# For each alpha, find the tightest u needed at x=alpha: u* = -(a-1)*alpha
u_needed = -(a - 1) * alphas   # negative: must push back

# Simulate a few trajectories starting just inside and just outside alpha_max
def simulate(x0, u_policy, N=25):
    xs = [x0]
    for _ in range(N - 1):
        u = np.clip(u_policy(xs[-1]), -u_max, u_max)
        xs.append(np.clip(a * xs[-1] + u, -X_lim, X_lim))
    return xs

# Optimal policy: always apply u = -(a-1)*x (keeps x on boundary)
u_opt = lambda x: -(a - 1) * x

fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))

# Left: feasibility diagram — alpha vs u*
ax = axes[0]
ax.plot(alphas, u_needed, color='steelblue', lw=2, label='$u^* = -(a-1)\\alpha$')
ax.axhline(-u_max, color='tomato', ls='--', lw=1.5, label=f'$-u_{{max}} = -{u_max}$')
ax.axvline(alpha_max, color='k', ls=':', lw=1.5, label=f'$\\alpha_{{max}} = {alpha_max}$')
ax.fill_betweenx([-u_max, 0], 0, alpha_max, alpha=0.15, color='green', label='Feasible')
ax.fill_betweenx([-u_max * 2.5, -u_max], 0, X_lim, alpha=0.12, color='red', label='Infeasible')
ax.set_xlabel("$\\alpha$"); ax.set_ylabel("Required $u^*$")
ax.set_xlim(0, X_lim); ax.set_ylim(-u_max * 2.5, 0.1)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)

# Right: trajectories for x0 inside, at, and outside alpha_max
ax2 = axes[1]
for x0, lbl, c in [(1.2, '$x_0=1.2$ (inside)', 'steelblue'),
                    (alpha_max, f'$x_0=\\alpha_{{max}}$', 'green'),
                    (1.7, '$x_0=1.7$ (outside)', 'tomato')]:
    traj = simulate(x0, u_opt)
    ax2.plot(traj, marker='o', ms=3, color=c, label=lbl)
ax2.axhline( alpha_max, color='k', ls='--', lw=1)
ax2.axhline(-alpha_max, color='k', ls='--', lw=1)
ax2.axhline( X_lim, color='gray', ls=':', lw=1)
ax2.axhline(-X_lim, color='gray', ls=':', lw=1)
ax2.set_xlabel("$k$"); ax2.set_ylabel("$x_k$")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)

plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P5/control_invariant_set_alpha.png", dpi=150)
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P4", exist_ok=True)

cases = [("a", 0.5), ("b", 1.0), ("c", 1.2)]
N = 20
x0_vals = [1.0, 0.5, -0.8, -1.0]
colors = ['steelblue', 'tomato', 'seagreen', 'purple']

for _, a in cases:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))

    # Left: trajectories
    ax = axes[0]
    for x0, c in zip(x0_vals, colors):
        traj = [x0 * a**k for k in range(N)]
        ax.plot(traj, marker='o', ms=3, color=c, label=f"$x_0={x0}$")
    ax.axhline( 1, color='k', ls='--', lw=1)
    ax.axhline(-1, color='k', ls='--', lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("$k$"); ax.set_ylabel("$x_k$")
    ax.set_title(f"$a = {a}$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)

    # Right: phase plot x_{k+1} vs x_k
    ax2 = axes[1]
    xvals = np.linspace(-1.2, 1.2, 200)
    ax2.plot(xvals, a * xvals, color='steelblue', lw=2, label=f"$x_{{k+1}} = {a}x_k$")
    ax2.plot(xvals, xvals, 'k--', lw=1, label="$x_{k+1} = x_k$")
    ax2.axhline( 1, color='gray', ls=':', lw=1)
    ax2.axhline(-1, color='gray', ls=':', lw=1)
    ax2.axvline( 1, color='gray', ls=':', lw=1)
    ax2.axvline(-1, color='gray', ls=':', lw=1)
    ax2.fill_between([-1, 1], [-1, -1], [1, 1], color='green', alpha=0.1, label='$X$')
    ax2.set_xlabel("$x_k$"); ax2.set_ylabel("$x_{k+1}$")
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=3)

    plt.subplots_adjust(bottom=0.35)
    plt.savefig(f"HW5P4/trajectory_a_{str(a).replace('.','p')}.png", dpi=150)
    plt.close()

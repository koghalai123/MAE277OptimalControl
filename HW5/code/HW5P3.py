import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P3", exist_ok=True)

X  = [-2.0,  2.0]
U  = [-0.5,  0.5]
Xf = [-0.25, 0.25]
X0 = [-0.25, 0.25]

def pre(T):
    return [max(T[0] - U[1], X[0]), min(T[1] - U[0], X[1])]

def reach(S):
    return [max(S[0] + U[0], X[0]), min(S[1] + U[1], X[1])]

K1 = pre(Xf);  K2 = pre(K1);  K3 = pre(K2)
R1 = reach(X0); R2 = reach(R1)

def plot_intervals(ax, intervals, labels, colors):
    for i, (iv, lbl, c) in enumerate(zip(intervals, labels, colors)):
        y = len(intervals) - 1 - i
        ax.barh(y, iv[1] - iv[0], left=iv[0], height=0.6,
                color=c, alpha=0.75, label=lbl)
        ax.text(iv[0], y, f" {iv[0]:.2f}", va='center', ha='right', fontsize=9, color=c)
        ax.text(iv[1], y, f" {iv[1]:.2f}", va='center', ha='left',  fontsize=9, color=c)
    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(-2.6, 2.6)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlabel("x")

fig, ax = plt.subplots(figsize=(4.5, 5.0))
plot_intervals(ax,
    [Xf, K1, K2, K3],
    ['$X_f$', '$K_1$', '$K_2$', '$K_3$'],
    ['green', 'steelblue', 'royalblue', 'navy'])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P3/controllable_sets_K1_K2_K3.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(4.5, 5.0))
plot_intervals(ax,
    [X0, R1, R2],
    ['$X_0$', '$R_1$', '$R_2$'],
    ['green', 'tomato', 'darkred'])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P3/reachable_sets_R1_R2_from_X0.png", dpi=150)
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P3", exist_ok=True)

# x_{k+1} = x_k + u_k,  X=[-2,2], U=[-0.5,0.5], Xf=[-0.25,0.25]
X  = [-2.0,  2.0]
U  = [-0.5,  0.5]
Xf = [-0.25, 0.25]

def pre(T):
    # Pre(T) = {x in X : x+u in T for some u in U} = [T[0]-U[1], T[1]-U[0]] ∩ X
    return [max(T[0] - U[1], X[0]), min(T[1] - U[0], X[1])]

def reach(S):
    # Reach(S) = {x+u : x in S, u in U} ∩ X
    return [max(S[0] + U[0], X[0]), min(S[1] + U[1], X[1])]

# (a)-(c) Controllable sets
# Pre(T) = {x in X : x+u in T for some u in U}
#         = {x : T[0]-U[1] <= x <= T[1]-U[0]} ∩ X
K1 = pre(Xf)
K2 = pre(K1)
K3 = pre(K2)

print("Controllable sets:  Pre(T) = [T_lo - U_hi, T_hi + |U_lo|] ∩ X\n")
for name, T, K in [("(a) K1 = Pre(Xf)", Xf, K1),
                   ("(b) K2 = Pre(K1)", K1, K2),
                   ("(c) K3 = Pre(K2)", K2, K3)]:
    lo = round(T[0] - U[1], 6); hi = round(T[1] - U[0], 6)
    print(f"{name}")
    print(f"    [{T[0]:.3f} - {U[1]:.1f},  {T[1]:.3f} + {abs(U[0]):.1f}] = [{lo:.3f}, {hi:.3f}]  =>  ∩ X = {K}")

print("\n    Pattern: each Pre expands the interval by 0.5 on each side")
print("    until it saturates at X = [-2, 2].")

# (d) Reachable sets from X0
# Reach(S) = {x+u : x in S, u in U} ∩ X = [S_lo+U_lo, S_hi+U_hi] ∩ X
X0 = [-0.25, 0.25]
R1 = reach(X0)
R2 = reach(R1)

print("\nReachable sets:  Reach(S) = [S_lo + U_lo, S_hi + U_hi] ∩ X\n")
for name, S, R in [("(d) R1 = Reach(X0)", X0, R1),
                   ("    R2 = Reach(R1)", R1, R2)]:
    lo = round(S[0] + U[0], 6); hi = round(S[1] + U[1], 6)
    print(f"{name}")
    print(f"    [{S[0]:.3f} - {abs(U[0]):.1f},  {S[1]:.3f} + {U[1]:.1f}] = [{lo:.3f}, {hi:.3f}]  =>  ∩ X = {R}")

# (e) printed explanation
print("""
(e) The controllable set Kk is the set of states from which there EXISTS
    a control sequence that steers the system INTO Xf within k steps --
    it is computed backward from the target.

    The reachable set Rk is the set of states the system CAN REACH at
    step k from X0 under any admissible input -- computed forward in time.

    Both use the same dynamics, but they answer opposite questions:
    controllability asks "which initial conditions allow us to hit the target?",
    while reachability asks "where can the state end up?".
""")

# Plot
def plot_intervals(ax, intervals, labels, colors):
    for i, (iv, lbl, c) in enumerate(zip(intervals, labels, colors)):
        y = len(intervals) - 1 - i
        ax.plot(iv, [y, y], color=c, lw=10, alpha=0.75,
                solid_capstyle='butt', label=lbl)
    ax.set_yticks([])
    ax.set_xlim(-2.6, 2.6)
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")

fig, ax = plt.subplots(figsize=(4.5, 5.0))
plot_intervals(ax,
    [Xf, K1, K2, K3],
    ['Xf', 'K1 = Pre(Xf)', 'K2 = Pre^2(Xf)', 'K3 = Pre^3(Xf)'],
    ['green', 'steelblue', 'royalblue', 'navy'])
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P3/controllable_sets_K1_K2_K3.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(4.5, 5.0))
plot_intervals(ax,
    [X0, R1, R2],
    ['X0', 'R1 (1-step)', 'R2 (2-step)'],
    ['green', 'tomato', 'darkred'])
ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.35)
plt.savefig("HW5P3/reachable_sets_R1_R2_from_X0.png", dpi=150)
plt.close()

print("Figures saved to HW5P3/")

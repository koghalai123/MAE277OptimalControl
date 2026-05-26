import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
os.makedirs("HW5P4", exist_ok=True)

# x_{k+1} = a*x_k,  X = [-1, 1]
# [-alpha, alpha] is positively invariant iff |a|*alpha <= alpha
# => |a| <= 1 (any alpha works), or alpha = 0 if |a| > 1

print("Positive invariance of [-alpha, alpha] under x_{k+1} = a*x_k:")
print("Condition: max|a*x| for x in [-a,a] = |a|*alpha <= alpha => |a| <= 1\n")

cases = [("a", 0.5), ("b", 1.0), ("c", 1.2)]

for part, a in cases:
    print(f"({part}) a = {a}")
    print(f"    x_{{k+1}} = {a} * x_k")
    print(f"    For x in [-1,1]: |x_{{k+1}}| = |{a}| * |x_k| <= {a}")
    if abs(a) <= 1:
        print(f"    |{a}| <= 1  =>  X = [-1, 1] IS positively invariant.\n")
    else:
        print(f"    |{a}| > 1  =>  X = [-1, 1] is NOT positively invariant.")
        print(f"    (e.g. x_k = 1 => x_{{k+1}} = {a} > 1, leaving X)")
        print(f"    Largest invariant [-alpha, alpha]: need {a}*alpha <= alpha")
        print(f"    => ({a} - 1)*alpha <= 0 => alpha = 0")
        print(f"    Largest positively invariant set: {{0}}\n")

print("(d) Relationship between stability and invariance:")
print("""
    For x_{k+1} = a*x_k:
    - |a| < 1 (stable):   system contracts, so X stays within itself -- invariant.
    - |a| = 1 (marginal): system neither grows nor shrinks -- X is invariant.
    - |a| > 1 (unstable): system expands, so any bounded set around the origin
      eventually gets violated. Only the equilibrium {0} is invariant.

    In general, a constraint set X containing the origin is positively invariant
    for a stable (or marginally stable) system, but NOT for an unstable one.
    Invariance of a compact set essentially requires the dynamics to not "escape"
    it, which is precisely what stability (contraction) guarantees.
""")

# Plot: one figure per case
N = 15
x0_vals = [1.0, 0.5, -0.8]

for part, a in cases:
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    for x0 in x0_vals:
        traj = [x0 * a**k for k in range(N)]
        ax.plot(traj, marker='o', markersize=3, label=f"x0={x0}")
    ax.axhline( 1, color='k', ls='--', lw=1, label='X boundary')
    ax.axhline(-1, color='k', ls='--', lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("$x_k$")
    ax.legend(fontsize=12, loc='lower center', bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(f"HW5P4/trajectory_a_{str(a).replace('.','p')}.png", dpi=150)
    plt.close()

print("Figures saved to HW5P4/")

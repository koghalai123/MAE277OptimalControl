import numpy as np
import matplotlib.pyplot as plt
import csv, os

Q = np.array([[10, 0], [0, 1]], dtype=float)
b = np.array([1, 2], dtype=float)
x0 = np.array([2, 2], dtype=float)
x_star = np.linalg.solve(Q, b)

def f(x):
    return 0.5 * x @ Q @ x - b @ x

def grad_f(x):
    return Q @ x - b


# --- Part 4: backtracking line search ---

def backtracking_gd(x0, n_iter=30, alpha0=1.0, rho=0.5, c=1e-4):
    x, its, alphas = x0.copy(), [x0.copy()], []
    for _ in range(n_iter):
        g = grad_f(x)
        a = alpha0
        while f(x - a*g) > f(x) - c*a*(g @ g):
            a *= rho
        alphas.append(a)
        x = x - a*g
        its.append(x.copy())
    return np.array(its), alphas

iterates_bt, alphas_bt = backtracking_gd(x0)
iterates_bt30, alphas_bt30 = iterates_bt, alphas_bt
print(f"Backtracking step sizes: iter1={alphas_bt[0]:.4f}, iter2={alphas_bt[1]:.4f}")


# --- Part 5: Newton's method ---

def newton(x0, n_iter=5):
    x, its = x0.copy(), [x0.copy()]
    for _ in range(n_iter):
        x = x - np.linalg.solve(Q, grad_f(x))
        its.append(x.copy())
    return np.array(its)

iterates_newton = newton(x0)
print(f"Newton x after 1 iteration: {iterates_newton[1]}")


# --- Part 6: fixed step and exact line search ---

def fixed_step_gd(x0, n_iter=3):
    x, its = x0.copy(), [x0.copy()]
    for _ in range(n_iter):
        x = x - grad_f(x)
        its.append(x.copy())
    return np.array(its)

def exact_ls_gd(x0, n_iter=30):
    x, its = x0.copy(), [x0.copy()]
    for _ in range(n_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < 1e-14:
            break
        a = (g @ g) / (g @ Q @ g)
        x = x - a*g
        its.append(x.copy())
    return np.array(its)

iterates_fixed = fixed_step_gd(x0)
iterates_exact = exact_ls_gd(x0)


# --- Save first 3 iterations to CSV ---

os.makedirs('HW3', exist_ok=True)

def save_csv(fname, iterates, extra=None):
    rows = []
    for k in range(min(3, len(iterates))):
        x, g = iterates[k], grad_f(iterates[k])
        row = {'k': k, 'x1': round(x[0],8), 'x2': round(x[1],8),
               'f(x)': round(f(x),10), 'grad1': round(g[0],8),
               'grad2': round(g[1],8), '||grad||': round(np.linalg.norm(g),8)}
        if extra:
            for col, vals in extra.items():
                row[col] = vals[k] if k < len(vals) else ''
        rows.append(row)
    with open(f'HW3/{fname}', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

save_csv('gd_fixed.csv',        iterates_fixed,  extra={'alpha': [1.0]*3})
save_csv('gd_exact_ls.csv',     iterates_exact,  extra={'alpha_k': [
    round((grad_f(iterates_exact[k])@grad_f(iterates_exact[k]))/(grad_f(iterates_exact[k])@Q@grad_f(iterates_exact[k])),6)
    if np.linalg.norm(grad_f(iterates_exact[k]))>1e-14 else 0.0 for k in range(3)]})
save_csv('gd_backtracking.csv', iterates_bt30,   extra={'alpha_k': [round(a,6) for a in alphas_bt30[:3]]})
save_csv('newton.csv',          iterates_newton, extra={'newton_dec': [
    round(float(np.sqrt(grad_f(iterates_newton[k])@np.linalg.solve(Q,grad_f(iterates_newton[k])))),8)
    for k in range(3)]})
print("CSVs saved.")


# --- Contour plot helper ---

def contour_plot(its, title, marker, fname):
    x1r = np.linspace(its[:,0].min()-1.5, its[:,0].max()+1.5, 400)
    x2r = np.linspace(its[:,1].min()-1.5, its[:,1].max()+1.5, 400)
    X1, X2 = np.meshgrid(x1r, x2r)
    Z = 0.5*(10*X1**2 + X2**2) - X1 - 2*X2

    fig, ax = plt.subplots(figsize=(7,5))
    cs = ax.contourf(X1, X2, Z, levels=40, cmap='Blues', alpha=0.6)
    ax.contour(X1, X2, Z, levels=40, colors='gray', linewidths=0.5, alpha=0.6)
    plt.colorbar(cs, ax=ax, label='f(x)')

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(its)))
    ax.plot(its[:,0], its[:,1], color='gray', linewidth=1.2, zorder=1)
    for k, pt in enumerate(its):
        ax.scatter(*pt, color=colors[k], marker=marker, s=60, zorder=3)
        ax.annotate(f'k={k}', pt, xytext=(8,6), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=colors[k],
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    ax.plot(*x_star, 'k*', markersize=12, label='x*')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_title(f'Contour – {title}'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'HW3/{fname}', dpi=150); plt.show()

contour_plot(iterates_fixed,   'GD Fixed Step (α=1)',        'x', 'contour_fixed.png')
contour_plot(iterates_exact,   'GD Exact Line Search',        'o', 'contour_exact.png')
contour_plot(iterates_bt30,    'GD Backtracking Line Search', 's', 'contour_backtrack.png')
contour_plot(iterates_newton,  "Newton's Method",             '^', 'contour_newton.png')


# --- Error and gradient norm plots ---

f_star = f(x_star)
all_methods = [
    (iterates_fixed,   'GD fixed (α=1)'),
    (iterates_exact,   'GD exact LS'),
    (iterates_bt30,    'GD backtracking'),
    (iterates_newton,  "Newton's method"),
]

fig, ax = plt.subplots(figsize=(7,4))
for its, label in all_methods:
    ax.semilogy([max(abs(f(x)-f_star), 1e-16) for x in its], label=label, linewidth=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('|f(xᵏ) - f(x*)|')
ax.set_title('Error vs iteration'); ax.legend()
plt.tight_layout(); plt.savefig('HW3/error.png', dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(7,4))
for its, label in all_methods:
    ax.semilogy([np.linalg.norm(grad_f(x)) for x in its], label=label, linewidth=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('||∇f(xᵏ)||')
ax.set_title('Gradient norm vs iteration'); ax.legend()
plt.tight_layout(); plt.savefig('HW3/grad_norm.png', dpi=150); plt.show()

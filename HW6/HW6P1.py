"""HW6 P1: MPC tracking control for a mass-spring system."""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

HERE = Path(__file__).parent
p = yaml.safe_load((HERE.parent / "plot_params.yaml").read_text())
plt.rcParams.update({"font.size": p["font"]["size"]})

# ── (a) System matrices ───────────────────────────────────────────────────────
m, k, Ts = 1.0, 1.0, 0.1

Ac = np.array([[0, 1], [-k/m, 0]])
Bc = np.array([[0], [1/m]])
Ad, Bd = cont2discrete((Ac, Bc, np.eye(2), np.zeros((2,1))), Ts, method='zoh')[:2]

print(f"Ac =\n{Ac}\nBc =\n{Bc}")
print(f"\nAd =\n{Ad}\nBd =\n{Bd}")

# ── (b) Augmented state  z = [Δx1, Δx2, e, u, x1] ───────────────────────────
# Δx propagates via Ad/Bd; e_{k+1} = Δx1_{k+1} + e_k (slow reference);
# u integrator: u_{k+1} = u_k + Δu_k; position: x1_{k+1} = x1_k + Δx1_{k+1}
A = np.array([
    [Ad[0,0], Ad[0,1], 0, 0, 0],
    [Ad[1,0], Ad[1,1], 0, 0, 0],
    [1,       0,       1, 0, 0],
    [0,       0,       0, 1, 0],
    [1,       0,       0, 0, 1],
])
B = np.array([[Bd[0,0]], [Bd[1,0]], [0], [1], [0]])

print(f"\nA_aug =\n{A}\nB_aug =\n{B}")

# ── (c)/(d) QP matrix builder ─────────────────────────────────────────────────
def form_qp(A, B, Q, R, P, xlim, ulim, N):
    """Return H, L, G, W, T, IMPC for the MPC QP.

    min  1/2 U'HU + (L z0)'U   s.t. G U <= W + T z0
    Δu0 = IMPC @ U
    """
    n, m_ = A.shape[0], B.shape[1]

    # Prediction: Z = Psi z0 + Gamma U
    Psi   = np.zeros((n*N, n))
    Gamma = np.zeros((n*N, m_*N))
    Apow  = A.copy()
    for i in range(N):
        Psi[i*n:(i+1)*n] = Apow
        Apow = A @ Apow
    for i in range(N):
        for j in range(i+1):
            Gamma[i*n:(i+1)*n, j*m_:(j+1)*m_] = np.linalg.matrix_power(A, i-j) @ B

    # Cost
    Qbar = linalg.block_diag(*([Q]*(N-1) + [P]))
    Rbar = np.kron(np.eye(N), R)
    H = Gamma.T @ Qbar @ Gamma + Rbar
    H = (H + H.T) / 2
    L = Gamma.T @ Qbar @ Psi

    # Constraints: state bounds and Δu bounds
    xmax = np.tile(xlim["max"], N)
    xmin = np.tile(xlim["min"], N)
    umax = np.tile([ulim["max"]], N)
    umin = np.tile([ulim["min"]], N)
    Iu   = np.eye(m_ * N)
    Zu   = np.zeros((m_*N, n))

    G = np.vstack([ Gamma, -Gamma,  Iu, -Iu])
    W = np.concatenate([xmax, -xmin, umax, -umin])
    T = np.vstack([-Psi,   Psi,   Zu,  Zu])

    IMPC = np.zeros((m_, m_*N))
    IMPC[0, 0] = 1.0
    return H, L, G, W, T, IMPC


def msd(_, x, u, mp, kp):
    """Mass-spring ODE."""
    return [x[1], -(kp/mp)*x[0] + (1/mp)*u]


def run_mpc(mp=1.0, kp=1.0, u_lim=0.2):
    """Closed-loop MPC simulation. Returns t, x1, u, r arrays."""
    N     = 15
    Q_mpc = np.diag([0., 0., 1., 0., 0.])
    R_mpc = np.array([[1.]])
    Pdxe  = solve_discrete_are(A[:3,:3], B[:3], np.diag([0.,0.,1.]), R_mpc)
    P_mpc = linalg.block_diag(Pdxe, np.zeros((2,2)))

    lN   = 1e6
    xlim = {"max": np.array([lN, lN, lN,  u_lim,  0.2]),
            "min": np.array([-lN,-lN,-lN, -u_lim, -0.2])}
    ulim = {"max": lN, "min": -lN}   # Δu unconstrained

    H, L, G, W, T, IMPC = form_qp(A, B, Q_mpc, R_mpc, P_mpc, xlim, ulim, N)

    refs = [0.15, -0.15, 0.19, -0.19, -0.25, 0.15, -0.1, 0.0]
    hold = 40
    x_c, u_c, dxprev, t_now = np.zeros(2), 0.0, np.zeros(2), 0.0
    t_h, x1_h, u_h, r_h = [], [], [], []

    for r in refs:
        for _ in range(hold):
            z   = np.array([dxprev[0], dxprev[1], x_c[0]-r, u_c, x_c[0]])
            q   = L @ z
            rhs = W + T @ z
            res = minimize(
                fun=lambda U: 0.5*(U@H@U) + q@U,
                x0=np.zeros(H.shape[0]),
                jac=lambda U: H@U + q,
                method='SLSQP',
                constraints={'type':'ineq','fun':lambda U: rhs-G@U,'jac':lambda U:-G},
                options={'ftol':1e-8,'maxiter':500,'disp':False}
            )
            u_c += float(np.squeeze(IMPC @ res.x))
            sol  = solve_ivp(lambda t,x: msd(t,x,u_c,mp,kp),
                             [t_now, t_now+Ts], x_c, rtol=1e-6, atol=1e-8)
            x_new  = sol.y[:,-1]
            dxprev = x_new - x_c
            t_h.append(t_now); x1_h.append(x_c[0])
            u_h.append(u_c);   r_h.append(r)
            x_c = x_new; t_now += Ts

    t_h.append(t_now); x1_h.append(x_c[0])
    u_h.append(u_c);   r_h.append(refs[-1])
    return tuple(map(np.array, (t_h, x1_h, u_h, r_h)))


# ── Simulations ───────────────────────────────────────────────────────────────
print("\nRunning (e) nominal ...")
t_e,  x1_e,  u_e,  r_e  = run_mpc(1.0, 1.0, 0.20)
print("Running (f) mismatch |u|≤0.20 ...")
t_f1, x1_f1, u_f1, r_f1 = run_mpc(0.8, 1.2, 0.20)
print("Running (f) mismatch |u|≤0.25 ...")
t_f2, x1_f2, u_f2, r_f2 = run_mpc(0.8, 1.2, 0.25)

# ── Plot helpers ──────────────────────────────────────────────────────────────
W_D  = p["figure"]["width_double"]
H_FG = p["figure"]["height"]
LEG  = p["legend"]
BOT  = p["axes"]["bottom_margin"]
SDPI = p["savefig"]["dpi"]

# Different shades of the same color per dataset
c_x1 = ["#1f4e79", "#2e75b6", "#9dc3e6"]   # blue family
c_r  = ["#1a1a1a", "#555555", "#aaaaaa"]   # grey family
c_u  = ["#1e5e1e", "#2e8b2e", "#82c782"]   # green family
c_bd = "#cc0000"                            # constraint bound

titles = ["(e) Nominal", "(f) Mismatch |u|≤0.20", "(f) Mismatch |u|≤0.25"]
sets   = [(t_e,  x1_e,  u_e,  r_e,  0.20),
          (t_f1, x1_f1, u_f1, r_f1, 0.20),
          (t_f2, x1_f2, u_f2, r_f2, 0.25)]

def make_fig(nrows=1):
    fig, axes = plt.subplots(nrows, 3, figsize=(W_D*1.5, H_FG))
    fig.subplots_adjust(bottom=BOT)
    return fig, axes

# ── Figure 1: position ────────────────────────────────────────────────────────
fig1, axes1 = make_fig()
for i, (ax, (t, x1, u, r, ul)) in enumerate(zip(axes1, sets)):
    ax.plot(t, x1, color=c_x1[i], label="$x_1$")
    ax.step(t, r,  color=c_r[i], linestyle='--', where='post', label="$r$")
    ax.axhline( 0.2, color=c_bd, linestyle=':', linewidth=1)
    ax.axhline(-0.2, color=c_bd, linestyle=':', linewidth=1, label="$x_1{=}\\pm0.2$")
    ax.set_title(titles[i]); ax.set_xlabel("Time (s)"); ax.grid(True)
    if i == 0: ax.set_ylabel("Position $x_1$")
    ax.legend(fontsize=LEG["fontsize"], loc=LEG["loc"],
              bbox_to_anchor=LEG["bbox_to_anchor"],
              bbox_transform=fig1.transFigure, ncol=LEG["ncol"])
fig1.savefig(HERE / "mpc_position.png", dpi=SDPI)

# ── Figure 2: control input ───────────────────────────────────────────────────
fig2, axes2 = make_fig()
for i, (ax, (t, x1, u, r, ul)) in enumerate(zip(axes2, sets)):
    ax.step(t, u, color=c_u[i], where='post', label="$u$")
    ax.axhline( ul, color=c_bd, linestyle=':', linewidth=1)
    ax.axhline(-ul, color=c_bd, linestyle=':', linewidth=1, label=f"$u{{=}}\\pm{ul}$")
    ax.set_title(titles[i]); ax.set_xlabel("Time (s)"); ax.grid(True)
    if i == 0: ax.set_ylabel("Control $u$")
    ax.legend(fontsize=LEG["fontsize"], loc=LEG["loc"],
              bbox_to_anchor=LEG["bbox_to_anchor"],
              bbox_transform=fig2.transFigure, ncol=LEG["ncol"])
fig2.savefig(HERE / "mpc_control.png", dpi=SDPI)

print("Plots saved: mpc_position.png, mpc_control.png")

# ── Discussion ────────────────────────────────────────────────────────────────
print("""
(e) r = -0.25 is infeasible (|x1| ≤ 0.2). The solver returns the best feasible
    solution, clamping x1 at -0.2 with nonzero steady-state error.

(f) With mp=0.8, kp=1.2 the nominal model is wrong, causing offset and
    oscillatory transients. |u| ≤ 0.25 gives extra actuation headroom that
    partially compensates for the mismatch, improving settling and reducing
    steady-state error compared to |u| ≤ 0.20.
""")

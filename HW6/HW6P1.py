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
    n, m_ = A.shape[0], B.shape[1]

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

    # Constraints
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
    N     = 15
    Q_mpc = np.diag([0., 0., 1., 0., 0.])
    R_mpc = np.array([[1.]])
    Pdxe  = solve_discrete_are(A[:3,:3], B[:3], np.diag([0.,0.,1.]), R_mpc)
    P_mpc = linalg.block_diag(Pdxe, np.zeros((2,2)))

    lN   = 1e6
    xlim = {"max": np.array([lN, lN, lN,  u_lim,  0.2]),
            "min": np.array([-lN,-lN,-lN, -u_lim, -0.2])}
    ulim = {"max": lN, "min": -lN} 

    H, L, G, W, T, IMPC = form_qp(A, B, Q_mpc, R_mpc, P_mpc, xlim, ulim, N)

    rng   = np.random.default_rng(2)
    vals  = rng.uniform(-0.19, 0.19, 7).tolist()
    holds = rng.integers(25, 60, 7).tolist()
    vals.insert(4, -0.25); holds.insert(4, 70)   # infeasible command in the middle
    refs_holds = list(zip(vals, holds))

    x_c, u_c, dxprev, t_now = np.zeros(2), 0.0, np.zeros(2), 0.0
    t_h, x1_h, u_h, r_h = [], [], [], []

    for r, hold in refs_holds:
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
    u_h.append(u_c);   r_h.append(refs_holds[-1][0])
    return tuple(map(np.array, (t_h, x1_h, u_h, r_h)))


# ── Simulations ───────────────────────────────────────────────────────────────
print("\nRunning (e) nominal ...")
t_e,  x1_e,  u_e,  r_e  = run_mpc(1.0, 1.0, 0.20)
print("Running (f) mismatch |u|≤0.20 ...")
t_f1, x1_f1, u_f1, r_f1 = run_mpc(0.8, 1.2, 0.20)
print("Running (f) mismatch |u|≤0.25 ...")
t_f2, x1_f2, u_f2, r_f2 = run_mpc(0.8, 1.2, 0.25)

# ── Plot helpers ──────────────────────────────────────────────────────────────
W_S  = p["figure"]["width_single"]
H_FG = p["figure"]["height"]
LEG  = p["legend"]
BOT  = p["axes"]["bottom_margin"]
SDPI = p["savefig"]["dpi"]

c_bd  = "#cc0000"
c_e   = "#1f4e79"               # part (e): blue
c_eg  = "#1e5e1e"               # part (e): green
c_f   = ["#2e75b6", "#9dc3e6"]  # part (f): two blue shades
c_fg  = ["#2e8b2e", "#82c782"]  # part (f): two green shades
c_fbd = ["#cc0000", "#cc6600"]  # part (f): constraint bound per case
c_ref = "#555555"

f_data   = [{"t": t_f1, "x1": x1_f1, "u": u_f1, "r": r_f1, "ul": 0.20},
            {"t": t_f2, "x1": x1_f2, "u": u_f2, "r": r_f2, "ul": 0.25}]
f_labels = ["$|u|{\\leq}0.20$", "$|u|{\\leq}0.25$"]

def new_fig():
    fig, ax = plt.subplots(figsize=(W_S, H_FG))
    fig.subplots_adjust(bottom=BOT)
    return fig, ax

def leg(fig, ax):
    ax.legend(fontsize=LEG["fontsize"], loc=LEG["loc"],
              bbox_to_anchor=LEG["bbox_to_anchor"],
              bbox_transform=fig.transFigure, ncol=LEG["ncol"])

# ── (e) position ──────────────────────────────────────────────────────────────
fig, ax = new_fig()
ax.plot(t_e, x1_e, color=c_e, label="$x_1$")
ax.step(t_e, r_e, color=c_ref, linestyle='--', where='post', label="$r$")
ax.axhline( 0.2, color=c_bd, linestyle=':', linewidth=1)
ax.axhline(-0.2, color=c_bd, linestyle=':', linewidth=1, label="$x_1{=}\\pm0.2$")
ax.set_title("(e) Position"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_e_position.png", dpi=SDPI)

# ── (e) tracking error ────────────────────────────────────────────────────────
fig, ax = new_fig()
ax.plot(t_e, x1_e - r_e, color=c_e, label="$e = x_1 - r$")
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, label="$e = 0$")
ax.set_title("(e) Tracking Error"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_e_error.png", dpi=SDPI)

# ── (e) control ───────────────────────────────────────────────────────────────
fig, ax = new_fig()
ax.step(t_e, u_e, color=c_eg, where='post', label="$u$")
ax.axhline( 0.2, color=c_bd, linestyle=':', linewidth=1)
ax.axhline(-0.2, color=c_bd, linestyle=':', linewidth=1, label="$u{=}\\pm0.2$")
ax.set_title("(e) Control"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Control (N)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_e_control.png", dpi=SDPI)

# ── (f) position ──────────────────────────────────────────────────────────────
fig, ax = new_fig()
for i, d in enumerate(f_data):
    ax.plot(d["t"], d["x1"], color=c_f[i], label=f"$x_1$ ({f_labels[i]})")
ax.step(f_data[0]["t"], f_data[0]["r"], color=c_ref, linestyle='--', where='post', label="$r$")
ax.axhline( 0.2, color=c_bd, linestyle=':', linewidth=1)
ax.axhline(-0.2, color=c_bd, linestyle=':', linewidth=1, label="$x_1{=}\\pm0.2$")
ax.set_title("(f) Position"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_f_position.png", dpi=SDPI)

# ── (f) tracking error ────────────────────────────────────────────────────────
fig, ax = new_fig()
for i, d in enumerate(f_data):
    ax.plot(d["t"], d["x1"] - d["r"], color=c_f[i], label=f"$e$ ({f_labels[i]})")
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, label="$e = 0$")
ax.set_title("(f) Tracking Error"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_f_error.png", dpi=SDPI)

# ── (f) control ───────────────────────────────────────────────────────────────
fig, ax = new_fig()
for i, d in enumerate(f_data):
    ax.step(d["t"], d["u"], color=c_fg[i], where='post', label=f"$u$ ({f_labels[i]})")
    ax.axhline( d["ul"], color=c_fbd[i], linestyle=':', linewidth=1, label=f"$\\pm{d['ul']}$")
    ax.axhline(-d["ul"], color=c_fbd[i], linestyle=':', linewidth=1)
ax.set_title("(f) Control"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Control (N)"); ax.grid(True)
leg(fig, ax); fig.savefig(HERE / "mpc_f_control.png", dpi=SDPI)

print("Plots saved: mpc_e_position.png  mpc_e_error.png  mpc_e_control.png")
print("             mpc_f_position.png  mpc_f_error.png  mpc_f_control.png")

# ── Discussion ────────────────────────────────────────────────────────────────
print("""
(e) r = -0.25 is infeasible (|x1| ≤ 0.2). The solver returns the best feasible
    solution, clamping x1 at -0.2 with nonzero steady-state error.

(f) With mp=0.8, kp=1.2 the nominal model is wrong, causing offset and
    oscillatory transients. |u| ≤ 0.25 gives extra actuation headroom that
    partially compensates for the mismatch, improving settling and reducing
    steady-state error compared to |u| ≤ 0.20.
""")

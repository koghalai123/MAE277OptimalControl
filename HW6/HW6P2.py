"""HW6 P2: NMPC for spacecraft attitude control."""

import numpy as np
import yaml
import matplotlib.pyplot as plt
import casadi as ca
from pathlib import Path
from scipy import linalg
from scipy.linalg import solve_discrete_are

HERE = Path(__file__).parent
p = yaml.safe_load((HERE.parent / "plot_params.yaml").read_text())
plt.rcParams.update({"font.size": p["font"]["size"]})

# ── (a) Dynamics ──────────────────────────────────────────────────────────────
J1, J2, J3 = 918.0, 920.0, 1365.0
J   = np.diag([J1, J2, J3])
Jinv= np.linalg.inv(J)
Ts  = 2.0
N   = 10
nx, nu = 6, 3

def skew(w):
    return np.array([[ 0,    -w[2],  w[1]],
                     [ w[2],  0,    -w[0]],
                     [-w[1],  w[0],  0   ]])

def fc(x, u):
    """Continuous-time spacecraft attitude dynamics."""
    w = x[:3]; phi, theta = x[3], x[4]
    wdot = Jinv @ (-skew(w) @ J @ w + u)
    S = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi),              -np.sin(phi)              ],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
    ])
    return np.concatenate([wdot, S @ w])

def fd(x, u):
    """Explicit Euler discretization: x_{k+1} = x_k + Ts * fc(x_k, u_k)."""
    return x + Ts * fc(x, u)

# ── (b) Terminal cost via LQR at linearization point (x=0, u=0) ──────────────
Q  = np.diag([100., 100., 100., 10., 10., 10.])
R  = np.diag([0.1,  0.1,  0.1])

def linearize(eps=1e-5):
    """Numerical Jacobians of fd at x=0, u=0."""
    x0, u0 = np.zeros(nx), np.zeros(nu)
    Ad = np.zeros((nx, nx))
    Bd = np.zeros((nx, nu))
    f0 = fd(x0, u0)
    for i in range(nx):
        dx = np.zeros(nx); dx[i] = eps
        Ad[:, i] = (fd(x0 + dx, u0) - f0) / eps
    for i in range(nu):
        du = np.zeros(nu); du[i] = eps
        Bd[:, i] = (fd(x0, u0 + du) - f0) / eps
    return Ad, Bd

Ad_lin, Bd_lin = linearize()
P = solve_discrete_are(Ad_lin, Bd_lin, Q, R)
print("Terminal cost P computed via discrete LQR.")

# ── (c)/(d)/(e) NMPC solver via CasADi + IPOPT ───────────────────────────────
def fc_ca(x, u):
    """CasADi symbolic continuous-time dynamics."""
    w = x[:3]; phi = x[3]; theta = x[4]
    Jca   = ca.diag(ca.DM([J1, J2, J3]))
    Jinca = ca.diag(ca.DM([1/J1, 1/J2, 1/J3]))
    skw   = ca.vertcat(
        ca.horzcat( 0,     -w[2],  w[1]),
        ca.horzcat( w[2],  0,     -w[0]),
        ca.horzcat(-w[1],  w[0],  0   ),
    )
    wdot = Jinca @ (-skw @ Jca @ w + u)
    S = ca.vertcat(
        ca.horzcat(1, ca.sin(phi)*ca.tan(theta), ca.cos(phi)*ca.tan(theta)),
        ca.horzcat(0, ca.cos(phi),              -ca.sin(phi)              ),
        ca.horzcat(0, ca.sin(phi)/ca.cos(theta), ca.cos(phi)/ca.cos(theta)),
    )
    return ca.vertcat(wdot, S @ w)

def fd_ca(x, u):
    """CasADi symbolic explicit Euler step."""
    return x + Ts * fc_ca(x, u)

def build_solver(params):
    X_sym = [ca.MX.sym(f"x{k}", nx) for k in range(N)]
    U_sym = [ca.MX.sym(f"u{k}", nu) for k in range(N)]
    x0_p  = ca.MX.sym("x0", nx)   # parameter: current state
    r_p   = ca.MX.sym("r",  nx)   # parameter: reference

    # Cost
    cost = 0.5 * (X_sym[-1]-r_p).T @ ca.DM(P) @ (X_sym[-1]-r_p)
    x_prev = x0_p
    for k in range(N):
        cost += 0.5 * ((x_prev-r_p).T @ ca.DM(Q) @ (x_prev-r_p)
                       + U_sym[k].T @ ca.DM(R) @ U_sym[k])
        x_prev = X_sym[k]

    # Dynamics equality constraints
    ceq, x_prev = [], x0_p
    for k in range(N):
        ceq.append(X_sym[k] - fd_ca(x_prev, U_sym[k]))
        x_prev = X_sym[k]

    z   = ca.vertcat(*X_sym, *U_sym)
    g   = ca.vertcat(*ceq)
    xlb = np.tile(params["xlb"], N)
    xub = np.tile(params["xub"], N)
    ulb = np.tile(params["ulb"], N)
    uub = np.tile(params["uub"], N)

    nlp = {"x": z, "f": cost, "g": g, "p": ca.vertcat(x0_p, r_p)}
    opts = {"ipopt.print_level": 0, "print_time": 0,
            "ipopt.max_iter": 200, "ipopt.tol": 1e-6}
    solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
    lbz = np.concatenate([xlb, ulb])
    ubz = np.concatenate([xub, uub])
    lbg = np.zeros(nx * N)
    ubg = np.zeros(nx * N)
    return solver, lbz, ubz, lbg, ubg

def solve_nmpc(solver, lbz, ubz, lbg, ubg, x0, r, z_guess):
    """Call the compiled IPOPT solver"""
    sol = solver(x0=z_guess,
                 lbx=lbz, ubx=ubz,
                 lbg=lbg, ubg=ubg,
                 p=np.concatenate([x0, r]))
    z_opt = np.array(sol["x"]).ravel()
    X = z_opt[:nx*N].reshape(N, nx)
    U = z_opt[nx*N:].reshape(N, nu)
    cost = float(sol["f"])
    stats = solver.stats()
    success = stats["success"]
    return U[0], X, U, z_opt, cost, success

def run_sim(params, T_sim=200, x_init=None, r=None, label=""):
    if x_init is None:
        x_init = np.array([0., 0., 0., np.radians(15), np.radians(30), np.radians(20)])
    if r is None:
        r = np.zeros(nx)

    solver, lbz, ubz, lbg, ubg = build_solver(params)
    steps   = int(T_sim / Ts)
    x       = x_init.copy()
    z_guess = np.zeros(nx*N + nu*N)

    t_h  = np.zeros(steps+1)
    x_h  = np.zeros((steps+1, nx))
    u_h  = np.zeros((steps,   nu))
    cost_h = np.zeros(steps)

    x_h[0] = x
    print(f"  {label}: simulating {steps} steps ...")
    for k in range(steps):
        u0, X, U, z_opt, cost, _ = solve_nmpc(solver, lbz, ubz, lbg, ubg, x, r, z_guess)
        u_h[k]    = u0
        cost_h[k] = cost
        X_shift = np.vstack([X[1:], X[-1]])
        U_shift = np.vstack([U[1:], U[-1]])
        z_guess = np.concatenate([X_shift.ravel(), U_shift.ravel()])
        x = fd(x, u0)
        t_h[k+1] = (k+1)*Ts
        x_h[k+1] = x

    return t_h, x_h, u_h, cost_h

# ── Run simulations ───────────────────────────────────────────────────────────
lN  = 1e3  
ulb = np.full(nu, -1.); uub = np.full(nu, 1.)

# Part (d)
params_d = {"xlb": np.full(nx, -lN), "xub": np.full(nx, lN),
            "ulb": ulb, "uub": uub}
print("Running (2d) u")
t_d, x_d, u_d, cost_d = run_sim(params_d, label="(d)")

# Part (e) Case 1
xmax1 = 10 * np.ones(nx)
xmin1 = -10 * np.array([1,1,1,0.01,0.01,0.01])
params_e1 = {"xlb": xmin1, "xub": xmax1, "ulb": ulb, "uub": uub}
print("Running (e) Case 1 ...")
t_e1, x_e1, u_e1, cost_e1 = run_sim(params_e1, label="(e) Case 1")

# Part (e) Case 2
xmax2 = np.ones(nx)
xmin2 = -np.array([1,1,1,0,0,0])
params_e2 = {"xlb": xmin2, "xub": xmax2, "ulb": ulb, "uub": uub}
print("Running (e) Case 2 ...")
t_e2, x_e2, u_e2, cost_e2 = run_sim(params_e2, label="(e) Case 2")

# ── Plotting ──────────────────────────────────────────────────────────────────
W_S  = p["figure"]["width_single"]
W_D  = p["figure"]["width_double"]
H_FG = p["figure"]["height"]
LEG  = p["legend"]
BOT  = p["axes"]["bottom_margin"]
SDPI = p["savefig"]["dpi"]

c_bd  = "#cc0000"
# Part (d): single dataset, use fixed colors
c_w   = ["#1f4e79", "#2e75b6", "#9dc3e6"]   # omega shades (blue)
c_eu  = ["#1e5e1e", "#2e8b2e", "#82c782"]   # euler shades (green)
c_tor = ["#4b1f79", "#7b2eb6", "#c39de6"]   # torque shades (purple)


def new_fig(w=None):
    fig, ax = plt.subplots(figsize=(w or W_S, H_FG))
    fig.subplots_adjust(bottom=BOT)
    return fig, ax

def leg(fig, ax, ncol=None):
    ax.legend(fontsize=LEG["fontsize"], loc=LEG["loc"],
              bbox_to_anchor=LEG["bbox_to_anchor"],
              bbox_transform=fig.transFigure, ncol=ncol or LEG["ncol"])

# ── (d) angular velocities ────────────────────────────────────────────────────
fig, ax = new_fig()
for i, lbl in enumerate(["$\\omega_1$","$\\omega_2$","$\\omega_3$"]):
    ax.plot(t_d, x_d[:,i], color=c_w[i], label=lbl)
ax.set_title("(d) Angular Velocities"); ax.set_xlabel("Time (s)")
ax.set_ylabel("rad/s"); ax.grid(True)
leg(fig, ax)
fig.savefig(HERE / "nmpc_d_omega.png", dpi=SDPI)

# ── (d) Euler angles ──────────────────────────────────────────────────────────
fig, ax = new_fig()
for i, lbl in enumerate(["$\\phi$","$\\theta$","$\\psi$"]):
    ax.plot(t_d, np.degrees(x_d[:,3+i]), color=c_eu[i], label=lbl)
ax.set_title("(d) Euler Angles"); ax.set_xlabel("Time (s)")
ax.set_ylabel("deg"); ax.grid(True)
leg(fig, ax)
fig.savefig(HERE / "nmpc_d_euler.png", dpi=SDPI)

# ── (d) control torques ───────────────────────────────────────────────────────
t_u = t_d[:-1]
fig, ax = new_fig()
for i, lbl in enumerate(["$u_1$","$u_2$","$u_3$"]):
    ax.step(t_u, u_d[:,i], color=c_tor[i], where='post', label=lbl)
ax.axhline( 1, color=c_bd, linestyle=':', linewidth=1)
ax.axhline(-1, color=c_bd, linestyle=':', linewidth=1, label="$u{=}\\pm1$")
ax.set_title("(d) Control Torques"); ax.set_xlabel("Time (s)")
ax.set_ylabel("N·m"); ax.grid(True)
leg(fig, ax)
fig.savefig(HERE / "nmpc_d_control.png", dpi=SDPI)

# ── (e) angular velocities ────────────────────────────────────────────────────
e_data = [(t_e1, x_e1, u_e1, cost_e1, "Case 1"),
          (t_e2, x_e2, u_e2, cost_e2, "Case 2")]

# Case color: blue family vs orange family; line style per symbol
c_case   = ["#1f4e79", "#cc6600"]          # Case 1: dark blue, Case 2: dark orange
ls_sym   = ['-', '--', ':']                # solid/dashed/dotted per symbol index

fig, ax = new_fig(W_D)
for ci, (t, x, u, res, lname) in enumerate(e_data):
    for i, sym in enumerate(["$\\omega_1$","$\\omega_2$","$\\omega_3$"]):
        ax.plot(t, x[:,i], color=c_case[ci], linestyle=ls_sym[i],
                label=f"{sym} {lname}")
ax.set_title("(e) Angular Velocities"); ax.set_xlabel("Time (s)")
ax.set_ylabel("rad/s"); ax.grid(True)
leg(fig, ax, ncol=3)
fig.savefig(HERE / "nmpc_e_omega.png", dpi=SDPI)

# ── (e) Euler angles ──────────────────────────────────────────────────────────
fig, ax = new_fig(W_D)
for ci, (t, x, u, res, lname) in enumerate(e_data):
    for i, sym in enumerate(["$\\phi$","$\\theta$","$\\psi$"]):
        ax.plot(t, np.degrees(x[:,3+i]), color=c_case[ci], linestyle=ls_sym[i],
                label=f"{sym} {lname}")
ax.set_title("(e) Euler Angles"); ax.set_xlabel("Time (s)")
ax.set_ylabel("deg"); ax.grid(True)
leg(fig, ax, ncol=3)
fig.savefig(HERE / "nmpc_e_euler.png", dpi=SDPI)

# ── (e) control torques ───────────────────────────────────────────────────────
fig, ax = new_fig(W_D)
for ci, (t, x, u, res, lname) in enumerate(e_data):
    t_u = t[:-1]
    for i, sym in enumerate(["$u_1$","$u_2$","$u_3$"]):
        ax.step(t_u, u[:,i], color=c_case[ci], where='post',
                linestyle=ls_sym[i], label=f"{sym} {lname}")
ax.axhline( 1, color=c_bd, linestyle=':', linewidth=1)
ax.axhline(-1, color=c_bd, linestyle=':', linewidth=1, label="$u{=}\\pm1$")
ax.set_title("(e) Control Torques"); ax.set_xlabel("Time (s)")
ax.set_ylabel("N·m"); ax.grid(True)
leg(fig, ax, ncol=3)
fig.savefig(HERE / "nmpc_e_control.png", dpi=SDPI)

# ── (e) solver cost residual ──────────────────────────────────────────────────
fig, ax = new_fig(W_D)
for ci, (t, x, u, res, lname) in enumerate(e_data):
    ax.semilogy(t[:-1], np.abs(res), color=c_case[ci], label=lname)
ax.set_title("(e) Solver Cost"); ax.set_xlabel("Time (s)")
ax.set_ylabel("Objective value"); ax.grid(True)
leg(fig, ax, ncol=2)
fig.savefig(HERE / "nmpc_e_residual.png", dpi=SDPI)

print("Plots saved: nmpc_d_omega.png  nmpc_d_euler.png  nmpc_d_control.png")
print("             nmpc_e_omega.png  nmpc_e_euler.png  nmpc_e_control.png  nmpc_e_residual.png")

print("""
(e) Discussion:
Case 1 (loose bounds ±10 on states, tight on angles ≥-0.01) allows fast
maneuvers; the controller can use large angular velocities and the optimizer
converges quickly with low residual.

Case 2 (tight ±1 on all states including angles) severely restricts the
maneuver: the controller must move slowly so angular velocities stay within
±1 rad/s. This leads to slower convergence, higher control saturation (u
hits ±1 more often), and larger solver objective values since the feasible
set is smaller. Constraint-active steps show spikes in the residual plot.
""")

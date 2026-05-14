from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.drone.trajectories import Trajectory, make_trajectory

G = 9.81


@dataclass(frozen=True)
class Ch4Params:
    # quadrotor parameters (InitFcn)
    m: float = 1.0
    C: float = 1.0
    J_phi: float = 1.0
    l: float = 1.0
    J_theta: float = 1.0
    J_psi: float = 1.0

    # controller parameters (InitFcn)
    kappa: float = 100.0
    a_1: float = 5.0
    a_2: float = 10.0
    a_3: float = 10.0
    a_4: float = 5.0
    a_5: float = 1.0
    gamma_1: float = 1.0
    gamma_2: float = 4.0
    gamma_3: float = 6.0
    gamma_4: float = 4.0
    L: float = 5.0
    ell: float = 0.9

    traj: Trajectory | None = None
    traj_name: str = "line"

    # optional: want constant spatial speed along curve
    use_const_spatial_speed: bool = False
    V_star: float = 2.0  # m/s (used if use_const_spatial_speed=True)

    # reference speed (comes from s_h1 integrator in the provided model – effectively constant)
    v_ref: float = 1.0
    s_h0: float = 2.0 / 3.0  # InitialCondition of s_h integrator in root


def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def accel_matlab(u: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Порт MATLAB Function из Quadrotor/MATLAB Function (chart_112).

    output = [x y z phi theta psi]
    u      = [u1 u2 u3 u4] (виртуальные управления, БЕЗ добавленного g)
    """
    u = np.asarray(u, dtype=float).reshape(4)
    output = np.asarray(output, dtype=float).reshape(6)

    phi = float(output[3])
    theta = float(output[4])
    psi = float(output[5])

    g = G
    d = float(u[0]) + g

    accel_lin = np.array([
        np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
        np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
        np.cos(theta) * np.cos(psi),
    ], dtype=float) * d + np.array([0.0, 0.0, -g], dtype=float)

    accel_ang = np.array([u[1], u[2], u[3]], dtype=float)

    return np.concatenate([accel_lin, accel_ang], axis=0)


def b_and_inv(output: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Порт MATLAB Function из Controller/MATLAB Function (chart_103), flag=full."""
    output = np.asarray(output, dtype=float).reshape(6)
    u = np.asarray(u, dtype=float).reshape(4)

    phi = float(output[3])
    theta = float(output[4])
    psi = float(output[5])

    ct = np.cos(theta)
    st = np.sin(theta)

    cp = np.cos(psi)
    sp = np.sin(psi)

    g = G
    d = float(u[0]) + g

    Rphi = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, np.cos(phi), -np.sin(phi)],
        [0.0, 0.0, np.sin(phi),  np.cos(phi)],
    ], dtype=float)

    B0 = np.array([
        [ct * cp,      0.0, -st * cp * d, -ct * sp * d],
        [0.0,          1.0,  0.0,          0.0],
        [st * cp,      0.0,  ct * cp * d, -st * sp * d],
        [-sp,          0.0,  0.0,         -cp * d],
    ], dtype=float)

    b = Rphi @ B0
    b_inv = np.linalg.inv(b)
    return b, b_inv


def prop_forces_from_u(u: np.ndarray, p: Ch4Params) -> np.ndarray:
    """Порт Gain блока в Controller: 1/4*[...]*M^(-1) * (u + [g,0,0,0])."""
    u = np.asarray(u, dtype=float).reshape(4)

    M = np.diag([1.0 / p.m, p.C / p.J_phi, p.l / p.J_theta, p.l / p.J_psi])
    Ainv = 0.25 * np.array([
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, 1, 1, 1],
        [1, -1, -1, 1],
    ], dtype=float)

    return (Ainv @ np.linalg.inv(M)) @ (u + np.array([G, 0.0, 0.0, 0.0], dtype=float))


@dataclass
class Ch4State:
    """Полное состояние замкнутой системы (растянутое состояние Simulink модели 23.slx).

    Plant:
      q    = [x, y, z, phi, theta, psi]          (6)
      qdot = [xdot, ydot, zdot, phidot, thetadot, psidot] (6)

    Controller internal chain (each is 4D vector):
      z1..z5 correspond to Integrator4..Integrator8 inside Controller subsystem.

    Dynamic extension for u1/u2:
      p1, u1_bar (Integrator1 and Integrator in Controller, producing u1 (sat) output)
      p2, u2     (Integrator3 and Integrator2 in Controller)
    """
    q: np.ndarray
    qdot: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    z3: np.ndarray
    z4: np.ndarray
    z5: np.ndarray
    p1: float
    u1_bar: float
    p2: float
    u2: float
    s_h: float  # root integrator s_h (reference parameter)

    @staticmethod
    def default() -> "Ch4State":
        # Initial conditions from Quadrotor/Integrator block:
        q0 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=float)
        qdot0 = np.zeros(6, dtype=float)

        # Controller integrators are not specified in the model; start from zeros.
        z = np.zeros(4, dtype=float)
        return Ch4State(
            q=q0,
            qdot=qdot0,
            z1=z.copy(),
            z2=z.copy(),
            z3=z.copy(),
            z4=z.copy(),
            z5=z.copy(),
            p1=0.0,
            u1_bar=0.0,
            p2=0.0,
            u2=0.0,
            s_h=2.0/3.0,
        )

    def pack(self) -> np.ndarray:
        return np.concatenate([
            self.q,
            self.qdot,
            self.z1, self.z2, self.z3, self.z4, self.z5,
            np.array([self.p1, self.u1_bar, self.p2, self.u2, self.s_h], dtype=float),
        ])

    @staticmethod
    def unpack(x: np.ndarray) -> "Ch4State":
        x = np.asarray(x, dtype=float).reshape(-1)
        q = x[0:6]
        qdot = x[6:12]
        i = 12
        z1 = x[i:i+4]; i += 4
        z2 = x[i:i+4]; i += 4
        z3 = x[i:i+4]; i += 4
        z4 = x[i:i+4]; i += 4
        z5 = x[i:i+4]; i += 4
        p1, u1_bar, p2, u2, s_h = x[i:i+5]
        return Ch4State(q=q, qdot=qdot, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5,
                        p1=float(p1), u1_bar=float(u1_bar), p2=float(p2), u2=float(u2), s_h=float(s_h))


def controller_virtual_u_and_F(st: Ch4State, p: Ch4Params) -> tuple[np.ndarray, np.ndarray, dict]:
    """Один алгебраический шаг контроллера (без интегрирования внутренних состояний).

    Возвращает:
      u = [u1, u2, u3, u4]  (u1 берётся С САТУРАЦИЕЙ ell*g как в блоке Saturation1)
      F = [F1..F4]
      dbg = полезные промежуточные сигналы
    """
    # Reference in the provided model (chart_131): x_h=y_h=z_h=s_h
    traj = p.traj if p.traj is not None else make_trajectory(p.traj_name)
    r, dr = traj.eval(st.s_h)
    x_h, y_h, z_h = float(r[0]), float(r[1]), float(r[2])

    # Errors (root sums): Xtil = X - x_h, etc; phitil = phi - 0
    Xtil = float(st.q[0] - x_h)
    Ytil = float(st.q[1] - y_h)
    Ztil = float(st.q[2] - z_h)
    phitil = float(st.q[3] - 0.0)

    # xi1tilde ordering (Subsystem3): [Ztil; d; Xtil; Ytil], with d = phitil (chart_141)
    xi1tilde = np.array([Ztil, phitil, Xtil, Ytil], dtype=float)

    # Error for high-gain chain: e = xi1tilde - z1
    e = xi1tilde - st.z1

    k = p.kappa
    # derivatives of chain states
    dz5 = (k**5) * p.a_5 * e
    dz4 = st.z5 + (k**4) * p.a_4 * e
    dz3 = st.z4 + (k**3) * p.a_3 * e
    dz2 = st.z3 + (k**2) * p.a_2 * e
    dz1 = st.z2 + (k**1) * p.a_1 * e

    # v = -(z5 + gamma4*z4 + gamma3*z3 + gamma2*z2 + gamma1*z1)  (Sum6 has '-----')
    v = -(st.z5 + p.gamma_4 * st.z4 + p.gamma_3 * st.z3 + p.gamma_2 * st.z2 + p.gamma_1 * st.z1)

    # current u vector for b calculation uses u1 output (sat) and u2,u3,u4
    u1 = float(clip(np.array([st.u1_bar]), -p.ell*G, p.ell*G)[0])
    u_curr = np.array([u1, st.u2, 0.0, 0.0], dtype=float)  # u3,u4 placeholders for b matrix (not needed in b)

    _, b_inv = b_and_inv(st.q, u_curr)
    w = b_inv @ v
    w_sat = clip(w, -p.L, p.L)  # Saturation block

    v1, v2, u3, u4 = map(float, w_sat)

    u = np.array([u1, st.u2, u3, u4], dtype=float)
    F = prop_forces_from_u(u, p)

    dbg = {
        "xi1tilde": xi1tilde,
        "e": e,
        "v": v,
        "w": w,
        "w_sat": w_sat,
        "x_h": x_h, "y_h": y_h, "z_h": z_h,
    }
    derivs = {"dz1": dz1, "dz2": dz2, "dz3": dz3, "dz4": dz4, "dz5": dz5, "v1": v1, "v2": v2}
    dbg.update(derivs)
    return u, F, dbg


def closed_loop_ode(t: float, x: np.ndarray, p: Ch4Params) -> np.ndarray:
    """Правая часть ОДУ замкнутой системы, строго по структуре 23.slx (глава 4)."""
    st = Ch4State.unpack(x)

    # Controller algebraic outputs
    u, _, dbg = controller_virtual_u_and_F(st, p)

    # Plant dynamics: qddot = accel_matlab(u, q)
    qddot = accel_matlab(u, st.q)

    # Controller internal derivatives (from dbg)
    dz1 = dbg["dz1"]; dz2 = dbg["dz2"]; dz3 = dbg["dz3"]; dz4 = dbg["dz4"]; dz5 = dbg["dz5"]
    v1 = float(dbg["v1"]); v2 = float(dbg["v2"])

    # Dynamic extension for u1/u2 (two integrators each)
    dp1 = v1
    du1_bar = st.p1
    dp2 = v2
    du2 = st.p2

    # Reference integrator s_h: in the provided model it's driven by s_h1 output,
    # which is effectively constant 1 (no input to s_h1 integrator is wired).
    if p.use_const_spatial_speed:
        # ds = V* / ||dr/ds||
        # где-то выше уже есть доступ к st.s_h
        traj = p.traj if getattr(p, "traj", None) is not None else make_trajectory(getattr(p, "traj_name", "line"))
        _, dr = traj.eval(float(st.s_h))  # <-- ВАЖНО: именно здесь появится dr

        if getattr(p, "use_const_spatial_speed", False):
            speed_gain = float(np.linalg.norm(dr))
            ds_h = (p.V_star / max(speed_gain, 1e-6))
        else:
            ds_h = p.v_ref
        speed_gain = float(np.linalg.norm(dr))
        # ds_h = (p.V_star / max(speed_gain, 1e-6))
    else:
        ds_h = p.v_ref

    dx = np.zeros_like(x)
    # qdot
    dx[0:6] = st.qdot
    # qddot
    dx[6:12] = qddot
    # controller chain
    i = 12
    dx[i:i+4] = dz1; i += 4
    dx[i:i+4] = dz2; i += 4
    dx[i:i+4] = dz3; i += 4
    dx[i:i+4] = dz4; i += 4
    dx[i:i+4] = dz5; i += 4
    # extension + s_h
    dx[i:i+5] = np.array([dp1, du1_bar, dp2, du2, ds_h], dtype=float)
    return dx


def rk4(f, x0: np.ndarray, t0: float, t1: float, dt: float, p: Ch4Params):
    n = int(np.ceil((t1 - t0) / dt))
    ts = np.linspace(t0, t0 + n*dt, n+1)
    xs = np.zeros((n+1, x0.size), dtype=float)
    xs[0] = x0
    for k in range(n):
        t = ts[k]
        x = xs[k]
        k1 = f(t, x, p)
        k2 = f(t + dt/2, x + dt/2*k1, p)
        k3 = f(t + dt/2, x + dt/2*k2, p)
        k4 = f(t + dt, x + dt*k3, p)
        xs[k+1] = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return ts, xs


def simulate(t1: float = 20.0, dt: float = 0.001, params: Ch4Params | None = None, x0: Ch4State | None = None):
    p = params or Ch4Params()
    st0 = x0 or Ch4State.default()
    x_init = st0.pack()
    ts, xs = rk4(closed_loop_ode, x_init, 0.0, t1, dt, p)

    # unpack key signals
    q = xs[:, 0:6]
    # reconstruct reference
    s_h = xs[:, -1]
    traj = p.traj if p.traj is not None else make_trajectory(p.traj_name)
    ref = np.zeros((len(s_h), 3), dtype=float)
    for i, s in enumerate(s_h):
        r, _ = traj.eval(float(s))
        ref[i, :] = r

    return {
        "t": ts,
        "q": q,
        "ref": ref,
        "s_h": s_h,
    }

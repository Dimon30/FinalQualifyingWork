from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from dynamics import thrust_direction, G, sat_tanh
from controllers.common import HighGainParams, DerivativeObserver4, sat_vec_tanh

@dataclass
class TrackingRef:
    y: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    y3: np.ndarray
    y4: np.ndarray

def fd_grad_hess_b(phi,theta,psi, h=1e-5):
    a = np.array([phi,theta,psi], dtype=float)
    def bfun(aa):
        return thrust_direction(aa[0],aa[1],aa[2])
    grad = np.zeros((3,3))
    for i in range(3):
        ap = a.copy(); ap[i]+=h
        am = a.copy(); am[i]-=h
        grad[:,i] = (bfun(ap) - bfun(am))/(2*h)
    hess = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            ap = a.copy(); ap[i]+=h; ap[j]+=h
            am = a.copy(); am[i]-=h; am[j]-=h
            apm = a.copy(); apm[i]+=h; apm[j]-=h
            amp = a.copy(); amp[i]-=h; amp[j]+=h
            hess[:,i,j] = (bfun(ap)-bfun(apm)-bfun(amp)+bfun(am))/(4*h*h)
    return bfun(a), grad, hess

class Ch3OutputTrackingController:
    def __init__(self, params: HighGainParams):
        self.p = params
        self.obs = DerivativeObserver4(dim=4, p=params)

    def _y4_affine(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi,theta,psi = state[6:9]
        phidot,thetadot,psidot = state[9:12]
        u1_bar, rho1, u2 = state[12], state[13], state[14]

        u1 = sat_tanh(u1_bar, self.p.L)

        b0, grad, hess = fd_grad_hess_b(phi,theta,psi)
        adot = np.array([phidot,thetadot,psidot], dtype=float)
        addot_base = np.array([u2, 0.0, 0.0], dtype=float)  # исключаем u3,u4 из q

        b_dot = grad @ adot
        quad = np.zeros(3)
        for i in range(3):
            for j in range(3):
                quad += hess[:,i,j]*adot[i]*adot[j]
        b_ddot_base = grad @ addot_base + quad

        q_pos = b_ddot_base*(u1+G) + 2*b_dot*rho1

        B = np.zeros((4,4))
        B[0:3,0] = b0
        B[3,1] = 1.0
        B[0:3,2] = grad[:,1]*(u1+G)  # u3
        B[0:3,3] = grad[:,2]*(u1+G)  # u4

        q = np.zeros(4)
        q[0:3] = q_pos
        return q, B

    def step(self, t: float, state: np.ndarray, ref: TrackingRef, dt: float) -> np.ndarray:
        y = np.array([state[0], state[1], state[2], state[6]], dtype=float)
        q0,_B0 = self._y4_affine(state)
        self.obs.step(y=y, y4_model=q0, dt=dt)
        x1,x2,x3,x4,sigma = self.obs.hat()

        g1,g2,g3,g4,_ = self.p.gamma
        v = (ref.y4
             - sigma
             - g1*(x1 - ref.y)
             - g2*(x2 - ref.y1)
             - g3*(x3 - ref.y2)
             - g4*(x4 - ref.y3))

        q,B = self._y4_affine(state)
        U = np.linalg.lstsq(B, (v - q), rcond=None)[0]
        return sat_vec_tanh(U, self.p.L)

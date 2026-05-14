from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_out(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def plot_3d_traj(p_ref, p, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # if p_ref is not None:
    #     ax.plot(p_ref[:,0], p_ref[:,1], p_ref[:,2], linestyle="--", label="Заданная")
    ax.plot(p[:,0], p[:,1], p[:,2])
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m"); ax.set_zlabel("z, m")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_errors(t, e, labels, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,lab in enumerate(labels):
        ax.plot(t, e[:,i], label=lab)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

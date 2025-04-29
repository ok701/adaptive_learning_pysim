#!/usr/bin/env python3
"""
Safe-Bayesian-Optimization 데모
────────────────────────────────────────────
목표   : lead-margin Δ, 스프링 강성 K 를 BO 로 찾아
        사용자 기계적 일 W_user 를 최대화

시뮬   : · 0→1 minimum-jerk 궤적  (T = 10 s)
        · ref ≥ 0.5 일 때 사용자 힘 +1 N 스텝
        · Δ 를 처음 초과하면 저항 스프링 ON  →  마지막까지 유지
        · 저항 힘  =  K·(lead−Δ)  +  B·(v_act−v_ref)   ← 선형 댐핑 포함

BO     : 2-D Gaussian Process  +  UCB(β=2)   (안전박스  Δ∈[100,1000], K∈[0,20])
그래프 : ① Expected-Improvement ② 궤적 ③ 사용자 힘·K 타임라인
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# ───────── 시뮬레이션 상수 ──────────────────────────
T, dt  = 10.0, 0.001
t      = np.arange(0, T, dt)
F_STEP = 1.0                # 사용자 힘 스텝 [N]
MASS   = 1.0                # 가상 질량
B_DAMP = 4.0                # ★ 댐핑 계수 [N·s/m]

BOX = np.array([[100, 1000],   # Δ  범위  (단위: mm, 0–1 m 정규화 ×1000)
                [  0,   20]])  # K  범위  (N/m)

BETA = 2.0                   # UCB 베타

# ───────── 헬퍼 함수 ────────────────────────────────
def min_jerk(s):                           # s ∈ [0,1]
    return 10*s**3 - 15*s**4 + 6*s**5

def simulate(Δ_mm: float, K: float):
    """
    Δ_mm : lead-margin [mm]   (e.g. 100 → 0.1 m)
    K    : spring stiffness [N/m]
    반환 : W_user, ref, act, F_user(t), mask_resist(t)
    """
    Δ = Δ_mm / 1000.0                      # m 로 변환
    x_ref = min_jerk(t / T)
    v_ref = np.gradient(x_ref, dt)

    x_act = np.zeros_like(t)
    v_act = np.zeros_like(t)
    F_user = (x_ref > 0.5) * F_STEP
    F_res  = np.zeros_like(t)
    mask   = np.zeros_like(t, dtype=bool)

    resist_on = False
    for k in range(1, len(t)):
        lead = x_act[k-1] - x_ref[k-1]

        if not resist_on and lead > Δ:
            resist_on = True                      # 최초 ON

        if resist_on:
            rel_vel   = v_act[k-1] - v_ref[k-1]
            F_res[k]  = K*(lead - Δ) + B_DAMP*rel_vel
            mask[k]   = True

        a          = (F_user[k] - F_res[k]) / MASS
        v_act[k]   = v_act[k-1] + a*dt
        x_act[k]   = x_act[k-1] + v_act[k]*dt

    W_user = np.trapezoid(F_user * v_act, dx=dt)
    return W_user, x_ref, x_act, F_user, mask

def expected_improv(X, Y, gp, cand):
    mu, sig = gp.predict(cand, return_std=True)
    sig = np.maximum(sig, 1e-9)
    imp = np.min(Y) - mu
    Z   = imp / sig
    ei  = imp * norm.cdf(Z) + sig * norm.pdf(Z)
    ei[sig == 0] = 0
    return ei

# ───────── BO 객체 초기화 ───────────────────────────
kernel = Matern(nu=2.5)
gp     = GaussianProcessRegressor(kernel, alpha=1e-6, normalize_y=True)

X_data, Y_data = [], []
init_points = np.array([[100,  5],    # Δ=100 mm, K=5 N/m
                        [500, 15]])   # Δ=500 mm, K=15 N/m

# ───────── BO 루프 (5 세트) ─────────────────────────
for run in range(5):

    # Δ, K 선택 ─────
    if run < len(init_points):
        Δ_sel, K_sel = init_points[run]
    else:
        Δ_grid = np.linspace(*BOX[0], 40)
        K_grid = np.linspace(*BOX[1], 40)
        grid   = np.array([[d, k] for d in Δ_grid for k in K_grid])  # (n,2)

        mu, sig = gp.predict(grid, return_std=True)
        ucb     = mu - BETA * sig
        Δ_sel, K_sel = grid[np.argmin(ucb)]

    # 시뮬레이션 ─────
    W, x_ref, x_act, F_user, mask = simulate(Δ_sel, K_sel)
    X_data.append([Δ_sel, K_sel])
    Y_data.append(-W)                       # 비용 = –W_user
    gp.fit(np.array(X_data), np.array(Y_data))

    print(f"run {run+1}: Δ={Δ_sel:.0f} mm  K={K_sel:.1f} N/m  "
          f"W_user={W:.4f} J")

    # ──────── 시각화 ───────────────────────────────
    Δ_lin = np.linspace(*BOX[0], 40)
    K_lin = np.linspace(*BOX[1], 40)
    D, K_mesh = np.meshgrid(Δ_lin, K_lin)
    EI = expected_improv(X_data, Y_data, gp,
                         np.c_[D.ravel(), K_mesh.ravel()]).reshape(D.shape)

    fig = plt.figure(figsize=(11, 4.5))

    # (1) Expected Improvement
    ax1 = fig.add_subplot(1, 3, 1)
    cs  = ax1.contourf(D, K_mesh, EI, 20, cmap=cm.viridis)
    ax1.scatter(*np.array(X_data).T, c='red')
    ax1.set(xlabel='Δ [mm]', ylabel='K [N/m]', title='Expected Improvement')
    fig.colorbar(cs, ax=ax1)

    # (2) Trajectory
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(t, x_ref, label='ref')
    ax2.plot(t, x_act, label='act')
    ax2.set(xlabel='time [s]', ylabel='pos',
            title=f'Trajectory  (run {run+1})')
    ax2.legend()

    # (3) Force & K timeline
    ax3  = fig.add_subplot(1, 3, 3)
    ax3.plot(t, F_user, label='F_user')
    ax3.set(xlabel='time [s]', ylabel='F_user [N]',
            title='User force & K timeline')

    ax3b = ax3.twinx()
    K_vis = np.where(mask, K_sel, 0.0)
    ax3b.plot(t, K_vis, 'r', lw=2, label='K_resist')
    ax3b.set_ylabel('K_resist [N/m]'); ax3b.set_ylim(0, BOX[1, 1])

    ax3.legend(loc='upper left'); ax3b.legend(loc='upper right')
    plt.tight_layout(); plt.show()

print("=== Safe-BO finished ===")

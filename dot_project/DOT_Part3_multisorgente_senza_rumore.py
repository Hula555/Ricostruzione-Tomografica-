# -*- coding: utf-8 -*-
"""
DOT Project - Parte 3 (versione multi-sorgente, coppie ai primi vicini,
inclusione estesa a blocco di voxel):
SVD di W e ricostruzione TSVD, problema inverso, senza rumore, mezzo
semi-infinito.

Rispetto alla versione a 1 sorgente + 8 rivelatori, la geometria e' ora
letta da `geometria_base` (8 sorgenti, 9 rivelatori). Ogni MISURA e'
identificata da (coppia sorgente-rivelatore, gate): le righe di W sono
ordinate come  riga = p*N_gates + g,  con p indice della coppia in `pairs`.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp

# %% =============================================================
# STEP 1 - Parametri ottici, geometria sorgenti/rivelatori, griglia, gate
# ==================================================================
mu_a0 = 0.01
mu_s0 = 1.00
n_idx = 1.4
D = 1/(3*mu_s0)
v = 3e2/n_idx

def A_coefficient(n):
    Reff = -1.4399/n**2 + 0.7099/n + 0.6681 + 0.0636*n
    return (1 + Reff)/(1 - Reff)

A_coeff = A_coefficient(n_idx)
z0 = 1/mu_s0
zb = 2*A_coeff*D

# --- geometria sorgenti / rivelatori ---------------------------------------
geometria_base = {
    "unit": "mm",
    "points": [
        {"type": "detector", "x": 0,   "y": 0},
        {"type": "source",   "x": 0,   "y": 12},
        {"type": "source",   "x": -12, "y": 0},
        {"type": "source",   "x": 12,  "y": 0},
        {"type": "source",   "x": 0,   "y": -12},
        {"type": "detector", "x": -12, "y": 12},
        {"type": "detector", "x": 0,   "y": 24},
        {"type": "detector", "x": 12,  "y": 12},
        {"type": "detector", "x": 12,  "y": -12},
        {"type": "detector", "x": -12, "y": -12},
        {"type": "detector", "x": 0,   "y": -24},
        {"type": "detector", "x": 24,  "y": 0},
        {"type": "detector", "x": -24, "y": 0},
        {"type": "source",   "x": -12, "y": 24},
        {"type": "source",   "x": 12,  "y": 24},
        {"type": "source",   "x": 12,  "y": -24},
        {"type": "source",   "x": -12, "y": -24},
    ],
}

src_xy = np.array([[p["x"], p["y"]] for p in geometria_base["points"]
                   if p["type"] == "source"], dtype=float)
det_xy = np.array([[p["x"], p["y"]] for p in geometria_base["points"]
                   if p["type"] == "detector"], dtype=float)
N_src, N_det = src_xy.shape[0], det_xy.shape[0]

# sorgenti: iniettate a z0 (reale) e -z0-2zb (immagine)
r_s0 = np.column_stack((src_xy, np.full(N_src, z0)))
r_si = np.column_stack((src_xy, np.full(N_src, -z0 - 2*zb)))
# rivelatori: posizione reale a z=0 (superficie fisica) e immagine a -2zb
r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))
r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))

# --- coppie sorgente-rivelatore usate come misure ---------------------------
# modalita_coppie = "primi_vicini": per ciascuna sorgente (in sequenza) si
#     tengono solo i rivelatori alla distanza MINIMA da essa (primi vicini,
#     entro una tolleranza tol_vicini).
# modalita_coppie = "tutte": tutte le coppie s-d (versione precedente).
modalita_coppie = "primi_vicini"
tol_vicini = 0.5     # [mm] tolleranza per considerare "uguali" due distanze

pairs = []
print("Selezione coppie sorgente-rivelatore:", modalita_coppie)
for s_i in range(N_src):
    dist_sd = np.linalg.norm(det_xy - src_xy[s_i], axis=1)      # (N_det,)
    if modalita_coppie == "primi_vicini":
        d_sel = np.where(dist_sd <= dist_sd.min() + tol_vicini)[0]
    else:
        d_sel = np.arange(N_det)
    pairs += [(s_i, int(d)) for d in d_sel]
    print(f"  S{s_i+1} ({src_xy[s_i,0]:+.0f},{src_xy[s_i,1]:+.0f}) -> "
          + ", ".join(f"D{d+1}" for d in d_sel)
          + f"   (rho = {dist_sd[d_sel].min():.1f} mm)")
N_pairs = len(pairs)
rho_pairs = np.array([np.linalg.norm(src_xy[s] - det_xy[d]) for s, d in pairs])
pair_index = {sd: p for p, sd in enumerate(pairs)}   # (s,d) -> p

# --- griglia di voxel --------------------------------------------------------
step = 4.0
x_ext = 64.0
z_ext = 32.0
x_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
y_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
z_coords = np.arange(0 + step/2, z_ext, step)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
r_V = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
N_vox = r_V.shape[0]
V_vox = step**3
ext_xy = (x_coords[0]-step/2, x_coords[-1]+step/2,
          y_coords[0]-step/2, y_coords[-1]+step/2)

# --- tempo e time gating (12 gate fissi da 0.2 ns) ---------------------------
dt = 0.05
t = np.arange(0.1, 10.0, dt)
t_start = 1.0
gate_width = 0.20
N_gates = 12
gates = [(t_start + gate_width*i, t_start + gate_width*(i + 1)) for i in range(N_gates)]
gate_indices = [(np.searchsorted(t, g[0]), np.searchsorted(t, g[1])) for g in gates]
for g_i, (i0, i1) in enumerate(gate_indices):
    if i1 - i0 < 2:
        raise ValueError(f"Gate {g_i} ({gates[g_i]}) troppo stretto rispetto a dt={dt}: "
                         f"contiene solo {i1 - i0} campioni.")

N_meas = N_pairs * N_gates

print("="*60)
print(f"GEOMETRIA: {N_src} sorgenti, {N_det} rivelatori -> {N_pairs} coppie s-d")
rho_u, rho_c = np.unique(np.round(rho_pairs, 2), return_counts=True)
print("Distanze s-d [mm] (numero di coppie): " +
      ", ".join(f"{r:.1f} ({c})" for r, c in zip(rho_u, rho_c)))
print(f"Gate: {N_gates} da {gate_width} ns, range {gates[0][0]:.1f}-{gates[-1][1]:.1f} ns")
print(f"Misure: {N_pairs} x {N_gates} = {N_meas}  |  Incognite (voxel): {N_vox}")
print("="*60 + "\n")


def disegna_optodi(ax, c_src='red', c_det='cyan', s_src=90, s_det=35, alpha=1.0, label=True):
    """Sorgenti (stelle) e rivelatori (cerchi) su un asse x-y."""
    ax.scatter(src_xy[:, 0], src_xy[:, 1], marker='*', c=c_src, s=s_src, alpha=alpha,
               edgecolors='k', linewidths=0.4, label='Sorgenti' if label else None)
    ax.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c=c_det, s=s_det, alpha=alpha,
               edgecolors='k', linewidths=0.4, label='Rivelatori' if label else None)


# %% =============================================================
# STEP 2 - Vettore dei voxel A: inclusione a blocco di nx x ny x nz voxel
# ==================================================================
# L'inclusione e' definita dal suo CENTRO GEOMETRICO (xp, yp, zp) e dal
# numero di voxel per asse (n_blk). Il centro e' la media dei centri dei
# voxel del blocco:
#   - n pari  (es. 2) -> il centro cade su una FACCIA/SPIGOLO fra voxel
#                        (per 2x2x2: su un vertice condiviso da 8 voxel)
#   - n dispari       -> il centro cade al centro di un voxel
# Esempio: voxel con centri x in {14,18}, y in {10,14}, z in {18,22}
#   -> blocco [12,20] x [8,16] x [16,24] mm, centro (16, 12, 20) mm.
dmu_a = 0.01
n_blk = (2, 2, 2)                  # voxel per asse (x, y, z)
xp, yp, zp = 16.0, 12.0, 20.0      # centro geometrico dell'inclusione [mm]

L_blk = np.array(n_blk) * step     # lati fisici del blocco [mm]
# disuguaglianza STRETTA: un voxel appartiene al blocco se il suo centro
# dista meno di L/2 dal centro dell'inclusione (evita i voxel "di bordo")
mask = ((np.abs(X - xp) < L_blk[0]/2) &
        (np.abs(Y - yp) < L_blk[1]/2) &
        (np.abs(Z - zp) < L_blk[2]/2))
A_rep = mask * dmu_a
A = A_rep.flatten()

N_incl = int(mask.sum())
V_incl = N_incl * V_vox
if N_incl != np.prod(n_blk):
    raise ValueError(f"Il blocco contiene {N_incl} voxel invece di {np.prod(n_blk)}: "
                     f"il centro ({xp},{yp},{zp}) non e' allineato alla griglia "
                     f"(n pari -> centro su un bordo di voxel, n dispari -> su un centro).")
r_true = np.array([xp, yp, zp])
iz_p = np.argmin(np.abs(z_coords - zp))    # strato di voxel piu' vicino al centro
                                            # (per 2 voxel in z: il primo dei due)

print(f"Inclusione {n_blk[0]}x{n_blk[1]}x{n_blk[2]} voxel: centro ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm, "
      f"lati {L_blk} mm\n  -> {N_incl} voxel, V_incl = {V_incl:.1f} mm^3\n"
      f"  centri dei voxel: x={np.unique(X[mask])}, y={np.unique(Y[mask])}, z={np.unique(Z[mask])}\n")

# --- geometria: pianta (x-y) + due sezioni laterali (x-z, y-z) -------------
# Il blocco dell'inclusione e' disegnato con i suoi lati reali L_blk in tutte
# e tre le viste; gli optodi stanno sulla superficie z = 0.
x0, x1 = xp - L_blk[0]/2, xp + L_blk[0]/2
y0, y1 = yp - L_blk[1]/2, yp + L_blk[1]/2
z0b, z1b = zp - L_blk[2]/2, zp + L_blk[2]/2
z_lim = (z_coords[-1] + step/2, 0)            # asse z verso il basso (profondita')

fig, axs = plt.subplots(1, 3, figsize=(16, 5.5),
                        gridspec_kw={'width_ratios': [1, 1, 1]})
ax = axs[0]                                    # pianta
for (s_i, d_i) in pairs:
    ax.plot([src_xy[s_i, 0], det_xy[d_i, 0]], [src_xy[s_i, 1], det_xy[d_i, 1]],
            color='gray', lw=0.6, alpha=0.5, zorder=1)
ax.add_patch(plt.Rectangle((x0, y0), L_blk[0], L_blk[1], fc='orange', ec='k',
                           alpha=0.7, zorder=2, label=f'Inclusione {n_blk[0]}x{n_blk[1]}x{n_blk[2]}'))
disegna_optodi(ax)
ax.scatter(xp, yp, marker='+', c='k', s=80, zorder=5, label='Centro inclusione')
ax.set(xlim=ext_xy[:2], ylim=ext_xy[2:], xlabel='x [mm]', ylabel='y [mm]', aspect='equal')
ax.set_xticks(np.arange(ext_xy[0], ext_xy[1] + step, step), minor=True)
ax.set_yticks(np.arange(ext_xy[2], ext_xy[3] + step, step), minor=True)
ax.grid(which='minor', alpha=0.15); ax.grid(which='major', alpha=0.4)
ax.set_title(f"Pianta (z = 0): {N_src} S, {N_det} D, {N_pairs} coppie")
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

for ax, (lab, c_opt, c0, L0, cp) in zip(
        axs[1:], [('x', 0, x0, L_blk[0], xp), ('y', 1, y0, L_blk[1], yp)]):
    ax.add_patch(plt.Rectangle((c0, z0b), L0, L_blk[2], fc='orange', ec='k', alpha=0.7))
    ax.scatter(cp, zp, marker='+', c='k', s=80, zorder=5)
    ax.scatter(det_xy[:, c_opt], np.zeros(N_det), marker='o', c='cyan', s=60,
               edgecolors='k', linewidths=0.4, zorder=4, clip_on=False)
    ax.scatter(src_xy[:, c_opt], np.zeros(N_src), marker='*', c='red', s=70,
               edgecolors='k', linewidths=0.4, zorder=5, clip_on=False)
    ax.set(xlim=ext_xy[:2], ylim=z_lim, xlabel=f'{lab} [mm]', ylabel='z [mm]', aspect='equal')
    ax.set_xticks(np.arange(ext_xy[0], ext_xy[1] + step, step), minor=True)
    ax.set_yticks(np.arange(0, z_coords[-1] + step, step), minor=True)
    ax.grid(which='minor', alpha=0.15); ax.grid(which='major', alpha=0.4)
    ax.set_title(f"Sezione laterale {lab}-z (proiezione degli optodi)")
fig.suptitle(f"Geometria del sistema e inclusione: centro ({xp:.0f}, {yp:.0f}, {zp:.0f}) mm, "
             f"blocco [{x0:.0f},{x1:.0f}]x[{y0:.0f},{y1:.0f}]x[{z0b:.0f},{z1b:.0f}] mm")
fig.tight_layout(); plt.show()

# --- rappresentazione 3D del vettore A, sull'INTERO dominio -----------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
x_edges = np.linspace(ext_xy[0], ext_xy[1], x_coords.size+1)
y_edges = np.linspace(ext_xy[2], ext_xy[3], y_coords.size+1)
z_edges = np.linspace(0, z_coords[-1]+step/2, z_coords.size+1)
Xe, Ye, Ze = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
ax.voxels(Xe, Ye, Ze, mask, facecolors='orange', edgecolor='k', alpha=0.9)
ax.scatter(src_xy[:, 0], src_xy[:, 1], 0, marker='*', c='red', s=60, label='Sorgenti')
ax.scatter(det_xy[:, 0], det_xy[:, 1], 0, marker='o', c='cyan', s=25, label='Rivelatori')
ax.set(xlim=ext_xy[:2], ylim=ext_xy[2:], zlim=(0, z_coords[-1]+step/2),
       xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]')
ax.set_box_aspect((ext_xy[1]-ext_xy[0], ext_xy[3]-ext_xy[2], z_coords[-1]+step/2))
ax.invert_zaxis()
ax.legend(fontsize=8)
fig.suptitle(f"Vettore A nel dominio {x_ext:.0f}x{x_ext:.0f}x{z_ext:.0f} mm\n"
             f"{N_incl} voxel perturbati ({n_blk[0]}x{n_blk[1]}x{n_blk[2]}), "
             f"volume {V_incl:.0f} mm$^3$, step {step} mm")
plt.show()

# --- sezioni 2D di A: una per ciascuno strato z dell'inclusione --------------
iz_layers = np.unique(np.where(mask)[2])
fig, axs = plt.subplots(1, len(iz_layers), figsize=(5.5*len(iz_layers), 5), squeeze=False)
for ax, iz in zip(axs[0], iz_layers):
    im = ax.imshow(A_rep[:, :, iz].T, origin='lower', extent=ext_xy, vmin=0, vmax=dmu_a)
    disegna_optodi(ax, label=(iz == iz_layers[0]))
    ax.scatter(xp, yp, marker='+', c='w', s=80)
    ax.set(xlabel='x [mm]', ylabel='y [mm]')
    ax.set_title(f"Vettore A, strato z = {z_coords[iz]:.0f} mm "
                 f"([{z_coords[iz]-step/2:.0f},{z_coords[iz]+step/2:.0f}] mm)")
    ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=8)
fig.colorbar(im, ax=axs, label=r'$\delta\mu_a$ [$mm^{-1}$]', shrink=0.85)
plt.show()


# %% =============================================================
# STEP 3 - Matrice di sensitivita' W  (righe: coppia p, gate g)
# ==================================================================
def deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r, r1, n=n_idx):
    """Perturbazione di fluenza (Born, I ordine) in mezzo infinito."""
    D = 1/(3*mu_s0)
    v = 3e2/n
    return (-(v**2)/((4*pi*D*v)**(5/2)*t**(3/2))*exp(-mu_a0*v*t)
            * dmu_a*V*(1/r + 1/r1)*exp(-(r+r1)**2/(4*D*v*t)))

def fluence_inf(mu_a0, mu_s0, n, t, r):
    D = 1/(3*mu_s0); v = 3e2/n
    return (v/(4*pi*D*v*t)**(3/2))*exp(-(r**2/(4*D*v*t)) - mu_a0*v*t)

# distanze precalcolate: (N_src, N_vox) e (N_det, N_vox)
r_sv_all  = np.linalg.norm(r_V[None, :, :] - r_s0[:, None, :], axis=2)
r_siv_all = np.linalg.norm(r_V[None, :, :] - r_si[:, None, :], axis=2)
r_vd_all  = np.linalg.norm(r_V[None, :, :] - r_d0[:, None, :], axis=2)
r_vdi_all = np.linalg.norm(r_V[None, :, :] - r_di[:, None, :], axis=2)

W = np.zeros((N_meas, N_vox))
tt = t[None, :]

for p, (s, d) in enumerate(pairs):
    r_sv, r_siv = r_sv_all[s][:, None], r_siv_all[s][:, None]
    r_vd, r_vdi = r_vd_all[d][:, None], r_vdi_all[d][:, None]

    dphi_vt = (deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv,  r_vd)
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv,  r_vdi)
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv, r_vd)
             + deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv, r_vdi))

    r_d_s  = np.linalg.norm(r_d0[d] - r_s0[s])
    r_d_si = np.linalg.norm(r_d0[d] - r_si[s])
    phi0_t = (fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s)
              - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si))

    for g, (i0, i1) in enumerate(gate_indices):
        int_dphi = np.sum(dphi_vt[:, i0:i1], axis=1) * dt
        int_phi0 = np.sum(phi0_t[i0:i1]) * dt
        W[p*N_gates + g, :] = int_dphi / int_phi0

print(f"Matrice W costruita: shape = {W.shape}")

W_3d = W.reshape(N_pairs, N_gates, x_coords.size, y_coords.size, z_coords.size)
Wlog_vmin, Wlog_vmax = -4, 1

# --- sezioni di W per UNA coppia s-d, a diverse profondita' e gate ---
p_show = 0
s_sh, d_sh = pairs[p_show]
z_show = [10.0, 18.0, 30.0]
g_show = [0, 5, 11]
fig, axs = plt.subplots(3, 3, figsize=(10, 11))
for r_i, zval in enumerate(z_show):
    iz = np.argmin(np.abs(z_coords - zval))
    for c_i, g in enumerate(g_show):
        Wsec = np.log10(np.clip(np.abs(W_3d[p_show, g, :, :, iz]), 1e-8, None)).T
        im = axs[r_i, c_i].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                                  extent=ext_xy)
        axs[r_i, c_i].scatter(*src_xy[s_sh], marker='*', c='red', s=40)
        axs[r_i, c_i].scatter(*det_xy[d_sh], marker='o', c='lime', s=20)
        axs[r_i, c_i].set_title(f"z={z_coords[iz]:.0f} mm, gate {g} "
                                f"({gates[g][0]:.1f}-{gates[g][1]:.1f} ns)", fontsize=9, pad=8)
        if c_i == 0: axs[r_i, c_i].set_ylabel('y [mm]')
        if r_i == 2: axs[r_i, c_i].set_xlabel('x [mm]')
fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, right=0.86)
fig.suptitle(r'Sezioni di $\log_{10}|W|$' f' - coppia S{s_sh+1}-D{d_sh+1} '
             f'(rho = {rho_pairs[p_show]:.1f} mm)')
fig.colorbar(im, ax=axs, shrink=0.7, label=r'$\log_{10}|W|$')
plt.show()

# --- una sorgente fissata, tutti i rivelatori (z e gate fissati) ---
s_fix, z_fix, g_fix = 0, 18.0, 3
iz = np.argmin(np.abs(z_coords - z_fix))
n_col = 3
n_row = int(np.ceil(N_det / n_col))
fig, axs = plt.subplots(n_row, n_col, figsize=(4*n_col, 3.8*n_row))
axs = np.atleast_1d(axs).flatten()
for d in range(N_det):
    ax = axs[d]
    if (s_fix, d) not in pair_index:
        ax.set_axis_off(); continue
    p = pair_index[(s_fix, d)]
    Wsec = np.log10(np.clip(np.abs(W_3d[p, g_fix, :, :, iz]), 1e-8, None)).T
    im = ax.imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax, extent=ext_xy)
    ax.scatter(*src_xy[s_fix], marker='*', c='red', s=50)
    ax.scatter(*det_xy[d], marker='o', c='lime', s=25)
    ax.set_title(f'S{s_fix+1}-D{d+1} (rho={rho_pairs[p]:.1f} mm)', fontsize=9)
for ax in axs[N_det:]:
    ax.set_axis_off()
fig.suptitle(f"Sensitivita' " r"$\log_{10}|W|$" f" per la sorgente S{s_fix+1}\n"
             f"z = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()


# %% =============================================================
# STEP 4 - Vettore delle misure M = W . A (senza rumore)
# ==================================================================
M_ideal = (W @ A).reshape(N_pairs, N_gates)
M_ideal_flat = M_ideal.flatten()        # stesso ordine (p*N_gates+g) delle righe di W

print(f"\nVettore delle misure: {N_pairs} coppie x {N_gates} gate. "
      f"|M| max = {np.abs(M_ideal).max():.3e}")
p_top = np.argsort(-np.abs(M_ideal).max(axis=1))[:5]
print("Coppie con segnale piu' intenso:")
for p in p_top:
    s, d = pairs[p]
    print(f"  S{s+1}-D{d+1} (rho={rho_pairs[p]:.1f} mm): " +
          " ".join(f"{val:+.4f}" for val in M_ideal[p, :]))

pair_labels = [f'S{s+1}-D{d+1}' for s, d in pairs]
vmax_M = np.abs(M_ideal).max()
fig, ax = plt.subplots(figsize=(8, max(5, 0.16*N_pairs)))
im = ax.imshow(M_ideal, aspect='auto', cmap='RdBu_r', vmin=-vmax_M, vmax=vmax_M)
ax.set_yticks(range(N_pairs)); ax.set_yticklabels(pair_labels, fontsize=6)
ax.set_xticks(range(N_gates))
ax.set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90)
ax.set_xlabel('time gate [ns]'); ax.set_ylabel('coppia sorgente-rivelatore')
ax.set_title('Vettore delle misure M = W . A (ideale, mezzo semi-infinito)')
fig.colorbar(im, label='C (-)')
fig.tight_layout(); plt.show()


# %% =============================================================
# STEP 5 - SVD di W
# ==================================================================
U, s, Vt = np.linalg.svd(W, full_matrices=False)
k_max = s.size   # = min(N_meas, N_vox)

print("="*60)
print("SVD DI W")
print("="*60)
print(f"U: {U.shape}   s: {s.shape}   Vt: {Vt.shape}")
print(f"Valore singolare massimo  s_0   = {s[0]:.3e}")
print(f"Valore singolare minimo   s_{k_max-1:<4d}= {s[-1]:.3e}")
print(f"Numero di condizionamento s_0/s_min = {s[0]/s[-1]:.2e}")
eps = np.finfo(float).eps
k_num = int(np.sum(s > s[0]*eps*max(W.shape)))
print(f"Rango numerico (s_i > s_0*eps*max(m,n)): {k_num} su {k_max}")
print("="*60 + "\n")

plt.figure()
plt.semilogy(np.arange(1, k_max + 1), s, '.-', ms=3)
plt.axhline(s[0]*eps*max(W.shape), color='r', ls='--', lw=1, label='soglia precisione macchina')
plt.xlabel('ordine $i$'); plt.ylabel('$s_i$')
plt.title("Valori singolari di $W$"); plt.legend(); plt.grid(); plt.show()


# %% =============================================================
# STEP 6 - Alcuni modi (u_i, v_i)
# ==================================================================
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
fig, axs = plt.subplots(len(orders_to_show), 2, figsize=(9, 3.3*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    U_img = U[:, i].reshape(N_pairs, N_gates)
    vmax_u = np.abs(U_img).max()
    axs[row, 0].imshow(U_img, aspect='auto', cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u,
                       origin='lower')
    axs[row, 0].set_title(f"$u_{{{i}}}$ (spazio misure) - $s_{{{i}}}$={s[i]:.2e}", fontsize=10)
    axs[row, 0].set_xlabel('gate'); axs[row, 0].set_ylabel('coppia s-d')

    V_section = Vt[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)[:, :, iz_p].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                       extent=ext_xy)
    disegna_optodi(axs[row, 1], c_src='k', c_det='w', s_src=30, s_det=10, alpha=0.6, label=False)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_p]:.0f} mm)",
                          fontsize=10)
fig.tight_layout(); plt.show()


# %% =============================================================
# STEP 7 - TSVD con dati ideali, scansione su k
# ==================================================================
# Senza rumore, in aritmetica esatta, A_hat(k) = V_k V_k^T A e l'errore e'
# monotono non crescente. Con molti piu' modi (k_max grande), pero', i
# valori singolari piu' piccoli possono scendere vicino alla precisione
# macchina: dividere per essi amplifica gli errori di arrotondamento
# (~ eps * s_0 / s_i) e la curva puo' RISALIRE negli ultimi modi, anche
# senza rumore di misura. In quel caso k di errore minimo < k_max.

coeffs_all = (U.T @ M_ideal_flat) / s          # coefficienti di tutti i modi

def tsvd_solve(M_flat, k):
    coeffs = (U[:, :k].T @ M_flat) / s[:k]
    return Vt[:k, :].T @ coeffs

k_values = np.arange(1, k_max + 1)
rel_err_ideal = np.zeros(k_max)
A_hat_cum = np.zeros(N_vox)
for j in range(k_max):                          # somma cumulativa dei modi
    A_hat_cum += coeffs_all[j] * Vt[j, :]
    rel_err_ideal[j] = np.linalg.norm(A_hat_cum - A) / np.linalg.norm(A)

k_min_err = int(k_values[np.argmin(rel_err_ideal)])

plt.figure()
plt.semilogy(k_values, rel_err_ideal, '.-', ms=3, label=r'errore su $A$ (dati ideali)')
plt.axvline(k_min_err, color='gray', ls='--', lw=1, label=f'$k$ di errore minimo = {k_min_err}')
plt.axvline(k_num, color='r', ls=':', lw=1, label=f'rango numerico = {k_num}')
plt.xlabel('numero di valori singolari usati, $k$')
plt.ylabel(r'errore relativo $\|\hat A-A\|/\|A\|$')
plt.title('Troncamento SVD con dati ideali')
plt.legend(); plt.grid(); plt.show()

print(f"Dati ideali: errore minimo = {rel_err_ideal.min():.3e} a k={k_min_err} "
      f"(su k_max={k_max}); errore a k_max = {rel_err_ideal[-1]:.3e}")


# %% =============================================================
# STEP 8 - Confronto visivo: A vero vs ricostruzioni TSVD
# ==================================================================
# L'inclusione occupa piu' strati in z: si mostra una riga per ciascuno.
k_show = sorted(set([max(1, k_max//8), k_max//2, k_min_err, k_max]))
iz_layers = np.unique(np.where(mask)[2])           # indici z degli strati dell'inclusione
Nx, Ny, Nz = x_coords.size, y_coords.size, z_coords.size

fig, axs = plt.subplots(len(iz_layers), len(k_show)+1,
                        figsize=(4*(len(k_show)+1), 4*len(iz_layers)), squeeze=False)
A_hats = {k: tsvd_solve(M_ideal_flat, k) for k in k_show}
for r_i, iz in enumerate(iz_layers):
    ax = axs[r_i, 0]
    ax.imshow(A_rep[:, :, iz].T, origin='lower', cmap='viridis', vmin=0, vmax=dmu_a,
              extent=ext_xy)
    disegna_optodi(ax, label=False)
    ax.set_title(f'$A$ vero, z = {z_coords[iz]:.0f} mm'); ax.set_ylabel('y [mm]')
    for ax, k in zip(axs[r_i, 1:], k_show):
        A_hat = A_hats[k]
        vmax_hat = np.abs(A_hat).max()
        im = ax.imshow(A_hat.reshape(Nx, Ny, Nz)[:, :, iz].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax_hat, vmax=vmax_hat, extent=ext_xy)
        disegna_optodi(ax, c_src='k', c_det='w', s_src=40, s_det=15, alpha=0.6, label=False)
        ax.add_patch(plt.Rectangle((xp - L_blk[0]/2, yp - L_blk[1]/2), L_blk[0], L_blk[1],
                                   fc='none', ec='lime', lw=1.2))
        ax.set_title(f'$\\hat A$, $k$={k}, z = {z_coords[iz]:.0f} mm\n'
                     f'err={rel_err_ideal[k-1]:.2e}', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
for ax in axs[-1, :]:
    ax.set_xlabel('x [mm]')
fig.suptitle(f"Ricostruzione TSVD (dati ideali) - inclusione {n_blk[0]}x{n_blk[1]}x{n_blk[2]} voxel, "
             f"centro ({xp:.0f}, {yp:.0f}, {zp:.0f}) mm (riquadro verde)")
fig.tight_layout(); plt.show()


# %% =============================================================
# STEP 8b - Localizzazione dell'inclusione estesa: tre metodi
# ==================================================================
# (1) voxel di picco: puo' coincidere al massimo con UNO degli 8 voxel,
#     quindi l'errore minimo sul centro e' mezzo voxel per asse
#     (sqrt(3)*step/2 ~ 3.5 mm per step = 4 mm).
# (2) baricentro pesato del cluster (A_hat >= soglia*picco, valori > 0):
#     puo' cadere FRA i voxel, quindi puo' stimare un centro su un vertice;
#     risente pero' dell'allargamento e del bias verso la superficie.
# (3) scansione di blocchi sui DATI: per ogni blocco candidato B di
#     dimensione n_blk si fitta dmu_a uniforme su B,
#         a_B = w_B^T M / ||w_B||^2 ,   w_B = sum_{l in B} W[:, l],
#     e si calcola il residuo ||M - a_B w_B||^2. Il blocco a residuo
#     minimo da' il centro (media dei centri dei suoi voxel) e dmu_a.
#     Non usa la TSVD (e' un fit a 1 parametro con un'ipotesi di forma).
soglia_cluster = 0.5

def scansione_blocchi(M_flat, nb):
    """Fit di dmu_a uniforme su tutti i blocchi nb=(bx,by,bz) della griglia.
    Restituisce centro del blocco migliore, a_B, residuo relativo, indici."""
    bx, by, bz = nb
    W4 = W.reshape(N_meas, Nx, Ny, Nz)
    sx, sy, sz = Nx-bx+1, Ny-by+1, Nz-bz+1
    Wb = np.zeros((N_meas, sx, sy, sz))
    for dx in range(bx):
        for dy in range(by):
            for dz in range(bz):
                Wb += W4[:, dx:dx+sx, dy:dy+sy, dz:dz+sz]
    Wb = Wb.reshape(N_meas, -1)                    # colonne = blocchi candidati
    proj = Wb.T @ M_flat
    nrm2 = np.sum(Wb**2, axis=0)
    res2 = M_flat @ M_flat - proj**2 / nrm2        # residuo del fit ai minimi quadrati
    b = np.argmin(res2)
    ib = np.unravel_index(b, (sx, sy, sz))
    centro = np.array([x_coords[ib[0]:ib[0]+bx].mean(),
                       y_coords[ib[1]:ib[1]+by].mean(),
                       z_coords[ib[2]:ib[2]+bz].mean()])
    blocco = np.zeros((Nx, Ny, Nz), bool)
    blocco[ib[0]:ib[0]+bx, ib[1]:ib[1]+by, ib[2]:ib[2]+bz] = True
    rel_res = np.sqrt(max(res2[b], 0) / (M_flat @ M_flat))
    return centro, proj[b]/nrm2[b], rel_res, blocco.flatten()

print("="*60)
print(f"LOCALIZZAZIONE - centro vero ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
print("="*60)
for k in k_show:
    A_hat = A_hats[k]
    j = np.argmax(A_hat)
    r_peak = r_V[j]
    cluster = (A_hat >= soglia_cluster * A_hat[j])
    r_bar = (A_hat[cluster, None] * r_V[cluster]).sum(0) / A_hat[cluster].sum()
    neg_pos = np.clip(-A_hat, 0, None).sum() / np.clip(A_hat, 0, None).sum()
    print(f"k={k:4d}  (neg/pos = {neg_pos:.2f}, cluster = {cluster.sum()} voxel)")
    print(f"   (1) voxel di picco   = ({r_peak[0]:6.2f}, {r_peak[1]:6.2f}, {r_peak[2]:6.2f}) mm"
          f"   |errore| = {np.linalg.norm(r_peak - r_true):.2f} mm")
    print(f"   (2) baricentro clust.= ({r_bar[0]:6.2f}, {r_bar[1]:6.2f}, {r_bar[2]:6.2f}) mm"
          f"   |errore| = {np.linalg.norm(r_bar - r_true):.2f} mm")

print("\n   (3) scansione di blocchi sui dati (indipendente da k), varie dimensioni:")
for nb in [(1, 1, 1), (2, 2, 2), (3, 3, 3)]:
    c_b, a_b, rr, _ = scansione_blocchi(M_ideal_flat, nb)
    print(f"       blocco {nb}: centro = ({c_b[0]:6.2f}, {c_b[1]:6.2f}, {c_b[2]:6.2f}) mm, "
          f"dmu_a = {a_b:.3e}, residuo relativo = {rr:.2e}, "
          f"|errore| = {np.linalg.norm(c_b - r_true):.2f} mm")
centro_blk, a_blk, _, blocco_blk = scansione_blocchi(M_ideal_flat, n_blk)
print("="*60)


# %% =============================================================
# STEP 9 - Stima quantitativa di dmu_a per l'inclusione estesa
# ==================================================================
# (b) generalizzata: per un'inclusione su un insieme di voxel B,
#     A_hat_k[j] = dmu_a * sum_{l in B} R_jl ,   R = V_k V_k^T,
#     quindi dmu_a = A_hat_k[j] / sum_{l in B} R_jl. Serve conoscere B:
#     si usa il blocco trovato dalla scansione (3).
print("="*60)
print(f"STIMA DI dmu_a  (valore vero = {dmu_a:.3e} mm^-1)")
print("="*60)
idx_B = np.where(blocco_blk)[0]
for k in k_show:
    A_hat = A_hats[k]
    j = np.argmax(A_hat)
    est_peak = A_hat[j]                                           # (a)
    R_jB = Vt[:k, j] @ Vt[:k, idx_B]                             # R_jl per l in B
    est_res = est_peak / R_jB.sum()                               # (b) generalizzata
    cluster = A_hat >= soglia_cluster * est_peak
    w_c = W[:, cluster].sum(axis=1)
    est_fit = (w_c @ M_ideal_flat) / (w_c @ w_c)                  # (c) fit sul cluster
    print(f"k={k:4d} (picco in {r_V[j]}, cluster = {cluster.sum()} voxel)")
    for nome, est in [("(a) picco grezzo           ", est_peak),
                      ("(b) picco / sum_B R_jl     ", est_res),
                      ("(c) fit sul cluster TSVD   ", est_fit)]:
        print(f"    {nome} = {est:.3e} mm^-1   (stima/vero = {est/dmu_a:.3f})")
print(f"(d) fit sul blocco della scansione  = {a_blk:.3e} mm^-1   "
      f"(stima/vero = {a_blk/dmu_a:.3f})")
print("="*60)

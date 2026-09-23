# -*- coding: utf-8 -*-
"""
DOT Project - Parte 3 (versione multi-sorgente, coppie ai primi vicini,
inclusione estesa a blocco di voxel):
SVD di W e ricostruzione TSVD, problema inverso, CON RUMORE POISSONIANO,
mezzo semi-infinito.

Parte da DOT_Part3_multisorgente_senza_rumore.py (Step 1-4 identici) e
aggiunge lo shot noise sui conteggi TCSPC di ciascuna coppia s-d (Step 4b),
con le correzioni gia' introdotte per la geometria a 1 sorgente:
  - budget fotonico N_tot_TPSF riferito all'INTERA TPSF di ogni coppia;
  - stimatore del contrasto quasi non distorto  N_p/(N_0+1) - 1;
  - sigma_M stimata dai conteggi misurati;
  - inversione PESATA (whitening W_w = Sigma^{-1/2} W, M_w = Sigma^{-1/2} M);
  - k della TSVD scelto col principio di discrepanza (senza conoscere A);
  - scansione di blocchi e stima di dmu_a ai minimi quadrati pesati,
    con barra d'errore statistica.

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
# NB (versione con rumore): con i soli primi vicini (24 coppie, rho = 12 mm)
# e 12 gate su 1.0-3.4 ns, sopra il rumore restano ~25 modi e la risoluzione
# a 20 mm di profondita' e' ~20 mm: l'inclusione 2x2x2 si localizza ma non
# si "vede" come immagine. Con tutte le coppie (72, rho = 12-34 mm) e 20
# gate su 0.4-4.4 ns i modi utili salgono a ~125 e, con la ricostruzione
# vincolata dello Step 11, l'inclusione viene ricostruita. Per tornare alla
# configurazione originale: "primi_vicini", t_start = 1.0, N_gates = 12.
modalita_coppie = "tutte"
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

# --- tempo e time gating (gate fissi da 0.2 ns) ---------------------------
dt = 0.05
t = np.arange(0.1, 10.0, dt)
t_start = 0.4        # (configurazione originale: 1.0)
gate_width = 0.20
N_gates = 20         # (configurazione originale: 12) -> range 0.4-4.4 ns
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
# STEP 4b - Rumore poissoniano (shot noise) sui CONTEGGI di ogni coppia
# ==================================================================
# Per ogni coppia p e gate g il dato grezzo TCSPC e' un conteggio
#     N_pg ~ Poisson(lambda_pg),   E[N] = Var[N] = lambda
#     lambda0_pg = N_tot_TPSF * int_g Phi0_p dt / int Phi0_p dt   (baseline)
#     lambdap_pg = lambda0_pg * (1 + M_pg)                          (Born)
# Baseline e misura perturbata sono due acquisizioni indipendenti, quindi
#     sigma_M^2 ~ (1+M)(2+M)/lambda0 ~ 2/lambda0     (metodo delta)
# NB: la frazione di TPSF che cade nei gate dipende da rho e dal range dei
# gate (con rho = 12 mm e gate 1.0-3.4 ns era solo ~2.5%).
# NB (inverse crime): dati e inversione usano lo stesso modello lineare.
N_tot_TPSF = 1e8          # conteggi sull'intera TPSF, per coppia s-d
rng = np.random.default_rng(42)

Phi0_int = np.zeros((N_pairs, N_gates))
Phi0_tot = np.zeros(N_pairs)
for p, (s, d) in enumerate(pairs):
    phi0_t = (fluence_inf(mu_a0, mu_s0, n_idx, t, np.linalg.norm(r_d0[d] - r_s0[s]))
              - fluence_inf(mu_a0, mu_s0, n_idx, t, np.linalg.norm(r_d0[d] - r_si[s])))
    Phi0_tot[p] = np.sum(phi0_t) * dt
    for g, (i0, i1) in enumerate(gate_indices):
        Phi0_int[p, g] = np.sum(phi0_t[i0:i1]) * dt
frac_gates = Phi0_int / Phi0_tot[:, None]          # frazione della TPSF in ogni gate

def simula_misura(N_tot, rng):
    """Una realizzazione di baseline + misura perturbata. Restituisce
    M stimato (N_pairs, N_gates) e sigma_M stimata dai conteggi."""
    lam0 = N_tot * frac_gates
    # float: con interi a 64 bit (N0+1)**3 va in overflow per N0 > ~2e6
    n0 = rng.poisson(lam0).astype(float)
    npert = rng.poisson(np.clip(lam0 * (1 + M_ideal), 0, None)).astype(float)
    M_est = npert / (n0 + 1) - 1           # quasi non distorto, mai divisione per 0
    sig = np.sqrt((npert + 1) / (n0 + 1)**2 + (npert + 1)**2 / (n0 + 1)**3)
    return M_est, sig

M_noisy, sigma_M = simula_misura(N_tot_TPSF, rng)
M_noisy_flat, sig_flat = M_noisy.flatten(), sigma_M.flatten()

N0_ideal = N_tot_TPSF * frac_gates
sigma_th = np.sqrt((1 + M_ideal) * (2 + M_ideal) / N0_ideal)
SNR_pg = np.abs(M_ideal) / sigma_th
SNR_tot = np.sqrt(np.sum(SNR_pg**2))

print("="*60)
print(f"RUMORE: N_tot = {N_tot_TPSF:.0e} conteggi/TPSF per coppia "
      f"(nei gate: {100*frac_gates.sum(1).mean():.2f}% della TPSF)")
print(f"Conteggi di baseline per gate: min {N0_ideal.min():.0f} / max {N0_ideal.max():.0f}")
print(f"SNR per misura: max {SNR_pg.max():.2f}, misure con SNR>1: {(SNR_pg > 1).sum()} su {N_meas}")
print(f"SNR complessivo sqrt(sum SNR^2) = {SNR_tot:.1f}")
print("="*60 + "\n")

fig, axs = plt.subplots(1, 2, figsize=(13, max(5, 0.16*N_pairs)), sharey=True)
vmax_M = np.abs(M_ideal).max()
for ax, Mplot, title in zip(axs, [M_ideal, M_noisy], ['M ideale', 'M con shot noise']):
    im = ax.imshow(Mplot, aspect='auto', cmap='RdBu_r', vmin=-vmax_M, vmax=vmax_M)
    ax.set_xticks(range(N_gates))
    ax.set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90, fontsize=7)
    ax.set_xlabel('time gate [ns]'); ax.set_title(title)
axs[0].set_yticks(range(N_pairs)); axs[0].set_yticklabels(pair_labels, fontsize=6)
axs[0].set_ylabel('coppia sorgente-rivelatore')
fig.colorbar(im, ax=axs, shrink=0.8, label='C (-)')
fig.suptitle(f'M ideale vs M con shot noise (N_tot = {N_tot_TPSF:.0e} conteggi/TPSF/coppia)')
plt.show()

p_best = int(np.argmax(np.abs(M_ideal).max(axis=1)))
g_idx = np.arange(N_gates)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(g_idx, M_ideal[p_best], 'o-', label='M ideale')
axs[0].errorbar(g_idx, M_noisy[p_best], yerr=sigma_M[p_best], fmt='s', capsize=3,
                label=r'M con rumore $\pm\hat\sigma_M$')
axs[0].set(xlabel='gate', ylabel='C (-)', title=f'Coppia {pair_labels[p_best]} (segnale massimo)')
axs[0].legend(); axs[0].grid()
im = axs[1].imshow(SNR_pg, aspect='auto', cmap='viridis')
axs[1].set(xlabel='gate', ylabel='coppia', title=r'SNR per misura $|M|/\sigma_M$')
axs[1].set_yticks(range(N_pairs)); axs[1].set_yticklabels(pair_labels, fontsize=6)
fig.colorbar(im, ax=axs[1])
plt.tight_layout(); plt.show()


# %% =============================================================
# STEP 5 - SVD di W (non pesata) e di W_w = Sigma^{-1/2} W (pesata)
# ==================================================================
# Il rumore e' eteroschedastico (sigma_M varia di ~35 volte fra i gate):
# la soluzione a massima verosimiglianza minimizza ||Sigma^{-1/2}(W A - M)||^2.
U, s, Vt = np.linalg.svd(W, full_matrices=False)
W_w = W / sig_flat[:, None]
M_noisy_w = M_noisy_flat / sig_flat
M_ideal_w = M_ideal_flat / sig_flat
Uw, sw, Vtw = np.linalg.svd(W_w, full_matrices=False)
k_max = s.size   # = min(N_meas, N_vox)

print("="*60)
print("SVD DI W e di W_w")
print("="*60)
print(f"U: {U.shape}   s: {s.shape}   Vt: {Vt.shape}")
print(f"W  : s_0 = {s[0]:.3e}, s_min = {s[-1]:.3e}, cond = {s[0]/s[-1]:.2e}")
print(f"W_w: s_0 = {sw[0]:.3e}, s_min = {sw[-1]:.3e}, cond = {sw[0]/sw[-1]:.2e}")
eps = np.finfo(float).eps
k_num = int(np.sum(s > s[0]*eps*max(W.shape)))
print(f"Rango numerico di W: {k_num} su {k_max}")

# --- grafico di Picard: coefficienti del dato pesato lungo i modi ---
# c_i = u_i^T M_w = s_i (v_i^T A) + e_i,  con e_i ~ N(0,1): il modo i porta
# informazione utile solo se |s_i v_i^T A| > ~1 (livello del rumore).
c_noisy = Uw.T @ M_noisy_w
c_signal = Uw.T @ M_ideal_w
n_modi_utili = int(np.sum(np.abs(c_signal) > 2))
print(f"Modi con segnale sopra il rumore (|u_i^T M_w,ideale| > 2): {n_modi_utili}")
print("="*60 + "\n")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
axs[0].semilogy(np.arange(1, k_max + 1), s/s[0], '.-', ms=3, label='W')
axs[0].semilogy(np.arange(1, k_max + 1), sw/sw[0], '.-', ms=3, label=r'$W_w$ (pesata)')
axs[0].set(xlabel='ordine $i$', ylabel='$s_i/s_0$', title='Valori singolari normalizzati')
axs[0].legend(); axs[0].grid()
axs[1].semilogy(np.arange(1, k_max + 1), np.abs(c_noisy), '.', ms=4, label=r'$|u_i^T M_w|$ (dati rumorosi)')
axs[1].semilogy(np.arange(1, k_max + 1), np.abs(c_signal), '-', lw=1, label=r'$|u_i^T M_w|$ (solo segnale)')
axs[1].axhline(1, color='r', ls='--', label='livello del rumore')
axs[1].set(xlabel='ordine $i$', ylabel='coefficiente', title='Grafico di Picard (dati pesati)')
axs[1].legend(fontsize=8); axs[1].grid()
plt.tight_layout(); plt.show()


# %% =============================================================
# STEP 6 - Alcuni modi (u_i, v_i) della matrice pesata W_w
# ==================================================================
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
fig, axs = plt.subplots(len(orders_to_show), 2, figsize=(9, 3.3*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    U_img = Uw[:, i].reshape(N_pairs, N_gates)
    vmax_u = np.abs(U_img).max()
    axs[row, 0].imshow(U_img, aspect='auto', cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u,
                       origin='lower')
    axs[row, 0].set_title(f"$u_{{{i}}}$ (spazio misure) - $s_{{{i}}}$={sw[i]:.2e}", fontsize=10)
    axs[row, 0].set_xlabel('gate'); axs[row, 0].set_ylabel('coppia s-d')

    V_section = Vtw[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)[:, :, iz_p].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                       extent=ext_xy)
    disegna_optodi(axs[row, 1], c_src='k', c_det='w', s_src=30, s_det=10, alpha=0.6, label=False)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_p]:.0f} mm)",
                          fontsize=10)
fig.suptitle(r'Modi singolari di $W_w$ (pesata)', y=1.0)
fig.tight_layout(); plt.show()


# %% =============================================================
# STEP 7 - TSVD: dati ideali (non pesata) vs dati rumorosi (pesata)
# ==================================================================
# Dati ideali: come nella versione senza rumore (errore monotono, a meno
# degli errori di arrotondamento sugli ultimi modi).
# Dati rumorosi: dividere per s_i piccoli amplifica il rumore, quindi il
# troncamento e' una regolarizzazione. k viene scelto col PRINCIPIO DI
# DISCREPANZA (Morozov): il piu' piccolo k con chi^2 = ||W_w A_hat - M_w||^2
# <= N_meas (valore atteso del chi^2 del solo rumore). Non richiede A.

def tsvd_generica(Uu, ss, VVt, m, k):
    return VVt[:k, :].T @ ((Uu[:, :k].T @ m) / ss[:k])

def tsvd_solve(M_flat, k):                       # non pesata (dati ideali)
    return tsvd_generica(U, s, Vt, M_flat, k)

def tsvd_solve_w(M_flat, k):                     # pesata (dati rumorosi)
    return tsvd_generica(Uw, sw, Vtw, M_flat / sig_flat, k)

k_values = np.arange(1, k_max + 1)
normA = np.linalg.norm(A)

# ideale: somma cumulativa dei modi
coeffs_all = (U.T @ M_ideal_flat) / s
rel_err_ideal = np.zeros(k_max)
A_hat_cum = np.zeros(N_vox)
for j in range(k_max):
    A_hat_cum += coeffs_all[j] * Vt[j, :]
    rel_err_ideal[j] = np.linalg.norm(A_hat_cum - A) / normA
k_min_err = int(k_values[np.argmin(rel_err_ideal)])

# rumoroso, pesata: somme cumulative (errore su A, errore da rumore, chi^2)
coef_n = c_noisy / sw
coef_i = c_signal / sw
rel_err_noisy = np.zeros(k_max)
rel_err_noise_only = np.zeros(k_max)
A_n_cum = np.zeros(N_vox); A_i_cum = np.zeros(N_vox)
for j in range(k_max):
    A_n_cum += coef_n[j] * Vtw[j, :]
    A_i_cum += coef_i[j] * Vtw[j, :]
    rel_err_noisy[j] = np.linalg.norm(A_n_cum - A) / normA
    rel_err_noise_only[j] = np.linalg.norm(A_n_cum - A_i_cum) / normA
chi2 = np.sum(M_noisy_w**2) - np.cumsum(c_noisy**2)   # residuo pesato dopo k modi

ok = np.where(chi2 <= N_meas)[0]
k_disc = int(k_values[ok[0]]) if ok.size else k_max
k_best = int(k_values[np.argmin(rel_err_noisy)])   # "oracolo": richiede A

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
axs[0].semilogy(k_values, rel_err_ideal, '.-', ms=3, label='dati ideali (non pesata)')
axs[0].semilogy(k_values, rel_err_noisy, '.-', ms=3, label='dati rumorosi (pesata)')
axs[0].axvline(k_disc, color='k', ls='--', lw=1, label=f'$k$ discrepanza = {k_disc}')
axs[0].axvline(k_best, color='gray', ls=':', lw=1, label=f'$k$ oracolo = {k_best}')
axs[0].set(xlabel='$k$', ylabel=r'$\|\hat A-A\|/\|A\|$', title='Errore su A')
axs[0].legend(fontsize=8); axs[0].grid()
axs[1].semilogy(k_values, rel_err_noise_only, '.-', ms=3)
axs[1].set(xlabel='$k$', ylabel=r'$\|\hat A_{noisy}(k)-\hat A_{ideal}(k)\|/\|A\|$',
           title='Errore indotto dal solo rumore (pesata)')
axs[1].grid()
axs[2].semilogy(k_values, np.clip(chi2, 1e-3, None), '.-', ms=3)
axs[2].axhline(N_meas, color='r', ls='--', label=r'$\chi^2 = N_{meas}$')
axs[2].axvline(k_disc, color='k', ls='--', lw=1)
axs[2].set(xlabel='$k$', ylabel=r'$\chi^2 = \|W_w\hat A - M_w\|^2$', title='Principio di discrepanza')
axs[2].legend(); axs[2].grid()
plt.tight_layout(); plt.show()

print(f"Dati ideali:    errore minimo = {rel_err_ideal.min():.3e} a k={k_min_err}")
print(f"Dati rumorosi:  errore minimo = {rel_err_noisy.min():.3e} a k={k_best} (oracolo)")
print(f"Dati rumorosi:  k discrepanza = {k_disc}, errore = {rel_err_noisy[k_disc-1]:.3e}")
print(f"Dati rumorosi, rango pieno k={k_max}: errore = {rel_err_noisy[-1]:.3e}\n")


# %% =============================================================
# STEP 8 - Confronto visivo: A vero vs ricostruzioni (dati rumorosi, pesata)
# ==================================================================
k_show = sorted(set([max(1, k_disc//2), k_disc, min(k_max, 2*k_disc)]))
iz_layers = np.unique(np.where(mask)[2])
Nx, Ny, Nz = x_coords.size, y_coords.size, z_coords.size

A_hats = {k: tsvd_solve_w(M_noisy_flat, k) for k in k_show}
fig, axs = plt.subplots(len(iz_layers), len(k_show)+2,
                        figsize=(4*(len(k_show)+2), 4*len(iz_layers)), squeeze=False)
A_hat_ideal = tsvd_solve(M_ideal_flat, k_min_err)
for r_i, iz in enumerate(iz_layers):
    ax = axs[r_i, 0]
    ax.imshow(A_rep[:, :, iz].T, origin='lower', cmap='viridis', vmin=0, vmax=dmu_a,
              extent=ext_xy)
    disegna_optodi(ax, label=False)
    ax.set_title(f'$A$ vero, z = {z_coords[iz]:.0f} mm'); ax.set_ylabel('y [mm]')
    panels = [(A_hat_ideal, f'ideale, $k$={k_min_err}')] + \
             [(A_hats[k], f'rumore, $k$={k}' + (' (discr.)' if k == k_disc else '')) for k in k_show]
    for ax, (A_hat, lab) in zip(axs[r_i, 1:], panels):
        vmax_hat = np.abs(A_hat).max()
        im = ax.imshow(A_hat.reshape(Nx, Ny, Nz)[:, :, iz].T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax_hat, vmax=vmax_hat, extent=ext_xy)
        disegna_optodi(ax, c_src='k', c_det='w', s_src=40, s_det=15, alpha=0.6, label=False)
        ax.add_patch(plt.Rectangle((xp - L_blk[0]/2, yp - L_blk[1]/2), L_blk[0], L_blk[1],
                                   fc='none', ec='lime', lw=1.2))
        ax.set_title(f'$\\hat A$ {lab}\nz = {z_coords[iz]:.0f} mm', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
for ax in axs[-1, :]:
    ax.set_xlabel('x [mm]')
fig.suptitle(f"Ricostruzione TSVD - dati ideali vs con shot noise (pesata), "
             f"N_tot = {N_tot_TPSF:.0e}; centro vero ({xp:.0f}, {yp:.0f}, {zp:.0f}) mm (riquadro verde)")
fig.tight_layout(); plt.show()


# %% =============================================================
# STEP 8b - Localizzazione dell'inclusione estesa (dati rumorosi)
# ==================================================================
# (1) voxel di picco, (2) baricentro del cluster: come nella versione
#     senza rumore, applicati alla TSVD pesata.
# (3) scansione di blocchi ai minimi quadrati PESATI:
#         a_B = (w_B^T Sigma^-1 M) / (w_B^T Sigma^-1 w_B),
#         Delta chi^2_B = (w_B^T Sigma^-1 M)^2 / (w_B^T Sigma^-1 w_B)
#     Il blocco con Delta chi^2 massimo (= residuo minimo) e' la stima.
#     Delta chi^2 e' anche un test di rivelazione: per "nessuna inclusione"
#     segue ~chi^2 a 1 grado di liberta' (massimizzato sui blocchi), quindi
#     valori di poche decine indicano un'inclusione significativa.
#     Barra d'errore statistica su a_B:  sigma_a = 1/sqrt(w_B^T Sigma^-1 w_B).
soglia_cluster = 0.5

def scansione_blocchi(M_flat, nb, sig=None):
    """Fit di dmu_a uniforme su tutti i blocchi nb=(bx,by,bz) della griglia,
    pesato con 1/sig^2 (sig=None -> non pesato). Restituisce centro del
    blocco migliore, a_B, sigma_a, Delta chi^2, maschera del blocco."""
    wgt = np.ones(N_meas) if sig is None else 1.0 / sig
    bx, by, bz = nb
    W4 = W.reshape(N_meas, Nx, Ny, Nz)
    sx, sy, sz = Nx-bx+1, Ny-by+1, Nz-bz+1
    Wb = np.zeros((N_meas, sx, sy, sz))
    for dx in range(bx):
        for dy in range(by):
            for dz in range(bz):
                Wb += W4[:, dx:dx+sx, dy:dy+sy, dz:dz+sz]
    Wb = Wb.reshape(N_meas, -1) * wgt[:, None]     # colonne = blocchi candidati
    Mw = M_flat * wgt
    proj = Wb.T @ Mw
    nrm2 = np.sum(Wb**2, axis=0)
    dchi2 = proj**2 / nrm2                         # riduzione del chi^2 col fit
    b = np.argmax(dchi2)
    ib = np.unravel_index(b, (sx, sy, sz))
    centro = np.array([x_coords[ib[0]:ib[0]+bx].mean(),
                       y_coords[ib[1]:ib[1]+by].mean(),
                       z_coords[ib[2]:ib[2]+bz].mean()])
    blocco = np.zeros((Nx, Ny, Nz), bool)
    blocco[ib[0]:ib[0]+bx, ib[1]:ib[1]+by, ib[2]:ib[2]+bz] = True
    return centro, proj[b]/nrm2[b], 1/np.sqrt(nrm2[b]), dchi2[b], blocco.flatten()

print("="*60)
print(f"LOCALIZZAZIONE (dati rumorosi) - centro vero ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
print("="*60)
for k in k_show:
    A_hat = A_hats[k]
    j = np.argmax(A_hat)
    r_peak = r_V[j]
    cluster = (A_hat >= soglia_cluster * A_hat[j])
    r_bar = (A_hat[cluster, None] * r_V[cluster]).sum(0) / A_hat[cluster].sum()
    neg_pos = np.clip(-A_hat, 0, None).sum() / np.clip(A_hat, 0, None).sum()
    print(f"k={k:4d}{' (discr.)' if k == k_disc else ''}  "
          f"(neg/pos = {neg_pos:.2f}, cluster = {cluster.sum()} voxel)")
    print(f"   (1) voxel di picco   = ({r_peak[0]:6.2f}, {r_peak[1]:6.2f}, {r_peak[2]:6.2f}) mm"
          f"   |errore| = {np.linalg.norm(r_peak - r_true):.2f} mm")
    print(f"   (2) baricentro clust.= ({r_bar[0]:6.2f}, {r_bar[1]:6.2f}, {r_bar[2]:6.2f}) mm"
          f"   |errore| = {np.linalg.norm(r_bar - r_true):.2f} mm")

print("\n   (3) scansione di blocchi PESATA sui dati rumorosi, varie dimensioni:")
for nb in [(1, 1, 1), (2, 2, 2), (3, 3, 3)]:
    c_b, a_b, sa_b, dchi, _ = scansione_blocchi(M_noisy_flat, nb, sig_flat)
    print(f"       blocco {nb}: centro = ({c_b[0]:6.2f}, {c_b[1]:6.2f}, {c_b[2]:6.2f}) mm, "
          f"dmu_a = {a_b:.3e} +- {sa_b:.1e}, Delta chi2 = {dchi:.0f}, "
          f"|errore| = {np.linalg.norm(c_b - r_true):.2f} mm")
centro_blk, a_blk, sa_blk, dchi_blk, blocco_blk = scansione_blocchi(M_noisy_flat, n_blk, sig_flat)
print("="*60)


# %% =============================================================
# STEP 9 - Stima quantitativa di dmu_a (dati rumorosi)
# ==================================================================
# (a) picco grezzo, (b) picco / sum_B R_jl con R = V_k V_k^T della
# matrice PESATA, (c) fit pesato sul cluster TSVD, (d) fit pesato sul
# blocco della scansione (con barra d'errore statistica).
print("="*60)
print(f"STIMA DI dmu_a  (valore vero = {dmu_a:.3e} mm^-1)")
print("="*60)
idx_B = np.where(blocco_blk)[0]
for k in k_show:
    A_hat = A_hats[k]
    j = np.argmax(A_hat)
    est_peak = A_hat[j]                                           # (a)
    R_jB = Vtw[:k, j] @ Vtw[:k, idx_B]                           # R_jl per l in B
    est_res = est_peak / R_jB.sum()                               # (b)
    cluster = A_hat >= soglia_cluster * est_peak
    w_c = W_w[:, cluster].sum(axis=1)
    est_fit = (w_c @ M_noisy_w) / (w_c @ w_c)                     # (c) fit pesato
    print(f"k={k:4d} (picco in {r_V[j]}, cluster = {cluster.sum()} voxel)")
    for nome, est in [("(a) picco grezzo           ", est_peak),
                      ("(b) picco / sum_B R_jl     ", est_res),
                      ("(c) fit pesato sul cluster ", est_fit)]:
        print(f"    {nome} = {est:.3e} mm^-1   (stima/vero = {est/dmu_a:.3f})")
print(f"(d) fit pesato sul blocco della scansione = {a_blk:.3e} +- {sa_blk:.1e} mm^-1   "
      f"(stima/vero = {a_blk/dmu_a:.3f})")
print("="*60 + "\n")


# %% =============================================================
# STEP 10 - Robustezza al rumore: piu' realizzazioni e piu' budget
# ==================================================================
# Un'unica realizzazione puo' essere fortunata o sfortunata: si ripete la
# simulazione per diversi N_tot e si riportano media e dispersione di
#  - errore sul centro dalla scansione di blocchi pesata,
#  - errore sul centro dal baricentro della TSVD pesata a k di discrepanza,
#  - rapporto stima/vero di dmu_a dalla scansione.
N_tot_scan = [1e6, 1e7, 1e8, 1e9]
N_real = 5
rng_mc = np.random.default_rng(123)
stats = {}
for N_tot in N_tot_scan:
    e_blk, e_bar, ratio, kd_list = [], [], [], []
    snr = np.sqrt(np.sum(M_ideal**2 * N_tot * frac_gates / ((1 + M_ideal) * (2 + M_ideal))))
    for _ in range(N_real):
        Mn, sg = simula_misura(N_tot, rng_mc)
        m, sgf = Mn.flatten(), sg.flatten()
        c_b, a_b, _, _, _ = scansione_blocchi(m, n_blk, sgf)
        e_blk.append(np.linalg.norm(c_b - r_true)); ratio.append(a_b / dmu_a)
        U2, s2, V2 = np.linalg.svd(W / sgf[:, None], full_matrices=False)
        mw = m / sgf; c2 = U2.T @ mw
        res = np.sum(mw**2) - np.cumsum(c2**2)
        kd = int(np.argmax(res <= N_meas)) + 1 if np.any(res <= N_meas) else k_max
        A_hat = V2[:kd].T @ (c2[:kd] / s2[:kd])
        j = np.argmax(A_hat); cl = A_hat >= soglia_cluster * A_hat[j]
        r_bar = (A_hat[cl, None] * r_V[cl]).sum(0) / A_hat[cl].sum()
        e_bar.append(np.linalg.norm(r_bar - r_true)); kd_list.append(kd)
    stats[N_tot] = (snr, np.array(e_blk), np.array(e_bar), np.array(ratio), np.array(kd_list))

print("="*60)
print(f"ROBUSTEZZA ({N_real} realizzazioni per budget)")
print("="*60)
print(f"{'N_tot':>7s} {'SNR':>6s} {'err blocchi [mm]':>18s} {'err baric. TSVD [mm]':>22s} "
      f"{'dmu stima/vero':>16s} {'k discr.':>9s}")
for N_tot, (snr, eb, ebar, rt, kd) in stats.items():
    print(f"{N_tot:7.0e} {snr:6.1f} {eb.mean():8.1f} +- {eb.std():4.1f}   "
          f"{ebar.mean():10.1f} +- {ebar.std():4.1f}   {rt.mean():8.2f} +- {rt.std():4.2f}   "
          f"{kd.mean():6.0f}")
print("="*60)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
xs = np.array(N_tot_scan)
for key, lab, mk in [(1, 'scansione di blocchi (pesata)', 'o'), (2, 'baricentro TSVD pesata (k discr.)', 's')]:
    mu_ = np.array([stats[n][key].mean() for n in N_tot_scan])
    sd_ = np.array([stats[n][key].std() for n in N_tot_scan])
    axs[0].errorbar(xs, mu_, yerr=sd_, fmt=mk+'-', capsize=3, label=lab)
axs[0].set(xscale='log', xlabel='N_tot (conteggi/TPSF/coppia)', ylabel='errore sul centro [mm]',
           title='Localizzazione vs budget fotonico')
axs[0].legend(); axs[0].grid()
mu_r = np.array([stats[n][3].mean() for n in N_tot_scan])
sd_r = np.array([stats[n][3].std() for n in N_tot_scan])
axs[1].errorbar(xs, mu_r, yerr=sd_r, fmt='o-', capsize=3)
axs[1].axhline(1, color='k', ls='--')
axs[1].set(xscale='log', xlabel='N_tot (conteggi/TPSF/coppia)', ylabel=r'$\delta\mu_a$ stima/vero',
           title=r'Stima di $\delta\mu_a$ (scansione di blocchi)')
axs[1].grid()
plt.tight_layout(); plt.show()


# %% =============================================================
# STEP 11 - Risoluzione della TSVD e ricostruzione VINCOLATA
# ==================================================================
# (a) Perche' l'immagine TSVD e' sfocata: con k modi la TSVD restituisce
#         A_hat = R A,   R = V_k V_k^T   (matrice di risoluzione)
#     La colonna di R relativa a un voxel e' la sua "point spread function"
#     (PSF): se la PSF e' piu' larga dell'inclusione, l'immagine e' una
#     macchia larga quanto la PSF, con ampiezza ridotta (volume parziale).
#     Il rumore limita k (Step 7), quindi limita la risoluzione.
j0 = np.where(mask.flatten())[0][0]              # un voxel dell'inclusione
R_col = Vtw[:k_disc, :].T @ Vtw[:k_disc, j0]
half = R_col >= 0.5 * R_col.max()
psf_ext = r_V[half].max(0) - r_V[half].min(0) + step
print("="*60)
print(f"RISOLUZIONE TSVD (k = {k_disc}): larghezza a meta' altezza della PSF")
print(f"   nel voxel {r_V[j0]}: x = {psf_ext[0]:.0f} mm, y = {psf_ext[1]:.0f} mm, "
      f"z = {psf_ext[2]:.0f} mm   (inclusione: {L_blk[0]:.0f} mm di lato)")
print("="*60 + "\n")

# (b) Ricostruzione vincolata: si aggiunge l'informazione a priori che
#     l'inclusione e' un ASSORBITORE (dmu_a >= 0) e LOCALIZZATO (sparso):
#         min_x  1/2 ||W_w D^-1 x - M_w||^2 + lam * sum(x),   x >= 0,
#         A_hat = D^-1 x,   D = diag(||colonne di W_w||)
#     D compensa il calo di sensibilita' con la profondita' (senza D la
#     penalita' L1 favorirebbe i voxel superficiali). I voxel quasi
#     invisibili (norma della colonna < soglia_oss * massimo, tipicamente
#     angoli profondi lontani dagli optodi) sono esclusi dalle incognite:
#     con D^-1 enorme, un loro valore minuscolo di x diventerebbe un
#     artefatto enorme in A_hat.
#     Si risolve con FISTA (gradiente proiettato accelerato). lam e' scelto
#     col principio di discrepanza: si parte da lam grande e lo si riduce
#     (warm start) finche' chi^2 <= N_meas + 2*sqrt(2*N_meas), cioe' entro
#     2 deviazioni standard dal chi^2 atteso del solo rumore.
soglia_oss = 1e-2

def fista_nn_l1(B, m, lam, x0=None, n_iter=2000):
    """min 1/2||B x - m||^2 + lam*sum(x), x >= 0 (FISTA)."""
    Lip = np.linalg.norm(B, 2)**2
    x = np.zeros(B.shape[1]) if x0 is None else x0.copy()
    y, tk = x.copy(), 1.0
    for _ in range(n_iter):
        x_new = np.maximum(y - (B.T @ (B @ y - m) + lam) / Lip, 0)
        t_new = (1 + np.sqrt(1 + 4*tk*tk)) / 2
        y = x_new + (tk - 1) / t_new * (x_new - x)
        x, tk = x_new, t_new
    return x

def ricostruzione_vincolata(M_flat, sig,
                            fattori=(0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4)):
    Bw = W / sig[:, None]
    col = np.linalg.norm(Bw, axis=0)
    oss = col >= soglia_oss * col.max()            # voxel osservabili
    B = Bw[:, oss] / col[oss]
    m = M_flat / sig
    lam_max = np.abs(B.T @ m).max()
    chi2_target = N_meas + 2*np.sqrt(2*N_meas)
    x = None
    for f in fattori:
        x = fista_nn_l1(B, m, f * lam_max, x0=x)
        chi2_x = np.sum((B @ x - m)**2)
        if chi2_x <= chi2_target:
            break
    A_out = np.zeros(N_vox)
    A_out[oss] = x / col[oss]
    return A_out, f, chi2_x, int(oss.sum())

A_con, f_lam, chi2_con, n_oss = ricostruzione_vincolata(M_noisy_flat, sig_flat)

def metriche(A_hat):
    j = np.argmax(A_hat)
    cl = A_hat >= soglia_cluster * A_hat[j]
    r_bar = (A_hat[cl, None] * r_V[cl]).sum(0) / A_hat[cl].sum()
    vero = A > 0
    dice = 2 * (cl & vero).sum() / (cl.sum() + vero.sum())
    return r_bar, dice, cl.sum(), A_hat[vero].sum() / A[vero].sum()

print("="*60)
print("CONFRONTO RICOSTRUZIONI (dati rumorosi)")
print("="*60)
print("Dice = sovrapposizione fra voxel >= 50% del picco e inclusione vera (1 = perfetta)")
for nome, A_hat in [(f"TSVD pesata, k={k_disc}", A_hats[k_disc]),
                    (f"vincolata (>=0, L1), lam={f_lam:g}*lam_max", A_con)]:
    r_bar, dice, ncl, frac_massa = metriche(A_hat)
    print(f"{nome:38s}: baricentro ({r_bar[0]:5.1f}, {r_bar[1]:5.1f}, {r_bar[2]:5.1f}) mm, "
          f"|errore| = {np.linalg.norm(r_bar - r_true):4.1f} mm, Dice = {dice:.2f}, "
          f"voxel = {ncl}, massa recuperata nei voxel veri = {100*frac_massa:.0f}%")
print(f"   ricostruzione vincolata: {n_oss} voxel osservabili su {N_vox}, "
      f"chi^2 = {chi2_con:.0f} (atteso {N_meas} +- {np.sqrt(2*N_meas):.0f})")
print(f"   dmu_a medio nei voxel dell'inclusione = {A_con[mask.flatten()].mean():.2e} "
      f"(vero {dmu_a:.2e})")
print("="*60)

# --- immagini: strati z dell'inclusione e sezioni verticali ---
fig, axs = plt.subplots(len(iz_layers), 3, figsize=(13, 4.2*len(iz_layers)), squeeze=False)
for r_i, iz in enumerate(iz_layers):
    for c_i, (img, tit, cmap, vmin, vmax) in enumerate([
            (A_rep, '$A$ vero', 'viridis', 0, dmu_a),
            (A_hats[k_disc].reshape(Nx, Ny, Nz), f'TSVD pesata, $k$={k_disc}', 'RdBu_r', None, None),
            (A_con.reshape(Nx, Ny, Nz), 'vincolata ($\\geq 0$, L1)', 'viridis', 0, None)]):
        ax = axs[r_i, c_i]
        sec = img[:, :, iz].T
        if vmin is None:
            vm = np.abs(img).max(); vmin, vmax = -vm, vm
        im = ax.imshow(sec, origin='lower', cmap=cmap, vmin=vmin,
                       vmax=vmax if vmax is not None else img.max(), extent=ext_xy)
        disegna_optodi(ax, c_src='k', c_det='w', s_src=30, s_det=12, alpha=0.6, label=False)
        ax.add_patch(plt.Rectangle((xp - L_blk[0]/2, yp - L_blk[1]/2), L_blk[0], L_blk[1],
                                   fc='none', ec='lime', lw=1.2))
        ax.set_title(f'{tit}, z = {z_coords[iz]:.0f} mm', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Ricostruzione con shot noise: TSVD vs ricostruzione vincolata (riquadro verde = vero)')
fig.tight_layout(); plt.show()

iy_c = np.argmin(np.abs(y_coords - (yp - step/2)))    # sezione x-z attraverso l'inclusione
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, (img, tit, cmap) in zip(axs, [(A_rep, '$A$ vero', 'viridis'),
                                      (A_hats[k_disc].reshape(Nx, Ny, Nz), f'TSVD pesata, $k$={k_disc}', 'RdBu_r'),
                                      (A_con.reshape(Nx, Ny, Nz), 'vincolata', 'viridis')]):
    sec = img[:, iy_c, :].T
    vm = np.abs(img).max()
    im = ax.imshow(sec, origin='upper', cmap=cmap, vmin=-vm if cmap == 'RdBu_r' else 0, vmax=vm,
                   extent=(ext_xy[0], ext_xy[1], z_coords[-1] + step/2, 0), aspect='equal')
    ax.add_patch(plt.Rectangle((xp - L_blk[0]/2, zp - L_blk[2]/2), L_blk[0], L_blk[2],
                               fc='none', ec='lime', lw=1.2))
    ax.set(xlabel='x [mm]', ylabel='z [mm]', title=f'{tit}, sezione y = {y_coords[iy_c]:.0f} mm')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); plt.show()

# -*- coding: utf-8 -*-
"""
DOT Project - Parte 4: configurazione multi-sorgente multiplexata
(switch ottico) su griglia di rivelatori fissa, con rumore poissoniano,
mezzo semi-infinito.

Generalizza il caso a sorgente singola (Parte 2) al caso realistico in
cui PIU' sorgenti vengono accoppiate al campione diffusivo una alla
volta tramite uno switch ottico: si accende la sorgente S1, tutti i
rivelatori misurano la propria DTOF (fluenza, poi contrasto nel caso
perturbato); si spegne S1 e si accende S2, i rivelatori misurano di
nuovo; e cosi' via per tutte le N_src sorgenti. Il numero di misure
indipendenti cresce quindi di un fattore N_src rispetto al caso a
sorgente singola, a parita' di numero di rivelatori e di gate, con
evidente beneficio sul condizionamento del problema inverso (Parte 3).

Geometria (fornita dall'utente tramite il tool di posizionamento):
  - 4 sorgenti, poste su un anello di raggio 12 mm nelle direzioni
    cardinali: (0,12), (-12,0), (12,0), (0,-12) mm
  - 13 rivelatori, disposizione irregolare centrata sull'origine, dal
    rivelatore centrale (0,0) fino ad un anello esterno a raggio
    24-27 mm circa

La fisica (fluenza in mezzo infinito, metodo delle immagini per il
mezzo semi-infinito, contrasto di Born al I ordine) e' IDENTICA alla
Parte 2: le funzioni deltafluence_inf/fluence_inf non cambiano, cambia
solo la geometria su cui vengono valutate (ora un doppio ciclo
sorgente-rivelatore anziche' un solo ciclo sui rivelatori) e di
conseguenza la forma di W, M e di tutti i plot associati.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp

# %% =============================================================
# STEP 1 - Proprieta' ottiche, griglia spazio-temporale, geometria
#          sorgenti (switch) e rivelatori
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
z0 = 1/mu_s0            # profondita' virtuale della sorgente reale (1 mfp)
zb = 2*A_coeff*D         # distanza del piano a fluenza nulla estrapolato

# --- sorgenti: multiplexate via switch ottico, una accesa alla volta ---
sources_xy = np.array([
    [0.0,  12.0],
    [-12.0, 0.0],
    [12.0,  0.0],
    [0.0, -12.0],
])
N_src = sources_xy.shape[0]

# posizione reale "virtuale" a profondita' z0, e posizione immagine
# (metodo delle immagini per l'estrapolated boundary condition)
r_s_all  = np.column_stack((sources_xy, np.full(N_src, z0)))
r_si_all = np.column_stack((sources_xy, np.full(N_src, -z0 - 2*zb)))

# --- rivelatori: griglia fissa, disposizione irregolare attorno all'origine ---
detectors_xy = np.array([
    [0.0,    0.0],
    [-12.0, 12.0],
    [0.0,   24.0],
    [12.0,  12.0],
    [12.0, -12.0],
    [-12.0,-12.0],
    [0.0,  -24.0],
    [24.0,   0.0],
    [-24.0,  0.0],
    [-12.0, 24.0],
    [12.0,  24.0],
    [12.0, -24.0],
    [-12.0,-24.0],
])
N_det = detectors_xy.shape[0]

# rivelatori: posizione reale sulla superficie fisica z=0, e immagine a -2zb
r_d0_all = np.column_stack((detectors_xy, np.full(N_det, 0.0)))
r_di_all = np.column_stack((detectors_xy, np.full(N_det, -2*zb)))

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

N_meas = N_src*N_det*N_gates
print("="*55)
print(f"RICHIAMO: N_src = {N_src} | N_det = {N_det} | N_gates = {N_gates} | N_vox = {N_vox}")
print(f"Misurazioni indipendenti: {N_src}x{N_det}x{N_gates} = {N_meas}  |  Incognite (voxel): {N_vox}")
print(f"Rispetto al caso a sorgente singola (1x8x12=96), fattore di crescita "
      f"delle misure = {N_meas/96:.1f}x")
print(f"Gate fissi da {gate_width} ns, range {gates[0][0]:.1f}-{gates[-1][1]:.1f} ns")
print("="*55 + "\n")

# --- plot della geometria: sorgenti (switch) + rivelatori fissi ---
plt.figure(figsize=(5.5, 5.5))
plt.scatter(sources_xy[:, 0], sources_xy[:, 1], marker='*', c='red', s=180,
            label='Sorgenti (switch)', zorder=3, edgecolor='k', linewidth=0.5)
plt.scatter(detectors_xy[:, 0], detectors_xy[:, 1], marker='o', c='cyan', s=80,
            label='Rivelatori', zorder=3, edgecolor='k', linewidth=0.5)
for i, (xx, yy) in enumerate(sources_xy):
    plt.annotate(f'S{i+1}', (xx, yy), textcoords='offset points', xytext=(7, 7), fontsize=9)
for i, (xx, yy) in enumerate(detectors_xy):
    plt.annotate(f'D{i+1}', (xx, yy), textcoords='offset points', xytext=(7, -10),
                 fontsize=7, color='dimgray')
plt.gca().set_aspect('equal')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f'Geometria sorgenti-rivelatori\n'
          f'{N_src} sorgenti multiplexate via switch, {N_det} rivelatori fissi')
plt.legend(loc='upper right', fontsize=8)
plt.grid(alpha=0.3); plt.show()


# %% =============================================================
# STEP 2 - Vettore dei voxel A: perturbazione cubica di volume fisico esplicito
# ==================================================================
dmu_a = 0.01
V_phys = V_vox

xp, yp, zp = 14.0, 10.0, 18.0

L_incl = V_phys**(1/3)
mask = ((np.abs(X - xp) <= L_incl/2) &
        (np.abs(Y - yp) <= L_incl/2) &
        (np.abs(Z - zp) <= L_incl/2))

A_rep = mask * dmu_a
A = A_rep.flatten()

N_incl = int(mask.sum())
V_incl = N_incl * V_vox

ix, iy, iz_p = [np.argmin(np.abs(c - v)) for c, v in
                 zip((x_coords, y_coords, z_coords), (xp, yp, zp))]

print(f"Perturbazione cubica: centro (xp,yp,zp) = ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm, "
      f"lato fisico L = {L_incl:.2f} mm, V_phys = {V_phys:.1f} mm^3\n"
      f"  -> rappresentata da {N_incl} voxel, V_incl (effettivo su griglia) = {V_incl:.1f} mm^3")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
x_edges = np.linspace(x_coords[0]-step/2, x_coords[-1]+step/2, x_coords.size+1)
y_edges = np.linspace(y_coords[0]-step/2, y_coords[-1]+step/2, y_coords.size+1)
z_edges = np.linspace(0, z_coords[-1]+step/2, z_coords.size+1)
Xe, Ye, Ze = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')
ax.voxels(Xe, Ye, Ze, A_rep, edgecolor='k', alpha=0.8)
ax.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]')
ax.invert_zaxis()
fig.suptitle(f"Vettore A: rappresentazione 3D nello spazio voxel\n"
             f"Step {step} mm, 1 voxel perturbato, volume {V_incl} mm$^3$")
plt.show()

plt.figure()
im = plt.imshow(A_rep[:, :, iz_p].T, origin='lower',
                 extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                         y_coords[0]-step/2, y_coords[-1]+step/2),
                 vmin=0, vmax=dmu_a)
plt.scatter(sources_xy[:, 0], sources_xy[:, 1], marker='*', c='red', s=140, label='Sorgenti')
plt.scatter(detectors_xy[:, 0], detectors_xy[:, 1], marker='o', c='cyan', s=40, label='Rivelatori')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f"Sezione del vettore A nello spazio voxel\n"
          f"z = {zp:.0f} mm, step {step} mm")
plt.legend(fontsize=8)
plt.colorbar(im, label=r'$\delta\mu_a$ [$mm^{-1}$]')
plt.grid(); plt.show()


# %% =============================================================
# STEP 3 - Costruzione della matrice di sensitivita' (Jacobiano) W
#          (ora con un ciclo aggiuntivo sulle sorgenti)
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

W = np.zeros((N_meas, N_vox))
tt = t[None, :]

# ordine delle righe: sorgente-major, poi rivelatore, poi gate -
# esattamente la sequenza di acquisizione fisica (switch su S, tutti i
# rivelatori/gate per quella sorgente, poi switch sulla sorgente
# successiva)
for s in range(N_src):
    r_sv  = np.linalg.norm(r_V - r_s_all[s], axis=1)    # sorgente s -> ogni voxel
    r_siv = np.linalg.norm(r_V - r_si_all[s], axis=1)   # immagine di s -> ogni voxel

    for d in range(N_det):
        r_vd  = np.linalg.norm(r_V - r_d0_all[d], axis=1)   # voxel -> rivelatore reale
        r_vdi = np.linalg.norm(r_V - r_di_all[d], axis=1)   # voxel -> rivelatore immagine

        dphi_vt = (deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vd[:, None])
                 - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vdi[:, None])
                 - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vd[:, None])
                 + deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vdi[:, None]))

        r_d_s  = np.linalg.norm(r_d0_all[d] - r_s_all[s])
        r_d_si = np.linalg.norm(r_d0_all[d] - r_si_all[s])
        phi0_t = fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s) - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si)

        for g, (i0, i1) in enumerate(gate_indices):
            int_dphi = np.sum(dphi_vt[:, i0:i1], axis=1) * dt
            int_phi0 = np.sum(phi0_t[i0:i1]) * dt
            row = (s*N_det + d)*N_gates + g
            W[row, :] = int_dphi / int_phi0

print(f"Matrice W costruita: shape = {W.shape}  "
      f"({N_src} sorgenti x {N_det} rivelatori x {N_gates} gate)")

W_5d = W.reshape(N_src, N_det, N_gates, x_coords.size, y_coords.size, z_coords.size)
Wlog_vmin, Wlog_vmax = -4, 1

# --- (a) sezioni 2D di W per una singola coppia sorgente-rivelatore,
#     a diverse profondita' e gate (come nella Parte 2, ora per S1-D1) ---
s_show, d_show = 0, 0
z_show = [10.0, 18.0, 30.0]
g_show = [0, 5, 11]
fig, axs = plt.subplots(3, 3, figsize=(10, 11))
for r_i, zval in enumerate(z_show):
    iz = np.argmin(np.abs(z_coords - zval))
    for c_i, g in enumerate(g_show):
        Wsec = np.log10(np.clip(np.abs(W_5d[s_show, d_show, g, :, :, iz]), 1e-8, None)).T
        im = axs[r_i, c_i].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
        axs[r_i, c_i].scatter(*sources_xy[s_show], marker='*', c='red', s=30)
        axs[r_i, c_i].scatter(*detectors_xy[d_show], marker='o', c='lime', s=20)
        axs[r_i, c_i].set_title(f"z={z_coords[iz]:.0f} mm, gate {g} "
                                 f"({gates[g][0]:.1f}-{gates[g][1]:.1f} ns)", fontsize=9, pad=8)
        if c_i == 0: axs[r_i, c_i].set_ylabel('y [mm]')
        if r_i == 2: axs[r_i, c_i].set_xlabel('x [mm]')
fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, right=0.86)
fig.suptitle(r'Sezioni di $\log_{10}|W|$ nello spazio voxel - coppia S1-D1'
             '\n(scala colore condivisa fra tutti i pannelli, 12 gate da 0.2 ns)')
fig.colorbar(im, ax=axs, shrink=0.7, label=r'$\log_{10}|W|$')
plt.show()

# --- (b) confronto fra tutti i 13 rivelatori, sorgente fissa ---
z_fix = 18.0
g_fix = 3
iz = np.argmin(np.abs(z_coords - z_fix))
fig, axs = plt.subplots(4, 4, figsize=(16, 12))
axs = axs.flatten()
for d in range(N_det):
    Wsec = np.log10(np.clip(np.abs(W_5d[s_show, d, g_fix, :, :, iz]), 1e-8, None)).T
    im = axs[d].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[d].scatter(*sources_xy[s_show], marker='*', c='red', s=40)
    axs[d].scatter(*detectors_xy[d], marker='o', c='lime', s=25)
    axs[d].set_title(f'D{d+1}', fontsize=9)
for j in range(N_det, len(axs)):
    axs[j].axis('off')
fig.suptitle(r"Sensitivita' $\log_{10}|W|$ per ciascun rivelatore, sorgente S1"
             f"\nz = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=list(axs[:N_det]), shrink=0.7)
plt.show()

# --- (c) confronto fra le 4 sorgenti, rivelatore fisso ---
fig, axs = plt.subplots(1, N_src, figsize=(4*N_src, 4.3))
for s in range(N_src):
    Wsec = np.log10(np.clip(np.abs(W_5d[s, d_show, g_fix, :, :, iz]), 1e-8, None)).T
    im = axs[s].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[s].scatter(*sources_xy[s], marker='*', c='red', s=40)
    axs[s].scatter(*detectors_xy[d_show], marker='o', c='lime', s=25)
    axs[s].set_title(f'S{s+1}', fontsize=10)
    axs[s].set_xlabel('x [mm]')
axs[0].set_ylabel('y [mm]')
fig.suptitle(r"Sensitivita' $\log_{10}|W|$ per ciascuna sorgente, rivelatore D1"
             f"\nz = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()


# %% =============================================================
# STEP 4 - Vettore delle misure M = W . A (problema diretto, senza rumore)
# ==================================================================
M_ideal = (W @ A).reshape(N_src, N_det, N_gates)

print("VETTORE DELLE MISURE M = W . A (ideale, senza rumore)")
for s in range(N_src):
    print(f"-- Sorgente S{s+1} --")
    for d in range(N_det):
        print(f"  D{d+1}: " + " ".join(f"{val:+.4f}" for val in M_ideal[s, d, :]))

vmax_M = np.abs(M_ideal).max()
fig, axs = plt.subplots(1, N_src, figsize=(4.2*N_src, 5.5), sharey=True)
for s in range(N_src):
    im = axs[s].imshow(M_ideal[s], aspect='auto', cmap='RdBu_r', vmin=-vmax_M, vmax=vmax_M)
    axs[s].set_xticks(range(N_gates))
    axs[s].set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90, fontsize=6)
    axs[s].set_xlabel('time gate [ns]')
    axs[s].set_title(f'Sorgente S{s+1}')
axs[0].set_yticks(range(N_det))
axs[0].set_yticklabels([f'D{d+1}' for d in range(N_det)])
axs[0].set_ylabel('rivelatore')
fig.colorbar(im, ax=axs, shrink=0.8, label='C (-)')
fig.suptitle('Vettore delle misure M = W . A per sorgente (ideale, mezzo semi-infinito)\n'
             '12 gate fissi da 0.2 ns, range 1.0-3.4 ns')
plt.show()


# %% =============================================================
# STEP 5 - Aggiunta di rumore poissoniano (shot noise) al vettore delle misure M
# ==================================================================
# Stessa logica della versione a sorgente singola: M = dPhi/Phi0 non e'
# la grandezza fisicamente rumorosa, lo sono i conteggi di fotoni per
# gate (statistica di Poisson). Si rumorano quindi i conteggi
# ideali di baseline e perturbati, indipendentemente per OGNI
# combinazione sorgente-rivelatore (ogni accensione della sorgente via
# switch e' un'acquisizione a se', con la propria statistica di shot
# noise, indipendente dalle altre sorgenti e dagli altri rivelatori).

np.random.seed(42)

# --- 1) conteggi ideali di riferimento (mezzo omogeneo, senza inclusione) ---
Phi0_int = np.zeros((N_src, N_det, N_gates))
for s in range(N_src):
    for d in range(N_det):
        r_d_s  = np.linalg.norm(r_d0_all[d] - r_s_all[s])
        r_d_si = np.linalg.norm(r_d0_all[d] - r_si_all[s])
        phi0_t = (fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s)
                  - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si))
        for g, (i0, i1) in enumerate(gate_indices):
            Phi0_int[s, d, g] = np.sum(phi0_t[i0:i1]) * dt

# --- 2) budget fotonico totale per ciascuna coppia sorgente-rivelatore ---
N_tot = 1e6
N0_ideal = N_tot * Phi0_int / Phi0_int.sum(axis=2, keepdims=True)

# --- 3) conteggi ideali della misura perturbata (con inclusione) ---
Npert_ideal = N0_ideal * (1 + M_ideal)

# --- 4) realizzazioni poissoniane indipendenti delle due acquisizioni ---
N0_noisy = np.random.poisson(N0_ideal)
Npert_noisy = np.random.poisson(np.clip(Npert_ideal, 0, None))

# --- 5) dato normalizzato rumoroso ---
with np.errstate(divide='ignore', invalid='ignore'):
    M_noisy = np.where(N0_noisy > 0,
                        (Npert_noisy - N0_noisy) / N0_noisy,
                        np.nan)

print(f"Budget fotonico: N_tot = {N_tot:.0e} conteggi/coppia sorgente-rivelatore")
print(f"Conteggi minimi/massimi di baseline per gate: "
      f"{N0_ideal.min():.1f} / {N0_ideal.max():.1f}")

# --- Plot 1: confronto ideale vs rumoroso, sorgente S1 (come nella Parte 2) ---
vmax = np.nanmax(np.abs(np.concatenate([M_ideal[s_show].ravel(), M_noisy[s_show].ravel()])))
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, Mplot, title in zip(
        axs, [M_ideal[s_show], M_noisy[s_show]],
        [f'M ideale, sorgente S{s_show+1} (senza rumore)',
         f'M con rumore poissoniano, sorgente S{s_show+1} (shot noise)']):
    im = ax.imshow(Mplot, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_gates))
    ax.set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90, fontsize=7)
    ax.set_xlabel('time gate [ns]')
    ax.set_title(title)
axs[0].set_yticks(range(N_det))
axs[0].set_yticklabels([f'D{d+1}' for d in range(N_det)])
axs[0].set_ylabel('rivelatore')
fig.colorbar(im, ax=axs, shrink=0.8, label='C (-)')
fig.suptitle(f'Confronto M ideale vs M con shot noise, sorgente S{s_show+1} '
             f'(N_tot = {N_tot:.0e} conteggi/coppia)')
plt.show()

# --- Plot 2: profilo di una singola coppia sorgente-rivelatore, gate per gate ---
plt.figure(figsize=(7, 4))
plt.plot(range(N_gates), M_ideal[s_show, d_show], 'o-', label='M ideale')
plt.plot(range(N_gates), M_noisy[s_show, d_show], 's--', label='M con rumore')
plt.xlabel('indice gate temporale')
plt.ylabel('C (-)')
plt.title(f'Coppia S{s_show+1}-D{d_show+1}: confronto ideale vs rumoroso')
plt.legend()
plt.grid()
plt.show()


# %% =============================================================
# STEP 6 - Decomposizione ai valori singolari di W
# ==================================================================
U, s_sing, Vt = np.linalg.svd(W, full_matrices=False)
k_max = s_sing.size   # = min(N_meas, N_vox)

print("="*55)
print("SVD DI W")
print("="*55)
print(f"U: {U.shape}   s: {s_sing.shape}   Vt: {Vt.shape}")
print(f"Valore singolare massimo  s_0        = {s_sing[0]:.3e}")
print(f"Valore singolare minimo   s_{k_max-1:<3d} = {s_sing[-1]:.3e}")
print(f"Rapporto s_0/s_min (numero di condizionamento di W) = {s_sing[0]/s_sing[-1]:.2e}")
print("="*55 + "\n")

plt.figure()
plt.semilogy(np.arange(1, k_max + 1), s_sing, 'o-', ms=4)
plt.xlabel('ordine $i$'); plt.ylabel('$s_i$')
plt.title("Valori singolari di $W$ (scala logaritmica)")
plt.grid(); plt.show()


# %% =============================================================
# STEP 7 - Alcuni modi (coppie u_i, v_i): dai piu' "forti" ai piu' "deboli"
# ==================================================================
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
iz_show = np.argmin(np.abs(z_coords - zp))

fig, axs = plt.subplots(len(orders_to_show), 2, figsize=(8, 3.1*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    # spazio delle misure: righe raggruppate per sorgente (N_det righe
    # per blocco), separatori orizzontali fra un blocco-sorgente e il
    # successivo
    U_img = U[:, i].reshape(N_src*N_det, N_gates)
    vmax_u = np.abs(U_img).max()
    axs[row, 0].imshow(U_img, aspect='auto', cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u,
                        origin='lower')
    for s_line in range(1, N_src):
        axs[row, 0].axhline(s_line*N_det - 0.5, color='k', linewidth=0.6)
    axs[row, 0].set_yticks([N_det*ii + N_det/2 - 0.5 for ii in range(N_src)])
    axs[row, 0].set_yticklabels([f'S{ii+1}' for ii in range(N_src)], fontsize=8)
    axs[row, 0].set_title(f"$u_{{{i}}}$ (spazio misure) — $s_{{{i}}}$={s_sing[i]:.2e}", fontsize=10)
    axs[row, 0].set_xlabel('gate')

    V_3d = Vt[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)
    V_section = V_3d[:, :, iz_show].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                        extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                                y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[row, 1].scatter(sources_xy[:, 0], sources_xy[:, 1], marker='*', c='k', s=40)
    axs[row, 1].scatter(detectors_xy[:, 0], detectors_xy[:, 1], marker='o', c='k', s=10, alpha=0.5)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_show]:.0f} mm)", fontsize=10)
fig.tight_layout()
plt.show()


# %% =============================================================
# STEP 8 - Ricostruzione a valori singolari troncati (TSVD), ideale vs con rumore
# ==================================================================
M_ideal_flat = M_ideal.flatten()
M_noisy_flat = np.nan_to_num(M_noisy, nan=0.0).flatten()

def tsvd_solve(M_flat, k):
    coeffs = (U[:, :k].T @ M_flat) / s_sing[:k]
    return Vt[:k, :].T @ coeffs

k_values = np.arange(1, k_max + 1)
rel_err_ideal = np.zeros_like(k_values, dtype=float)
rel_err_noisy = np.zeros_like(k_values, dtype=float)

for j, k in enumerate(k_values):
    A_hat_i = tsvd_solve(M_ideal_flat, k)
    A_hat_n = tsvd_solve(M_noisy_flat, k)
    rel_err_ideal[j] = np.linalg.norm(A_hat_i - A) / np.linalg.norm(A)
    rel_err_noisy[j] = np.linalg.norm(A_hat_n - A) / np.linalg.norm(A)

k_best = k_values[np.argmin(rel_err_noisy)]

plt.figure()
plt.semilogy(k_values, rel_err_ideal, 'o-', ms=3, label=r'errore su $A$ (dati IDEALI)')
plt.semilogy(k_values, rel_err_noisy, 's-', ms=3, label=r'errore su $A$ (dati con RUMORE)')
plt.axvline(k_best, color='gray', linestyle='--', linewidth=1,
            label=f'$k$ ottimale (oracolo) = {k_best}')
plt.xlabel('numero di valori singolari usati, $k$'); plt.ylabel('errore relativo $\\|\\hat A-A\\|/\\|A\\|$')
plt.title(f'Troncamento SVD ({N_src} sorgenti x {N_det} rivelatori): '
          'ideale (monotono) vs con rumore (minimo a k finito)')
plt.legend(); plt.grid(); plt.show()

print(f"Dati ideali: errore minimo = {rel_err_ideal.min():.3f} a k={k_values[np.argmin(rel_err_ideal)]}")
print(f"Dati con rumore: errore minimo = {rel_err_noisy.min():.3f} a k={k_best}")
print(f"Dati con rumore, a k={k_max} (rango pieno): errore = {rel_err_noisy[-1]:.3e}")


# %% =============================================================
# STEP 9 - Confronto visivo: A vero vs ricostruzioni (dati con rumore) a diversi k
# ==================================================================
k_show = sorted(set([max(1, k_best//4 if k_best > 4 else k_best),
                      k_best, min(k_max, k_best*3 if k_best*3 <= k_max else k_max)]))
iz_show2 = np.argmin(np.abs(z_coords - zp))

fig, axs = plt.subplots(1, len(k_show)+1, figsize=(4*(len(k_show)+1), 4.2))

A_section_true = A_rep[:, :, iz_show2].T
axs[0].imshow(A_section_true, origin='lower', cmap='viridis', vmin=0, vmax=dmu_a,
              extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                      y_coords[0]-step/2, y_coords[-1]+step/2))
axs[0].scatter(sources_xy[:, 0], sources_xy[:, 1], marker='*', c='red', s=90)
axs[0].scatter(detectors_xy[:, 0], detectors_xy[:, 1], marker='o', c='cyan', s=25)
axs[0].set_title('$A$ vero'); axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]')

for ax, k in zip(axs[1:], k_show):
    A_hat = tsvd_solve(M_noisy_flat, k)
    A_hat_3d = A_hat.reshape(x_coords.size, y_coords.size, z_coords.size)
    section = A_hat_3d[:, :, iz_show2].T
    vmax_hat = np.abs(A_hat).max()
    im = ax.imshow(section, origin='lower', cmap='RdBu_r', vmin=-vmax_hat, vmax=vmax_hat,
                    extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                            y_coords[0]-step/2, y_coords[-1]+step/2))
    ax.scatter(sources_xy[:, 0], sources_xy[:, 1], marker='*', c='k', s=90)
    ax.scatter(detectors_xy[:, 0], detectors_xy[:, 1], marker='o', c='k', s=20, alpha=0.5)
    ax.set_title(f'$\\hat A$ (rumore), $k$={k}\nerr={rel_err_noisy[k-1]:.2f}')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"Ricostruzione TSVD da dati con rumore, $z\\approx${z_coords[iz_show2]:.0f} mm "
             f"(vero: $x_p$={xp:.0f}, $y_p$={yp:.0f}, $z_p$={zp:.0f} mm)\n"
             f"{N_src} sorgenti multiplexate x {N_det} rivelatori")
fig.tight_layout()
plt.show()

# --- localizzazione: voxel di picco + massa negativa/positiva + numero
# di voxel per il 50% della massa ricostruita ---
print("="*55)
print("LOCALIZZAZIONE (dati con rumore): voxel di picco vs posizione vera")
print("="*55)
print(f"Vero voxel perturbato:  ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
for k in k_show:
    A_hat = tsvd_solve(M_noisy_flat, k)
    peak = r_V[np.argmax(A_hat)]
    pos_mass = np.clip(A_hat, 0, None).sum()
    neg_mass = np.clip(-A_hat, 0, None).sum()
    w_abs = np.abs(A_hat)
    idx_sorted = np.argsort(-w_abs)
    cum = np.cumsum(w_abs[idx_sorted]); cum /= cum[-1]
    n50 = np.searchsorted(cum, 0.5) + 1
    print(f"k={k:3d}:  picco = ({peak[0]:5.2f}, {peak[1]:5.2f}, {peak[2]:5.2f}) mm   "
          f"| massa negativa/positiva = {neg_mass/max(pos_mass,1e-30):.2f}   "
          f"| voxel per il 50% della massa |A_hat| = {n50}")
print("="*55)

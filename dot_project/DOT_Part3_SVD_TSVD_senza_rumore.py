# -*- coding: utf-8 -*-
"""
DOT Project - Parte 3: decomposizione ai valori singolari (SVD) di W e
ricostruzione a valori singolari troncati (TSVD), problema inverso,
senza rumore, mezzo semi-infinito.

Prosegue da DOT_Part2_W_A_M_senza_rumore.py: qui W, A e M = W . A sono
ricostruiti (Step 1-4, identici) e poi si affronta il problema inverso
(ricostruire A a partire da M) tramite SVD di W. I dati restano
IDEALI (nessun rumore poissoniano aggiunto a M): l'obiettivo e'
mostrare il comportamento "di base" della TSVD in assenza di rumore,
prima di introdurre eventuali sorgenti di errore in un secondo momento.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp

# %% =============================================================
# STEP 1 - Richiamo delle grandezze di Parte 1 e della griglia (per rendere
#          il file autosufficiente)
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

r_s0 = np.array([0.0, 0.0, z0])
r_si = np.array([0.0, 0.0, -z0 - 2*zb])

rho_sd = 12.0
angoli = np.arange(0, 360, 45)
det_xy = np.array([[rho_sd*np.cos(np.radians(a)),
                     rho_sd*np.sin(np.radians(a))] for a in angoli])
N_det = det_xy.shape[0]
# rivelatori: posizione reale a z=0 (superficie fisica), non z0
r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))
r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))

step = 4.0
x_ext = 64.0
z_ext = 32.0     # dimezzata rispetto alla versione originale (64 -> 32 mm)
x_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
y_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
z_coords = np.arange(0 + step/2, z_ext, step)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
r_V = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
N_vox = r_V.shape[0]
V_vox = step**3

dt = 0.05
t = np.arange(0.1, 10.0, dt)

# --- TIME GATING A FINESTRE FISSE (12 gate da 0.2 ns, "stile C") ---------
# Schema piu' conservativo rispetto ai precedenti: 12 gate da 200 ps sul
# range 1.0-3.4 ns (2.4 ns totali). Rispetto allo schema a 16 gate su 1-5
# ns, qui si sacrifica un po' di range temporale (e quindi di profondita'
# potenzialmente esplorabile) in cambio di conteggi per gate molto piu'
# solidi, restando in linea con lo schema reale a gate fissi riportato in
# Farina et al. 2017 (10 gate da 220 ps).
t_start = 1.0
gate_width = 0.20
N_gates = 12

gates = [(t_start + gate_width*i, t_start + gate_width*(i + 1)) for i in range(N_gates)]
gate_indices = [(np.searchsorted(t, g[0]), np.searchsorted(t, g[1])) for g in gates]

# controllo di sicurezza: ogni gate deve contenere almeno un paio di campioni
for g_i, (i0, i1) in enumerate(gate_indices):
    if i1 - i0 < 2:
        raise ValueError(f"Gate {g_i} ({gates[g_i]}) troppo stretto rispetto a dt={dt}: "
                          f"contiene solo {i1 - i0} campioni.")

print("="*55)
print("RICHIAMO: N_det =", N_det, "| N_vox =", N_vox, "| N_gates =", N_gates)
print(f"Misurazioni indipendenti: {N_det*N_gates}  |  Incognite (voxel): {N_vox}")
print(f"Gate fissi da {gate_width} ns, range {gates[0][0]:.1f}-{gates[-1][1]:.1f} ns")
print("="*55 + "\n")


# %% =============================================================
# STEP 2 - Vettore dei voxel A: perturbazione cubica di volume fisico esplicito
# ==================================================================
dmu_a = 0.01              # variazione di assorbimento nella perturbazione [mm^-1]
V_phys = V_vox             # volume FISICO della perturbazione [mm^3]: aumentare
                            # questo valore per estenderla su piu' voxel (es.
                            # con V_phys = V_vox si ottiene il caso puntiforme
                            # (singolo voxel), identico al comportamento precedente

# Non aumentare troppo dmu_a*V_phys per non uscire dall'approx di Born

xp, yp, zp = 14.0, 10.0, 18.0   # centro fisico della perturbazione [mm]

L_incl = V_phys**(1/3)     # lato del cubo di volume equivalente a V_phys
mask = ((np.abs(X - xp) <= L_incl/2) &
        (np.abs(Y - yp) <= L_incl/2) &
        (np.abs(Z - zp) <= L_incl/2))

A_rep = mask * dmu_a           # maschera booleana, valido finche' l'inclusione e' cubica
A = A_rep.flatten()               # vettore voxel 1D (stesso ordine di r_V)

N_incl = int(mask.sum())
V_incl = N_incl * V_vox           # volume EFFETTIVAMENTE rappresentato in griglia

ix, iy, iz_p = [np.argmin(np.abs(c - v)) for c, v in
                 zip((x_coords, y_coords, z_coords), (xp, yp, zp))]  # solo per i plot successivi

print(f"Perturbazione cubica: centro (xp,yp,zp) = ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm, "
      f"lato fisico L = {L_incl:.2f} mm, V_phys = {V_phys:.1f} mm^3\n"
      f"  -> rappresentata da {N_incl} voxel, V_incl (effettivo su griglia) = {V_incl:.1f} mm^3")


# --- rappresentazione 3D del vettore A nello spazio dei voxel ---
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

# --- sezione 2D nel piano z del voxel perturbato, con sorgente e rivelatori ---
plt.figure()
im = plt.imshow(A_rep[:, :, iz_p].T, origin='lower',
                 extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                         y_coords[0]-step/2, y_coords[-1]+step/2),
                 vmin=0, vmax=dmu_a)
plt.scatter(0, 0, marker='*', c='red', s=120, label='Sorgente')
plt.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', label='Rivelatori')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f"Sezione del vettore A nello spazio voxel\n"
          f"z = {zp:.0f} mm, step {step} mm")
plt.legend(fontsize=8)
plt.colorbar(im, label=r'$\delta\mu_a$ [$mm^{-1}$]')
plt.grid(); plt.show()


# %% =============================================================
# STEP 3 - Costruzione della matrice di sensitivita' (Jacobiano) W
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

r_sv  = np.linalg.norm(r_V - r_s0, axis=1)     # sorgente reale -> ogni voxel
r_siv = np.linalg.norm(r_V - r_si, axis=1)     # sorgente immagine -> ogni voxel

W = np.zeros((N_det*N_gates, N_vox))
tt = t[None, :]                                # riga (1, N_t), per il broadcasting

for d in range(N_det):
    r_vd  = np.linalg.norm(r_V - r_d0[d], axis=1)   # voxel -> rivelatore reale
    r_vdi = np.linalg.norm(r_V - r_di[d], axis=1)   # voxel -> rivelatore immagine

    dphi_vt = (deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vd[:, None])
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vdi[:, None])
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vd[:, None])
             + deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vdi[:, None]))

    r_d_s  = np.linalg.norm(r_d0[d] - r_s0)
    r_d_si = np.linalg.norm(r_d0[d] - r_si)
    phi0_t = fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s) - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si)

    for g, (i0, i1) in enumerate(gate_indices):
        # somma di Riemann (rettangoli): ogni campione pesa dt, nessun peso 1/2 ai bordi
        int_dphi = np.sum(dphi_vt[:, i0:i1], axis=1) * dt   # (N_vox,)
        int_phi0 = np.sum(phi0_t[i0:i1]) * dt                # scalare
        W[d*N_gates + g, :] = int_dphi / int_phi0

print(f"Matrice W costruita: shape = {W.shape}")

# --- sezioni 2D di W per il rivelatore D1, a diverse profondita' e gate ---
# Con 12 gate, scegliamo indici distribuiti sull'intero range (early/mid/late)
W_3d = W.reshape(N_det, N_gates, x_coords.size, y_coords.size, z_coords.size)
d_show = 0
z_show = [10.0, 18.0, 30.0]
g_show = [0, 5, 11]
Wlog_vmin, Wlog_vmax = -4, 1
fig, axs = plt.subplots(3, 3, figsize=(10, 11))
for r_i, zval in enumerate(z_show):
    iz = np.argmin(np.abs(z_coords - zval))
    for c_i, g in enumerate(g_show):
        Wsec = np.log10(np.clip(np.abs(W_3d[d_show, g, :, :, iz]), 1e-8, None)).T
        im = axs[r_i, c_i].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
        axs[r_i, c_i].scatter(0, 0, marker='*', c='red', s=30)
        axs[r_i, c_i].scatter(det_xy[d_show, 0], det_xy[d_show, 1], marker='o', c='lime', s=20)
        axs[r_i, c_i].set_title(f"z={z_coords[iz]:.0f} mm, gate {g} "
                                 f"({gates[g][0]:.1f}-{gates[g][1]:.1f} ns)", fontsize=9, pad=8)
        if c_i == 0: axs[r_i, c_i].set_ylabel('y [mm]')
        if r_i == 2: axs[r_i, c_i].set_xlabel('x [mm]')
fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, right=0.86)
fig.suptitle(r'Sezioni di $\log_{10}|W|$ nello spazio voxel - rivelatore D1'
             '\n(scala colore condivisa fra tutti i pannelli, 12 gate da 0.2 ns)')
fig.colorbar(im, ax=axs, shrink=0.7, label=r'$\log_{10}|W|$')
plt.show()

# --- confronto fra tutti gli 8 rivelatori, a profondita' e gate fissati ---
# Nota: W dipende solo dalla geometria sorgente-voxel-rivelatore sul mezzo
# di fondo omogeneo (deltafluence_inf viene chiamata con dmu_a=1.0), quindi
# NON riflette la posizione o l'intensita' dell'inclusione reale A: la
# simmetria di rotazione fra i pannelli e' quindi attesa e corretta, non
# un'anomalia. Per vedere l'effetto dell'inclusione bisogna guardare M=W@A.
z_fix = 18.0
g_fix = 3        # gate precoce (1.6-1.8 ns): fotoni meno diffusi, mappa di
                 # sensitivita' piu' localizzata rispetto ai gate tardivi
iz = np.argmin(np.abs(z_coords - z_fix))
fig, axs = plt.subplots(2, 4, figsize=(14, 7))
axs = axs.flatten()
for d in range(N_det):
    Wsec = np.log10(np.clip(np.abs(W_3d[d, g_fix, :, :, iz]), 1e-8, None)).T
    im = axs[d].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[d].scatter(0, 0, marker='*', c='cyan', s=40)
    axs[d].scatter(det_xy[d, 0], det_xy[d, 1], marker='o', c='lime', s=25)
    axs[d].set_title(f'D{d+1}')
fig.suptitle(f"Sensitivita' " r"$\log_{10}|W|$" f" per ciascun rivelatore\n"
             f"z = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()


# %% =============================================================
# STEP 4 - Vettore delle misure M = W . A (problema diretto, senza rumore)
# ==================================================================
M_ideal = (W @ A).reshape(N_det, N_gates)

print("VETTORE DELLE MISURE M = W . A (ideale, senza rumore)")
for d in range(N_det):
    print(f"  D{d+1}: " + " ".join(f"{val:+.4f}" for val in M_ideal[d, :]))

plt.figure(figsize=(8, 5))
im = plt.imshow(M_ideal, aspect='auto', cmap='RdBu_r',
                 vmin=-np.abs(M_ideal).max(), vmax=np.abs(M_ideal).max())
plt.yticks(range(N_det), [f'D{d+1}' for d in range(N_det)])
plt.xticks(range(N_gates), [f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90)
plt.xlabel('time gate [ns] (12 gate da 0.2 ns)'); plt.ylabel('rivelatore')
plt.title('Vettore delle misure M = W . A (ideale, mezzo semi-infinito)\n12 gate fissi da 0.2 ns, range 1.0-3.4 ns')
plt.colorbar(im, label='C (-)')
plt.tight_layout(); plt.show()

M_ideal_flat = M_ideal.flatten()          # stesso ordine (d*N_gates+g) delle righe di W


# %% =============================================================
# STEP 5 - Decomposizione ai valori singolari di W
# ==================================================================
U, s, Vt = np.linalg.svd(W, full_matrices=False)
k_max = s.size   # = min(N_det*N_gates, N_vox)

print("="*55)
print("SVD DI W")
print("="*55)
print(f"U: {U.shape}   s: {s.shape}   Vt: {Vt.shape}")
print(f"Valore singolare massimo  s_0        = {s[0]:.3e}")
print(f"Valore singolare minimo   s_{k_max-1:<3d} = {s[-1]:.3e}")
print(f"Rapporto s_0/s_min (numero di condizionamento di W) = {s[0]/s[-1]:.2e}")
print("="*55 + "\n")

plt.figure()
plt.semilogy(np.arange(1, k_max + 1), s, 'o-', ms=4)
plt.xlabel('ordine $i$'); plt.ylabel('$s_i$')
plt.title("Valori singolari di $W$ (scala logaritmica)")
plt.grid(); plt.show()


# %% =============================================================
# STEP 6 - Alcuni modi (coppie u_i, v_i): dai piu' "forti" ai piu' "deboli"
# ==================================================================
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
iz_show = np.argmin(np.abs(z_coords - zp))   # stessa quota della sezione di A

fig, axs = plt.subplots(len(orders_to_show), 2, figsize=(8, 3.1*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    U_img = U[:, i].reshape(N_det, N_gates)
    vmax_u = np.abs(U_img).max()
    axs[row, 0].imshow(U_img, aspect='auto', cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u,
                        origin='lower')
    axs[row, 0].set_title(f"$u_{{{i}}}$ (spazio misure) — $s_{{{i}}}$={s[i]:.2e}", fontsize=10)
    axs[row, 0].set_xlabel('gate'); axs[row, 0].set_ylabel('rivelatore (0=D1)')

    V_3d = Vt[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)
    V_section = V_3d[:, :, iz_show].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                        extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                                y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[row, 1].scatter(0, 0, marker='*', c='k', s=40)
    axs[row, 1].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=10, alpha=0.5)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_show]:.0f} mm)", fontsize=10)
fig.tight_layout()
plt.show()


# %% =============================================================
# STEP 7 - Ricostruzione a valori singolari troncati (TSVD), dati ideali
# ==================================================================
# A_hat(k) = V[:,:k] @ diag(1/s[:k]) @ U[:,:k].T @ M
#
# Senza rumore, dividere per valori singolari piccoli non amplifica
# alcun errore di misura (non ce n'e'): l'unico effetto del troncamento
# e' escludere componenti di A che pesano poco sulla misura. Ci si
# aspetta quindi un comportamento MONOTONO: l'errore di ricostruzione
# non puo' che diminuire (o al piu' restare costante) man mano che si
# aggiungono valori singolari, con il minimo raggiunto a k = k_max
# (rango pieno). Questo e' l'opposto del comportamento che si osserva
# con dati rumorosi, dove il rumore nei modi a piccolo s_i domina e
# produce un minimo a k intermedio (regolarizzazione per troncamento) -
# comportamento che verra' analizzato separatamente introducendo il
# rumore sulla misura.

def tsvd_solve(M_flat, k):
    coeffs = (U[:, :k].T @ M_flat) / s[:k]
    return Vt[:k, :].T @ coeffs

k_values = np.arange(1, k_max + 1)
rel_err_ideal = np.zeros_like(k_values, dtype=float)

for j, k in enumerate(k_values):
    A_hat_i = tsvd_solve(M_ideal_flat, k)
    rel_err_ideal[j] = np.linalg.norm(A_hat_i - A) / np.linalg.norm(A)

k_min_err = k_values[np.argmin(rel_err_ideal)]

plt.figure()
plt.semilogy(k_values, rel_err_ideal, 'o-', ms=3, label=r'errore su $A$ (dati ideali)')
plt.axvline(k_min_err, color='gray', linestyle='--', linewidth=1,
            label=f'$k$ di errore minimo = {k_min_err}')
plt.xlabel('numero di valori singolari usati, $k$'); plt.ylabel('errore relativo $\\|\\hat A-A\\|/\\|A\\|$')
plt.title('Troncamento SVD con dati ideali: errore monotono in $k$')
plt.legend(); plt.grid(); plt.show()

print(f"Dati ideali: errore minimo = {rel_err_ideal.min():.3e} a k={k_min_err} (su k_max={k_max})")


# %% =============================================================
# STEP 8 - Confronto visivo: A vero vs ricostruzioni (dati ideali) a diversi k
# ==================================================================
k_show = sorted(set([max(1, k_max//8), k_max//2, k_max]))
iz_show2 = np.argmin(np.abs(z_coords - zp))

fig, axs = plt.subplots(1, len(k_show)+1, figsize=(4*(len(k_show)+1), 4.2))

A_section_true = A_rep[:, :, iz_show2].T
axs[0].imshow(A_section_true, origin='lower', cmap='viridis', vmin=0, vmax=dmu_a,
              extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                      y_coords[0]-step/2, y_coords[-1]+step/2))
axs[0].scatter(0, 0, marker='*', c='red', s=100)
axs[0].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', s=30)
axs[0].set_title('$A$ vero'); axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]')

for ax, k in zip(axs[1:], k_show):
    A_hat = tsvd_solve(M_ideal_flat, k)
    A_hat_3d = A_hat.reshape(x_coords.size, y_coords.size, z_coords.size)
    section = A_hat_3d[:, :, iz_show2].T
    vmax_hat = np.abs(A_hat).max()
    im = ax.imshow(section, origin='lower', cmap='RdBu_r', vmin=-vmax_hat, vmax=vmax_hat,
                    extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                            y_coords[0]-step/2, y_coords[-1]+step/2))
    ax.scatter(0, 0, marker='*', c='k', s=100)
    ax.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=20, alpha=0.5)
    ax.set_title(f'$\\hat A$ (ideale), $k$={k}\nerr={rel_err_ideal[k-1]:.2e}')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"Ricostruzione TSVD da dati ideali, $z\\approx${z_coords[iz_show2]:.0f} mm "
             f"(vero: $x_p$={xp:.0f}, $y_p$={yp:.0f}, $z_p$={zp:.0f} mm)")
fig.tight_layout()
plt.show()

# --- localizzazione: voxel di picco (non centroide |A_hat|: i lobi
# negativi della soluzione a norma minima "pesano" nella media come se
# fossero segnale positivo) + massa negativa/positiva + numero di voxel
# per il 50% della massa ricostruita ---
print("="*55)
print("LOCALIZZAZIONE (dati ideali): voxel di picco vs posizione vera")
print("="*55)
print(f"Vero voxel perturbato:  ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
for k in k_show:
    A_hat = tsvd_solve(M_ideal_flat, k)
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

# --- confronto di INTENSITA': dmu_a vero vs ricostruito ---
# Due letture distinte: (a) il valore ricostruito esattamente nel/i
# voxel dove si trova la perturbazione vera (localizzazione nota a
# priori, qui usata solo come sonda) e (b) il valore di picco globale
# di A_hat, ovunque esso cada (la lettura realistica, quando la
# posizione vera non e' nota).
idx_true = np.flatnonzero(mask.flatten())   # voxel della perturbazione vera

print("\n" + "="*55)
print("CONFRONTO INTENSITA': dmu_a vero vs dmu_a ricostruito")
print("="*55)
print(f"dmu_a vero (nella perturbazione) = {dmu_a:+.4f} mm^-1")
for k in k_show:
    A_hat = tsvd_solve(M_ideal_flat, k)
    val_true_voxel = A_hat[idx_true].mean()
    val_peak = A_hat[np.argmax(A_hat)]
    print(f"k={k:3d}:  ricostruito nel voxel vero = {val_true_voxel:+.4f} mm^-1 "
          f"(rapporto ric./vero = {val_true_voxel/dmu_a:+.2f})   "
          f"| picco globale ricostruito = {val_peak:+.4f} mm^-1 "
          f"(rapporto picco/vero = {val_peak/dmu_a:+.2f})")
print("="*55)


# %% =============================================================
# STEP 9 - Ricostruzione vincolata (detect-then-fit): forzare tutta la
#          perturbazione nel voxel di picco individuato dalla TSVD
# ==================================================================
# Una volta localizzato il voxel piu' plausibile (picco di A_hat), si
# puo' risolvere un problema molto piu' piccolo e ben posto: un'unica
# incognita scalare (l'ampiezza dmu_a in quel voxel), contro le 96
# misure disponibili. E' un minimo quadrato 1D lineare:
#
#   min_x || W[:, j]*x - M ||^2   =>   x = (W[:,j] . M) / (W[:,j] . W[:,j])
#
# A differenza della TSVD "libera" (che distribuisce l'energia su
# centinaia di voxel per minimizzare la norma della soluzione), qui si
# impone a priori che la soluzione sia sparsa e concentrata in un solo
# voxel: il problema passa da fortemente sottodeterminato (96 misure,
# 2048 incognite) a fortemente sovradeterminato (96 misure, 1
# incognita), quindi l'ampiezza recuperata e' molto meno smorzata.
# Il prezzo e' che questa ipotesi (un solo voxel, nella posizione
# individuata dalla TSVD) va verificata: se e' sbagliata (voxel di
# picco errato, inclusione estesa o multipla) il fit forza comunque
# tutta l'energia li', introducendo un bias sistematico.

print("\n" + "="*55)
print("RICOSTRUZIONE VINCOLATA (detect-then-fit): forzatura sul voxel di picco")
print("="*55)
print(f"dmu_a vero (nella perturbazione) = {dmu_a:+.4f} mm^-1\n")
for k in k_show:
    A_hat = tsvd_solve(M_ideal_flat, k)
    idx_peak = np.argmax(A_hat)
    peak_pos = r_V[idx_peak]
    voxel_corretto = idx_peak in idx_true

    w_j = W[:, idx_peak]
    x_fit = (w_j @ M_ideal_flat) / (w_j @ w_j)
    resid_rel = np.linalg.norm(w_j*x_fit - M_ideal_flat) / np.linalg.norm(M_ideal_flat)

    print(f"k={k:3d}:  voxel di picco = ({peak_pos[0]:5.2f}, {peak_pos[1]:5.2f}, {peak_pos[2]:5.2f}) mm "
          f"{'(voxel vero)' if voxel_corretto else '(voxel SBAGLIATO)'}")
    print(f"        dmu_a forzato su quel voxel = {x_fit:+.4f} mm^-1 "
          f"(rapporto forzato/vero = {x_fit/dmu_a:+.2f})   "
          f"| residuo relativo del fit sui dati = {resid_rel:.2e}")
print("="*55)


# %% =============================================================
# STEP 10 - Effetto della dimensione dell'inclusione sul broadening
#           di intensita' (dmu_a fisso, volume variabile)
# ==================================================================
# Negli Step 8/9 si e' visto che una perturbazione PUNTIFORME (1
# voxel) viene fortemente attenuata in ampiezza dalla ricostruzione
# TSVD "libera" (senza vincolo di sparsita'): la soluzione a norma
# minima spalma la sua energia su centinaia di voxel. Qui si verifica
# se questo effetto dipende dalla dimensione fisica dell'inclusione: a
# dmu_a fisso, si allarga progressivamente il cubo perturbato (stesso
# centro xp,yp,zp di prima) e si confronta l'ampiezza ricostruita (a
# rango pieno, k=k_max, il caso migliore fra i troncamenti visto lo
# Step 7) con il vero dmu_a.
#
# Intuizione: poiche' W e' lineare e non dipende da A, la ricostruzione
# a norma minima si comporta come una convoluzione dell'oggetto vero
# con una point-spread function (PSF) fissata dalla geometria del
# sistema (rivelatori/gate), non dall'oggetto stesso. Un'inclusione
# piu' piccola della PSF ha tutta la sua energia diluita nella
# sfocatura (forte sottostima, come visto per il singolo voxel); una
# inclusione piu' grande della PSF e' invece sfocata solo sui bordi:
# al centro ogni voxel "riceve" contributo da voxel vicini con lo
# STESSO dmu_a vero, quindi l'ampiezza recuperata nel nucleo
# dell'inclusione si riavvicina (e puo' perfino superare leggermente,
# per effetti di discretizzazione a rango finito) il valore vero.

# lati testati: 4, 8, 16, 24 mm - scelti per restare entro i margini
# della griglia rispetto al centro (xp,yp,zp) = (14, 10, 18) mm
L_test_values = [step, 2*step, 4*step, 6*step]
center_p = np.array([xp, yp, zp])

peak_ratios, core_ratios, N_incl_list = [], [], []

print("="*55)
print("EFFETTO DELLA DIMENSIONE DELL'INCLUSIONE SUL BROADENING (k = k_max)")
print("="*55)
print(f"dmu_a vero (nella perturbazione) = {dmu_a:+.4f} mm^-1")
print(f"{'L_incl [mm]':>12} {'N_incl':>7} {'V_incl [mm^3]':>14} {'picco/vero':>12} {'nucleo/vero':>12}")
for L_test in L_test_values:
    mask_test = ((np.abs(X - xp) <= L_test/2) &
                 (np.abs(Y - yp) <= L_test/2) &
                 (np.abs(Z - zp) <= L_test/2))
    N_incl_test = int(mask_test.sum())
    if N_incl_test == 0:
        continue

    A_test = (mask_test * dmu_a).flatten()
    M_test_flat = W @ A_test
    A_hat_test = tsvd_solve(M_test_flat, k_max)

    idx_true_test = np.flatnonzero(mask_test.flatten())
    idx_core = idx_true_test[np.argmin(np.linalg.norm(r_V[idx_true_test] - center_p, axis=1))]

    peak_ratio = A_hat_test.max() / dmu_a
    core_ratio = A_hat_test[idx_core] / dmu_a
    peak_ratios.append(peak_ratio); core_ratios.append(core_ratio); N_incl_list.append(N_incl_test)

    print(f"{L_test:12.1f} {N_incl_test:7d} {N_incl_test*V_vox:14.0f} {peak_ratio:12.3f} {core_ratio:12.3f}")
print("="*55)

plt.figure(figsize=(6, 4.5))
plt.plot(L_test_values[:len(core_ratios)], core_ratios, 'o-', label='nucleo (voxel centrale) / vero')
plt.plot(L_test_values[:len(peak_ratios)], peak_ratios, 's--', label='picco globale / vero')
plt.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='ricostruzione esatta')
plt.xlabel("lato dell'inclusione cubica [mm]")
plt.ylabel(r'rapporto $\hat{A}$ / $\delta\mu_a$ vero')
plt.title("Recupero di ampiezza al crescere della dimensione dell'inclusione\n"
          "(dati ideali, ricostruzione TSVD a rango pieno k=k_max)")
plt.legend(); plt.grid(); plt.show()

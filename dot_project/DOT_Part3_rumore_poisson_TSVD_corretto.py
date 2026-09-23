# -*- coding: utf-8 -*-
"""
DOT Project - Parte 3 (versione corretta): rumore poissoniano (shot noise)
sul vettore delle misure M e ricostruzione TSVD pesata.

Correzioni rispetto a DOT_Part3_rumore_poisson_TSVD.py (vedi
DOT_Part3_rumore_poissoniano.ipynb per la derivazione):
  1. budget fotonico riferito all'INTERA TPSF, non ai soli gate;
  2. stimatore del contrasto quasi non distorto N_p/(N_0+1) - 1;
  3. sigma_M stimata dai conteggi misurati e TSVD pesata (whitening);
  4. scelta di k senza oracolo (principio di discrepanza), oltre al k oracolo;
  5. metriche: errore indotto dal rumore e localizzazione, oltre a ||A_hat-A||;
  6. generatore random moderno, commenti e titoli corretti, variabili inutili rimosse.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp

# STEP 1 - Richiamo delle grandezze di Parte 1 e della griglia (per rendere
#          il file autosufficiente)
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



# STEP 2 - Vettore dei voxel A: perturbazione cubica di volume fisico esplicito

dmu_a = 0.01              # variazione di assorbimento nella perturbazione [mm^-1]
V_phys = V_vox             # volume FISICO della perturbazione [mm^3]: aumentare
                            # questo valore per estenderla su piu' voxel;
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

ix, iy, iz_p = [np.argmin(np.abs(c - c0)) for c, c0 in
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
             f"Step {step} mm, {N_incl} voxel perturbati, volume {V_incl:.0f} mm$^3$")
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


# STEP 3 - Costruzione della matrice di sensitivita' (Jacobiano) W
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
g_fix = 3        # gate relativamente precoce (1.6-1.8 ns): cammini piu' brevi,
                 # banana piu' stretta e meno profonda rispetto ai gate tardivi
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


# STEP 4 - Vettore delle misure M = W . A (problema diretto, senza rumore)
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


# STEP 5 - Rumore poissoniano (shot noise) sui CONTEGGI, non su M
# ================================================================================
# In TCSPC il dato grezzo e' il numero di fotoni per gate:
#     N_dg ~ Poisson(lambda_dg),   E[N] = Var[N] = lambda,   SNR = sqrt(lambda)
# con
#     lambda0_dg = eta_d * int_g Phi0_d dt          (baseline)
#     lambdap_dg = lambda0_dg * (1 + M_dg)           (con inclusione, Born)
# Il contrasto stimato da due acquisizioni indipendenti ha varianza
#     sigma_M^2 ~ (1+M)(2+M)/lambda0  ~ 2/lambda0     (metodo delta)
# Il fattore 2 nasce dal fatto che anche la baseline e' misurata (rumorosa).
# Lo shot noise e' la conseguenza intrinseca della natura discreta della
# rivelazione di fotoni, quindi irriducibile. I gate con pochi conteggi sono
# quelli TARDIVI: sono i piu' sensibili alle inclusioni profonde, ma anche i
# piu' rumorosi (sigma_M cresce di ~35 volte fra primo e ultimo gate).
#
# NOTA (inverse crime): i dati sono generati con lo stesso modello lineare
# (Born, stessa griglia) usato per l'inversione; il rumore e' l'unica
# sorgente d'errore, quindi i risultati sono ottimistici.

rng = np.random.default_rng(42)  # riproducibilita'

# --- 1) Integrali di phi0: nei gate e sull'intera TPSF ---
Phi0_int = np.zeros((N_det, N_gates))
Phi0_tot = np.zeros(N_det)
for d in range(N_det):
    r_d_s = np.linalg.norm(r_d0[d] - r_s0)
    r_d_si = np.linalg.norm(r_d0[d] - r_si)
    phi0_t = (fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s)
              - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si))
    Phi0_tot[d] = np.sum(phi0_t) * dt
    for g, (i0, i1) in enumerate(gate_indices):
        Phi0_int[d, g] = np.sum(phi0_t[i0:i1]) * dt

frac_gates = Phi0_int.sum(axis=1) / Phi0_tot
print(f"Picco della TPSF a t = {t[np.argmax(phi0_t)]:.2f} ns (primo gate a {gates[0][0]:.1f} ns)")
print(f"Frazione della TPSF contenuta nei {N_gates} gate: {100*frac_gates[0]:.2f} %")

# --- 2) Budget fotonico riferito all'INTERA TPSF ---
# Il picco della TPSF (~0.2 ns) cade PRIMA del primo gate: i gate campionano
# solo la coda (~2.5% dei fotoni). N_tot_TPSF e' quindi il numero di
# conteggi dell'intera curva, e ai gate ne arriva solo la frazione relativa.
# Con 1e6 conteggi/TPSF lo SNR complessivo e' ~0.2 (inclusione invisibile);
# 1e8 (~30-100 s a qualche Mcps) porta lo SNR complessivo a ~2.5.
N_tot_TPSF = 1e8
N0_ideal = N_tot_TPSF * Phi0_int / Phi0_tot[:, None]

# --- 3) Conteggi ideali della misura perturbata (con inclusione) ---
Npert_ideal = N0_ideal * (1 + M_ideal)

# --- 4) Realizzazioni poissoniane indipendenti delle due acquisizioni ---
N0_noisy = rng.poisson(N0_ideal)
Npert_noisy = rng.poisson(np.clip(Npert_ideal, 0, None))

# --- 5) Contrasto stimato dai conteggi ---
# (N_p - N_0)/N_0 ha bias ~ 1/lambda0 (E[1/N] != 1/lambda), confrontabile
# con |M| nei gate tardivi. N_p/(N_0+1) - 1 e' quasi non distorto
# (E[1/(N_0+1)] = (1 - exp(-lambda0))/lambda0) e non divide mai per zero.
M_noisy = Npert_noisy / (N0_noisy + 1) - 1

# --- 6) Deviazione standard di M stimata DAI DATI (come su misure reali) ---
sigma_M = np.sqrt((Npert_noisy + 1) / (N0_noisy + 1)**2
                  + (Npert_noisy + 1)**2 / (N0_noisy + 1)**3)
sigma_M_th = np.sqrt((1 + M_ideal) * (2 + M_ideal) / N0_ideal)   # solo per confronto
SNR_gate = np.abs(M_ideal) / sigma_M_th

print(f"Budget fotonico: N_tot = {N_tot_TPSF:.0e} conteggi/TPSF/rivelatore")
print(f"Conteggi di baseline per gate: min {N0_ideal.min():.1f} / max {N0_ideal.max():.1f}")
print(f"SNR per gate: min {SNR_gate.min():.2f} / max {SNR_gate.max():.2f}  |  "
      f"SNR complessivo sqrt(sum SNR^2) = {np.sqrt((SNR_gate**2).sum()):.2f}")

# --- Plot 1: confronto matriciale M ideale vs M con rumore ---
vmax = np.abs(np.concatenate([M_ideal.ravel(), M_noisy.ravel()])).max()
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, Mplot, title in zip(
        axs, [M_ideal, M_noisy],
        ['M ideale (senza rumore)', 'M con rumore poissoniano (shot noise)']):
    im = ax.imshow(Mplot, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_gates))
    ax.set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates],
                        rotation=90, fontsize=7)
    ax.set_xlabel('time gate [ns]')
    ax.set_title(title)
axs[0].set_yticks(range(N_det))
axs[0].set_yticklabels([f'D{d+1}' for d in range(N_det)])
axs[0].set_ylabel('rivelatore')
fig.colorbar(im, ax=axs, shrink=0.8, label='C (-)')
fig.suptitle(f'Confronto M ideale vs M con shot noise '
             f'(N_tot = {N_tot_TPSF:.0e} conteggi/TPSF/rivelatore)')
plt.show()

# --- Plot 2: profilo di un singolo rivelatore con banda +-sigma e SNR ---
d_show = 0
g_idx = np.arange(N_gates)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(g_idx, M_ideal[d_show], 'o-', label='M ideale')
axs[0].errorbar(g_idx, M_noisy[d_show], yerr=sigma_M[d_show], fmt='s', capsize=3,
                label=r'M con rumore $\pm\hat\sigma_M$')
axs[0].set(xlabel='indice gate temporale', ylabel='C (-)',
           title=f'Rivelatore D{d_show + 1}: ideale vs rumoroso')
axs[0].legend(); axs[0].grid()
axs[1].semilogy(g_idx, SNR_gate[d_show], 'o-')
axs[1].axhline(1, color='r', ls='--', label='SNR = 1')
axs[1].set(xlabel='indice gate temporale', ylabel=r'$|M|/\sigma_M$',
           title=f'SNR per gate (D{d_show + 1})')
axs[1].legend(); axs[1].grid()
plt.tight_layout(); plt.show()


# STEP 6 - SVD di W e della matrice pesata W_w = Sigma^{-1/2} W
# Il rumore e' eteroschedastico: la soluzione a massima verosimiglianza
# minimizza ||Sigma^{-1/2}(W A - M)||^2, quindi la TSVD va fatta su W_w.
U, s, Vt = np.linalg.svd(W, full_matrices=False)
sig_flat = sigma_M.flatten()                   # stesso ordine (d*N_gates+g) delle righe di W
W_w = W / sig_flat[:, None]
Uw, sw, Vtw = np.linalg.svd(W_w, full_matrices=False)
k_max = s.size   # = min(N_det*N_gates, N_vox)

print("="*55)
print("SVD DI W (non pesata) e di W_w (pesata)")
print("="*55)
print(f"U: {U.shape}   s: {s.shape}   Vt: {Vt.shape}")
print(f"W  : s_0 = {s[0]:.3e},  s_min = {s[-1]:.3e},  cond = {s[0]/s[-1]:.2e}")
print(f"W_w: s_0 = {sw[0]:.3e},  s_min = {sw[-1]:.3e},  cond = {sw[0]/sw[-1]:.2e}")
print("="*55 + "\n")

plt.figure()
plt.semilogy(np.arange(1, k_max + 1), s/s[0], 'o-', ms=4, label='W')
plt.semilogy(np.arange(1, k_max + 1), sw/sw[0], 's-', ms=4, label=r'$W_w=\Sigma^{-1/2}W$')
plt.xlabel('ordine $i$'); plt.ylabel('$s_i/s_0$')
plt.title("Valori singolari normalizzati (scala logaritmica)")
plt.legend(); plt.grid(); plt.show()


# STEP 7 - Alcuni modi (coppie u_i, v_i) della matrice pesata
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
iz_show = np.argmin(np.abs(z_coords - zp))   # stessa quota della sezione di A

fig, axs = plt.subplots(len(orders_to_show), 2, figsize=(8, 3.1*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    U_img = Uw[:, i].reshape(N_det, N_gates)
    vmax_u = np.abs(U_img).max()
    axs[row, 0].imshow(U_img, aspect='auto', cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u,
                        origin='lower')
    axs[row, 0].set_title(f"$u_{{{i}}}$ (spazio misure) — $s_{{{i}}}$={sw[i]:.2e}", fontsize=10)
    axs[row, 0].set_xlabel('gate'); axs[row, 0].set_ylabel('rivelatore (0=D1)')

    V_3d = Vtw[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)
    V_section = V_3d[:, :, iz_show].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                        extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                                y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[row, 1].scatter(0, 0, marker='*', c='k', s=40)
    axs[row, 1].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=10, alpha=0.5)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_show]:.0f} mm)", fontsize=10)
fig.suptitle(r'Modi singolari di $W_w$ (pesata)', y=1.0)
fig.tight_layout()
plt.show()


# STEP 8 - Ricostruzione TSVD: non pesata vs pesata, ideale vs con rumore
# A_hat(k) = V[:,:k] diag(1/s[:k]) U[:,:k]^T M      (M -> M/sigma per la pesata)
# Con dati rumorosi dividere per s_i piccoli amplifica il rumore: il
# troncamento e' una regolarizzazione (compromesso bias/varianza).
M_ideal_flat = M_ideal.flatten()
M_noisy_flat = M_noisy.flatten()
M_noisy_w = M_noisy_flat / sig_flat
M_ideal_w = M_ideal_flat / sig_flat

def tsvd_solve(Uu, ss, VVt, m, k):
    return VVt[:k, :].T @ ((Uu[:, :k].T @ m) / ss[:k])

def peak_dist(A_hat):
    return np.linalg.norm(r_V[np.argmax(A_hat)] - np.array([xp, yp, zp]))

k_values = np.arange(1, k_max + 1)
normA = np.linalg.norm(A)
err_ideal, err_raw, err_w = (np.zeros(k_max) for _ in range(3))
noise_raw, noise_w = np.zeros(k_max), np.zeros(k_max)
chi2_w, loc_raw, loc_w = np.zeros(k_max), np.zeros(k_max), np.zeros(k_max)

for j, k in enumerate(k_values):
    A_i   = tsvd_solve(U, s, Vt, M_ideal_flat, k)
    A_r   = tsvd_solve(U, s, Vt, M_noisy_flat, k)
    A_wi  = tsvd_solve(Uw, sw, Vtw, M_ideal_w, k)
    A_w   = tsvd_solve(Uw, sw, Vtw, M_noisy_w, k)
    err_ideal[j] = np.linalg.norm(A_i - A) / normA
    err_raw[j]   = np.linalg.norm(A_r - A) / normA
    err_w[j]     = np.linalg.norm(A_w - A) / normA
    noise_raw[j] = np.linalg.norm(A_r - A_i) / normA     # errore dovuto al solo rumore
    noise_w[j]   = np.linalg.norm(A_w - A_wi) / normA
    chi2_w[j]    = np.sum((W_w @ A_w - M_noisy_w)**2)    # residuo pesato (chi^2)
    loc_raw[j], loc_w[j] = peak_dist(A_r), peak_dist(A_w)

# Scelta di k SENZA conoscere A (principio di discrepanza di Morozov):
# il piu' piccolo k per cui il residuo pesato scende al livello del rumore,
# chi^2 <= N_misure (valore atteso di chi^2 per rumore a varianza unitaria).
N_meas = N_det * N_gates
ok = np.where(chi2_w <= N_meas)[0]
k_disc = int(k_values[ok[0]]) if ok.size else k_max
k_best = int(k_values[np.argmin(err_w)])   # k "oracolo": richiede A, solo per confronto

fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
axs[0].semilogy(k_values, err_ideal, 'o-', ms=3, label='dati ideali')
axs[0].semilogy(k_values, err_raw, 's-', ms=3, label='rumore, TSVD non pesata')
axs[0].semilogy(k_values, err_w, '^-', ms=3, label='rumore, TSVD pesata')
axs[0].axvline(k_disc, color='k', ls='--', lw=1, label=f'$k$ discrepanza = {k_disc}')
axs[0].axvline(k_best, color='gray', ls=':', lw=1, label=f'$k$ oracolo = {k_best}')
axs[0].set(xlabel='$k$', ylabel=r'$\|\hat A-A\|/\|A\|$', title='Errore su A')
axs[0].legend(fontsize=8); axs[0].grid()
axs[1].semilogy(k_values, noise_raw, 's-', ms=3, label='non pesata')
axs[1].semilogy(k_values, noise_w, '^-', ms=3, label='pesata')
axs[1].set(xlabel='$k$', ylabel=r'$\|\hat A_{noisy}(k)-\hat A_{ideal}(k)\|/\|A\|$',
           title='Errore indotto dal solo rumore')
axs[1].legend(); axs[1].grid()
axs[2].semilogy(k_values, chi2_w, 'o-', ms=3)
axs[2].axhline(N_meas, color='r', ls='--', label=r'$\chi^2 = N_{misure}$')
axs[2].axvline(k_disc, color='k', ls='--', lw=1)
axs[2].set(xlabel='$k$', ylabel=r'$\chi^2=\|W_w\hat A-M_w\|^2$', title='Principio di discrepanza')
axs[2].legend(); axs[2].grid()
plt.tight_layout(); plt.show()

# NB: ||A_hat-A||/||A|| resta ~0.97 anche con dati ideali: la soluzione a
# norma minima di un sistema 96 x 2048 non puo' concentrare la massa in un
# singolo voxel. Per questo si guardano anche errore da rumore e localizzazione.
print(f"Dati ideali: errore minimo = {err_ideal.min():.3f} a k={k_values[np.argmin(err_ideal)]}")
print(f"Rumore, non pesata: errore minimo = {err_raw.min():.3f} a k={k_values[np.argmin(err_raw)]}")
print(f"Rumore, pesata:     errore minimo = {err_w.min():.3f} a k={k_best} (oracolo)")
print(f"Rumore, pesata:     k da principio di discrepanza = {k_disc}, errore = {err_w[k_disc-1]:.3f}")
print(f"Rumore, pesata, rango pieno k={k_max}: errore = {err_w[-1]:.3e}")


# STEP 9 - Confronto visivo: A vero vs ricostruzioni pesate (dati con rumore)
k_show = sorted(set([max(1, k_disc // 2), k_disc, min(k_max, 2 * k_disc)]))
iz_show2 = np.argmin(np.abs(z_coords - zp))
ext = (x_coords[0]-step/2, x_coords[-1]+step/2, y_coords[0]-step/2, y_coords[-1]+step/2)

fig, axs = plt.subplots(1, len(k_show)+1, figsize=(4*(len(k_show)+1), 4.2))
axs[0].imshow(A_rep[:, :, iz_show2].T, origin='lower', cmap='viridis',
              vmin=0, vmax=dmu_a, extent=ext)
axs[0].scatter(0, 0, marker='*', c='red', s=100)
axs[0].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', s=30)
axs[0].set_title('$A$ vero'); axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]')

for ax, k in zip(axs[1:], k_show):
    A_hat = tsvd_solve(Uw, sw, Vtw, M_noisy_w, k)
    section = A_hat.reshape(x_coords.size, y_coords.size, z_coords.size)[:, :, iz_show2].T
    vmax_hat = np.abs(A_hat).max()
    im = ax.imshow(section, origin='lower', cmap='RdBu_r', vmin=-vmax_hat, vmax=vmax_hat, extent=ext)
    ax.scatter(0, 0, marker='*', c='k', s=100)
    ax.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=20, alpha=0.5)
    ax.set_title(f'$\\hat A$ pesata, $k$={k}\nerr={err_w[k-1]:.2f}, picco a {loc_w[k-1]:.0f} mm dal vero')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"Ricostruzione TSVD pesata da dati con rumore, $z\\approx${z_coords[iz_show2]:.0f} mm "
             f"(vero: $x_p$={xp:.0f}, $y_p$={yp:.0f}, $z_p$={zp:.0f} mm)")
fig.tight_layout()
plt.show()

# --- localizzazione: voxel di picco (non centroide di |A_hat|: i lobi
# negativi della soluzione a norma minima peserebbero come segnale) ---
print("="*55)
print("LOCALIZZAZIONE (dati con rumore, TSVD pesata): picco vs posizione vera")
print("="*55)
print(f"Vero voxel perturbato:  ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
for k in k_show:
    A_hat = tsvd_solve(Uw, sw, Vtw, M_noisy_w, k)
    peak = r_V[np.argmax(A_hat)]
    pos_mass = np.clip(A_hat, 0, None).sum()
    neg_mass = np.clip(-A_hat, 0, None).sum()
    w_abs = np.abs(A_hat)
    cum = np.cumsum(np.sort(w_abs)[::-1]); cum /= cum[-1]
    n50 = np.searchsorted(cum, 0.5) + 1
    print(f"k={k:3d}:  picco = ({peak[0]:6.2f}, {peak[1]:6.2f}, {peak[2]:6.2f}) mm "
          f"(dist. {loc_w[k-1]:5.1f} mm, non pesata {loc_raw[k-1]:5.1f} mm)   "
          f"| massa neg/pos = {neg_mass/max(pos_mass, 1e-30):.2f}   "
          f"| voxel per il 50% di |A_hat| = {n50}")
print("="*55)

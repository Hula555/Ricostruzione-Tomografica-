# -*- coding: utf-8 -*-
"""
DOT Project - Parte 1 (revisione)
Time-Domain Diffuse Optical Tomography in un MEZZO SEMI-INFINITO
Lorenzo Lazzari, 10532056

Rispetto alla versione originale (mezzo infinito, sorgente e rivelatore
coincidenti in r=0), qui si introducono tre differenze fondamentali:

 1) MEZZO SEMI-INFINITO: l'interfaccia z=0 separa l'aria dal mezzo
    diffondente. A causa del mismatch di indice di rifrazione, la
    fluenza non si annulla sull'interfaccia fisica ma su un piano
    "estrapolato" z = -zb. Sorgente e rivelatori vengono modellati con
    il METODO DELLE IMMAGINI, ma con una profondita' reale diversa fra i
    due: la SORGENTE è inserita ad una profondità z0 (un cammino libero
    medio di trasporto sotto la superficie), l'approssimazione standard
    per una fibra di iniezione il cui fascio collimato diventa
    effettivamente isotropo solo dopo un cammino libero medio nel
    mezzo. I RIVELATORI, invece, raccolgono la luce esattamente
    all'interfaccia fisica (non iniettano un fascio che debba
    "isotropizzarsi"): la loro posizione reale è quindi a z=0. In
    entrambi i casi, ad ogni punto reale si associa una sorgente/punto
    "immagine" di segno opposto, speculare rispetto al piano
    estrapolato z=-zb.

 2) SORGENTE E RIVELATORE NON COINCIDENTI: 8 rivelatori disposti a
    raggiera attorno alla sorgente (4 assi cartesiani + 4 bisettrici),
    a distanza rho = 12 mm.

 3) La sorgente è posta nell'origine (0,0,0); il dominio spaziale (x,y)
    è simmetrico rispetto all'origine (comprende coordinate negative),
    mentre z >= 0 rappresenta la profondità nel mezzo.
"""

# %% =============================================================
# STEP 1 - Proprieta' ottiche del mezzo (bulk)
# ==================================================================
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, pi, sqrt

mu_a0 = 0.01                     # assorbimento di base [mm^-1]
mu_s0 = 1.00                     # scattering ridotto di base [mm^-1]
mu_s  = np.linspace(0.5, 2, 5)         # per lo studio parametrico
mu_a  = np.linspace(0, 0.04, 5)        # per lo studio parametrico
n_idx = 1.4                      # indice di rifrazione relativo (mezzo/aria)

D = 1/(3*mu_s0)                  # coefficiente di diffusione [mm]
v = 300/n_idx                    # velocita' della luce nel mezzo [mm/ns]

Nsample = 1024
t = np.linspace(0.05, 10, Nsample)     # asse dei tempi [ns]


# %% =============================================================
# STEP 2 - Condizione al contorno estrapolata (mezzo semi-infinito)
# ==================================================================
# Per un mezzo semi-infinito con interfaccia piana z=0, a causa del
# mismatch di indice di rifrazione fra mezzo e aria, la fluenza deve
# annullarsi non sull'interfaccia fisica ma su un piano "estrapolato"
# posto a z = -zb. La distanza zb dipende dal coefficiente di
# riflessione efficace R_eff all'interfaccia, che a sua volta dipende
# dall'indice di rifrazione relativo n (formula approssimata di uso
# comune in letteratura DOT, es. Haskell et al. 1994):
#
#   R_eff(n) = -1.4399/n^2 + 0.7099/n + 0.6681 + 0.0636*n
#   A(n)     = (1+R_eff)/(1-R_eff)
#   zb       = 2*A*D

def A_coefficient(n):
    """Coefficiente di mismatch A = (1+Reff)/(1-Reff)."""
    Reff = -1.4399/n**2 + 0.7099/n + 0.6681 + 0.0636*n
    return (1 + Reff)/(1 - Reff)

A_coeff = A_coefficient(n_idx)
z0 = 1/mu_s0                     # profondita' di inserzione della sorgente
                                  # (un cammino libero medio di trasporto)
zb = 2*A_coeff*D                 # distanza del piano estrapolato dall'interfaccia

print("="*55)
print("PARAMETRI DEL MEZZO SEMI-INFINITO")
print("="*55)
print(f"Indice di rifrazione relativo n:            {n_idx}")
print(f"Coefficiente di mismatch A:                 {A_coeff:.3f}")
print(f"Profondita' di inserzione sorgente z0:       {z0:.2f} mm")
print(f"Distanza piano estrapolato zb:               {zb:.2f} mm")
print(f"Piano estrapolato a z = -{zb:.2f} mm")
print("="*55 + "\n")

# Metodo delle immagini per la sorgente: inserita a profondita' z0 sotto
# l'interfaccia, le si associa una sorgente "immagine" di segno opposto,
# speculare rispetto al piano estrapolato z=-zb, cosi' che la fluenza
# totale si annulli esattamente su tale piano. (Per i rivelatori si veda
# lo Step 3: stessa idea, ma a profondita' reale z=0.)

r_s0 = np.array([0.0, 0.0, z0])             # sorgente reale
r_si = np.array([0.0, 0.0, -z0 - 2*zb])     # sorgente immagine


# %% =============================================================
# STEP 3 - Geometria sorgente-rivelatori
# ==================================================================
# Sorgente nell'origine (0,0,0). 8 rivelatori disposti a raggiera
# attorno alla sorgente, a distanza rho_sd, lungo i 4 assi cartesiani
# e le 4 bisettrici (angoli 0, 45, 90, ..., 315 gradi).

rho_sd = 12.0                    # mm, distanza sorgente-rivelatore
angoli = np.arange(0, 360, 45)   # 8 direzioni
det_xy = np.array([[rho_sd*np.cos(np.radians(a)),
                     rho_sd*np.sin(np.radians(a))] for a in angoli])
N_det = det_xy.shape[0]

# A differenza della sorgente, per i rivelatori NON si applica la stessa
# approssimazione "isotropa a profondita' z0": quel trucco approssima la
# fibra di INIEZIONE, il cui fascio collimato diventa effettivamente
# isotropo solo dopo un cammino libero medio di trasporto nel mezzo. Un
# rivelatore, invece, raccoglie la luce esattamente all'interfaccia fisica:
# la sua posizione reale e' quindi a z=0, non a z=z0. L'immagine (per il
# metodo delle immagini) resta speculare rispetto al piano estrapolato
# z=-zb, ma ora della posizione REALE z=0: z_immagine = -2*zb - 0 = -2*zb.
r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))
r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))

plt.rcParams['figure.dpi'] = 150
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(0, 0, marker='*', s=250, c='red', label='Sorgente', zorder=3)
ax.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', s=80, c='royalblue',
           label='Rivelatori', zorder=3)
for i, (x, y) in enumerate(det_xy):
    ax.annotate(f'D{i+1}', (x, y), textcoords='offset points',
                xytext=(6, 6), ha='left', va='bottom', fontsize=9)
circle = plt.Circle((0, 0), rho_sd, fill=False, linestyle='--', color='gray')
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title(f"Geometria sorgente-rivelatori (vista dall'alto)\n"
             f"$\\rho$ = {rho_sd} mm, {N_det} rivelatori")
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
plt.show()


# %% =============================================================
# STEP 4 - Fluenza omogenea: mezzo infinito (funzione base) e
#          mezzo semi-infinito (metodo delle immagini)
# ==================================================================
def fluence_inf(mu_a0, mu_s0, n, t, r):
    """Fluenza in un mezzo infinito omogeneo, sorgente impulsiva in r=0."""
    D = 1/(3*mu_s0)
    v = 300/n
    return (v/(4*pi*D*v*t)**1.5) * exp(-(r**2)/(4*D*v*t) - mu_a0*v*t)

def fluence_semiinf(mu_a0, mu_s0, n, t, r_field, r_sorg, r_imm):
    """Fluenza nel mezzo semi-infinito nel punto r_field, dovuta ad una
    sorgente reale in r_sorg e alla sua immagine in r_imm (sottratta)."""
    r_reale = np.linalg.norm(np.asarray(r_field) - np.asarray(r_sorg))
    r_immag = np.linalg.norm(np.asarray(r_field) - np.asarray(r_imm))
    return fluence_inf(mu_a0, mu_s0, n, t, r_reale) - \
           fluence_inf(mu_a0, mu_s0, n, t, r_immag)

# Rivelatore di riferimento per gli studi parametrici: D1 (asse x, y=0)
rd_ref = r_d0[0]

# --- fluenza al rivelatore D1, al variare di mu_s (mu_a0 fissato) ---
plt.figure()
for i in range(5):
    Di = 1/(3*mu_s[i]); z0_i = 1/mu_s[i]; zb_i = 2*A_coeff*Di
    rs_i = np.array([0, 0, z0_i])
    ri_i = np.array([0, 0, -z0_i - 2*zb_i])
    rd_i = np.array([rho_sd, 0, 0.0])       # rivelatore alla superficie fisica (z=0)
    y = fluence_semiinf(mu_a0, mu_s[i], n_idx, t, rd_i, rs_i, ri_i)
    plt.semilogy(t, y)
plt.title(f"Fluenza ($\\phi$) al rivelatore D1 - mezzo semi-infinito\n"
          f"fixed $\\mu_a$ = {mu_a0} $mm^{{-1}}$, $\\rho$ = {rho_sd} mm")
plt.xlabel('t (ns)'); plt.ylabel('$\\phi$ ($mm^{-2}ns^{-1}$)')
plt.legend(np.round(mu_s, 3), title="$\\mu_s'$ ($mm^{-1}$)")
plt.grid(); plt.show()

# --- fluenza al rivelatore D1, al variare di mu_a (mu_s0 fissato) ---
plt.figure()
for i in range(5):
    y = fluence_semiinf(mu_a[i], mu_s0, n_idx, t, rd_ref, r_s0, r_si)
    plt.semilogy(t, y)
plt.title(f"Fluenza ($\\phi$) al rivelatore D1 - mezzo semi-infinito\n"
          f"fixed $\\mu_s'$ = {mu_s0} $mm^{{-1}}$, $\\rho$ = {rho_sd} mm")
plt.xlabel('t (ns)'); plt.ylabel('$\\phi$ ($mm^{-2}ns^{-1}$)')
plt.legend(mu_a, title='$\\mu_a$ ($mm^{-1}$)')
plt.grid(); plt.show()


# %% =============================================================
# STEP 5 - Contrasto: variazione della fluenza dovuta ad una
#          inclusione assorbente (approssimazione di Born)
# ==================================================================
# Nel mezzo infinito (Parte 1 originale), la perturbazione dovuta ad
# una piccola inclusione dmu_a, volume V, in posizione rp, con
# sorgente e rivelatore a distanza r ed r1 dall'inclusione, e':
#
#   dphi0(r,r1,t) = - v^2/(4 pi D v)^(5/2) / t^(3/2) * dmu_a * V *
#                     (1/r + 1/r1) * exp(-mu_a0 v t - (r+r1)^2/(4 D v t))
#
# Nel mezzo semi-infinito, per linearita' dell'equazione di diffusione,
# la perturbazione totale e' la somma (con segno) dei 4 contributi
# ottenuti combinando sorgente/rivelatore reali e immagine (le
# immagini della sorgente e del rivelatore compaiono ciascuna con un
# segno negativo, cosi' come nel calcolo della fluenza omogenea):
#
#   dphi_tot = dphi0(r_s,r_d) - dphi0(r_s,r_di) - dphi0(r_si,r_d) + dphi0(r_si,r_di)

def deltaphi_inf(mu_a0, mu_s0, n, t, dmu_a, V, r, r1):
    D = 1/(3*mu_s0); v = 300/n
    return -(v**2)/((4*pi*D*v)**2.5 * t**1.5) * exp(-mu_a0*v*t) * \
            dmu_a*V*(1/r + 1/r1)*exp(-(r+r1)**2/(4*D*v*t))

def deltaphi_semiinf(mu_a0, mu_s0, n, t, dmu_a, V, rp, rs, rsi, rd, rdi):
    rp = np.asarray(rp)
    r_s, r_si_ = np.linalg.norm(rp-rs), np.linalg.norm(rp-rsi)
    r_d, r_di_ = np.linalg.norm(rp-rd), np.linalg.norm(rp-rdi)
    t1 = deltaphi_inf(mu_a0, mu_s0, n, t, dmu_a, V, r_s,  r_d)
    t2 = deltaphi_inf(mu_a0, mu_s0, n, t, dmu_a, V, r_s,  r_di_)
    t3 = deltaphi_inf(mu_a0, mu_s0, n, t, dmu_a, V, r_si_, r_d)
    t4 = deltaphi_inf(mu_a0, mu_s0, n, t, dmu_a, V, r_si_, r_di_)
    return t1 - t2 - t3 + t4

dmu_a = 0.01      # variazione di assorbimento dell'inclusione [mm^-1]
V_incl = 1000.0   # volume dell'inclusione [mm^3]

# --- Contrasto in tempo al rivelatore D1, al variare della profondita' zp ---
zp_values = np.linspace(5, 30, 6)
plt.figure()
for zp in zp_values:
    rp = np.array([0, 0, zp])
    dphi = deltaphi_semiinf(mu_a0, mu_s0, n_idx, t, dmu_a, V_incl,
                             rp, r_s0, r_si, rd_ref, r_di[0])
    phi0 = fluence_semiinf(mu_a0, mu_s0, n_idx, t, rd_ref, r_s0, r_si)
    Contrast = np.abs(dphi/phi0)
    plt.plot(t, Contrast)
plt.xlabel('t (ns)'); plt.ylabel('C (-)')
plt.title(f"Contrasto assoluto ($\\delta\\phi_0/\\phi_0$) al rivelatore D1\n"
          f"$\\rho$ = {rho_sd} mm")
plt.legend(np.round(zp_values, 1), title='$z_p$ (mm)', loc=1)
plt.grid(); plt.show()

# --- Contrasto lungo x_p, confrontando TUTTI E 8 i rivelatori (t, zp fissati) ---
# Nota: al contrario dei grafici precedenti (dove l'inclusione era sull'asse
# sorgente-z e, per simmetria, tutti gli 8 rivelatori misurano esattamente lo
# stesso segnale), qui l'inclusione si sposta lungo x_p e "esce" dall'asse
# di simmetria: i rivelatori vicini alla traiettoria (es. D1, allineato con
# x_p) vedono un contrasto maggiore rispetto a quelli opposti o perpendicolari
# (es. D5). E' proprio questa diversita' di risposta fra i rivelatori a
# contenere l'informazione spaziale sfruttata poi nella ricostruzione.
xp_arr = np.linspace(-60, 60, Nsample)
zp_fix = 20.0
t_fix = 6.0
plt.figure()
for d in range(N_det):
    C = np.zeros_like(xp_arr)
    for k, xp in enumerate(xp_arr):
        rp = np.array([xp, 0, zp_fix])
        dphi = deltaphi_semiinf(mu_a0, mu_s0, n_idx, t_fix, dmu_a, V_incl,
                                 rp, r_s0, r_si, r_d0[d], r_di[d])
        phi0 = fluence_semiinf(mu_a0, mu_s0, n_idx, t_fix, r_d0[d], r_s0, r_si)
        C[k] = abs(dphi/phi0)
    plt.plot(xp_arr, C, label=f'D{d+1}')
plt.xlabel('$x_p$ (mm)'); plt.ylabel('C (-)')
plt.title(f"Contrasto lungo $x_p$ per tutti i rivelatori\n"
          f"$z_p$ = {zp_fix} mm, t = {t_fix} ns")
plt.legend(ncol=2, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.grid(); plt.tight_layout(); plt.show()

# --- Mappe 2D del contrasto (x,y), un pannello per rivelatore (zp, t fissati) ---
# Ogni pannello mostra la "regione di sensitivita'" propria di quel rivelatore:
# il picco di contrasto si sposta verso la posizione del rivelatore stesso,
# a conferma che ogni coppia sorgente-rivelatore campiona preferenzialmente
# una diversa porzione del mezzo (base della tomografia).
t_fix = 8.0
zp_fix = 20.0
xg = np.linspace(-40, 40, 41)
yg = np.linspace(-40, 40, 41)
Xg, Yg = np.meshgrid(xg, yg)

fig, axs = plt.subplots(2, 4, figsize=(14, 7))
axs = axs.flatten()
for d in range(N_det):
    C2d = np.zeros_like(Xg)
    for a in range(Xg.shape[0]):
        for b in range(Xg.shape[1]):
            rp = np.array([Xg[a, b], Yg[a, b], zp_fix])
            dphi = deltaphi_semiinf(mu_a0, mu_s0, n_idx, t_fix, dmu_a, V_incl,
                                     rp, r_s0, r_si, r_d0[d], r_di[d])
            phi0 = fluence_semiinf(mu_a0, mu_s0, n_idx, t_fix, r_d0[d], r_s0, r_si)
            C2d[a, b] = abs(dphi/phi0)
    im = axs[d].imshow(C2d, origin='lower', extent=(-40, 40, -40, 40), cmap='inferno')
    axs[d].scatter(0, 0, marker='*', c='cyan')
    axs[d].scatter(det_xy[d, 0], det_xy[d, 1], marker='o', c='lime')
    axs[d].set_title(f'D{d+1}')
    axs[d].set_xlabel('x [mm]')
axs[0].set_ylabel('y [mm]'); axs[4].set_ylabel('y [mm]')
fig.suptitle(f'Contrasto 2D per ciascun rivelatore, $z_p$ = {zp_fix} mm, t = {t_fix} ns')
fig.colorbar(im, ax=axs, shrink=0.8, label='C (-)')
plt.show()

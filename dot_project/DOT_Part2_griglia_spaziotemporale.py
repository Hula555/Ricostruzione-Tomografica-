# -*- coding: utf-8 -*-
"""
DOT Project - Parte 2 (revisione: mezzo semi-infinito)
Discretizzazione spazio-temporale per il problema diretto: griglia di
voxel e finestre temporali (time gates), con la geometria sorgente-
rivelatori del mezzo semi-infinito introdotta nella Parte 1.

Adattato dal notebook originale di Lorenzo Lazzari (10532056). In questa
revisione ci si ferma alla discretizzazione dello spazio e del tempo
(il "framework" del problema diretto): la costruzione della matrice di
sensitivita' W, del vettore dei voxel A e del vettore delle misure M
sono rimandate alla prossima parte.

Nuove ipotesi geometriche rispetto all'originale (si veda la Parte 1
per la derivazione teorica del bordo estrapolato e del metodo delle
immagini):
 1) Mezzo semi-infinito: il piano z=0 e' l'interfaccia fisica aria/mezzo,
    quindi la griglia si estende SOLO per z >= 0.
 2) 8 rivelatori disposti a raggiera (4 quadranti + 4 bisettrici) a
    distanza rho_sd = 12 mm dalla sorgente.
 3) La sorgente e' nell'origine (0,0,0); il dominio in x,y e' simmetrico
    (con coordinate negative), la stessa estensione per entrambi gli assi.
"""

# %% =============================================================
# STEP 1 - Richiamo dei parametri e geometria sorgente-rivelatori
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Riprendiamo da qui le grandezze derivate nella Parte 1 (bordo
# estrapolato zb, profondita' equivalente z0 della sorgente reale,
# posizione della sorgente immagine) e le usiamo per costruire la
# geometria COMPLETA del sistema: non piu' un solo rivelatore
# rappresentativo, ma gli 8 richiesti dalla nuova ipotesi (2).
#
# La disposizione "a raggiera" e' realizzata mettendo un rivelatore ogni
# 45 gradi lungo la circonferenza di raggio rho_sd centrata sulla
# sorgente: gli angoli 0/90/180/270 gradi corrispondono ai 4 assi
# cartesiani (i "confini" fra quadranti), mentre 45/135/225/315 gradi ne
# sono le bisettrici. In formule, per l'angolo a_k = k*45 gradi
# (k=0,...,7):
#
#   x_k = rho_sd cos(a_k) ,   y_k = rho_sd sin(a_k)
#
# Ogni rivelatore viene trattato con il metodo delle immagini della
# Parte 1, ma a differenza della sorgente (dove l'approssimazione
# "isotropa a profondita' z0" modella la fibra di iniezione) il
# rivelatore raccoglie la luce esattamente all'interfaccia: la sua
# posizione reale e' quindi (x_k,y_k,0), quella immagine e' -2zb.

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, pi, sqrt

# --- proprieta' ottiche di base (identiche alla Parte 1) ---
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

r_s0 = np.array([0.0, 0.0, z0])              # sorgente reale
r_si = np.array([0.0, 0.0, -z0 - 2*zb])      # sorgente immagine

# --- geometria degli 8 rivelatori a raggiera ---
rho_sd = 12.0                              # distanza sorgente-rivelatore [mm]
angoli = np.arange(0, 360, 45)             # 8 angoli, passo 45 gradi
det_xy = np.array([[rho_sd*np.cos(np.radians(a)),
                     rho_sd*np.sin(np.radians(a))] for a in angoli])
N_det = det_xy.shape[0]

# A differenza della sorgente (approssimata come isotropa a profondita'
# z0, il trucco standard per una fibra di iniezione), i rivelatori
# raccolgono la luce esattamente all'interfaccia fisica: la loro
# posizione reale e' z=0, con immagine speculare a z=-2*zb.
r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))             # rivelatori reali (z=0)
r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))           # rivelatori immagine

print("="*55)
print("GEOMETRIA SORGENTE-RIVELATORI (mezzo semi-infinito)")
print("="*55)
print(f"Sorgente: origine (0,0,0); sorgente reale equivalente a z0={z0:.2f} mm")
print(f"Rivelatori: posizione reale a z=0 (superficie fisica)")
print(f"Bordo estrapolato a z = {-zb:.2f} mm (A={A_coeff:.3f})")
print(f"N. rivelatori: {N_det}, a raggiera, rho_sd = {rho_sd:.0f} mm")
for k, (xy, ang) in enumerate(zip(det_xy, angoli)):
    print(f"  D{k+1}: angolo={ang:3d} deg -> (x,y) = ({xy[0]:+6.2f}, {xy[1]:+6.2f}) mm")
print("="*55 + "\n")

# --- vista dall'alto: sorgente e rivelatori sul piano dell'interfaccia ---
plt.rcParams['figure.dpi'] = 150
plt.figure(figsize=(5, 5))
plt.scatter(0, 0, marker='*', s=250, c='red', label='Sorgente (origine)')
plt.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', s=80, c='royalblue',
            label='Rivelatori (reali)')
for k, xy in enumerate(det_xy):
    plt.annotate(f'D{k+1}', xy, xytext=(3, 3), textcoords='offset points', fontsize=8)
circ = plt.Circle((0, 0), rho_sd, fill=False, linestyle='--', color='gray')
plt.gca().add_patch(circ)
plt.gca().set_aspect('equal')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f"Geometria sorgente-rivelatori (vista dall'alto)\n"
          f"8 rivelatori a raggiera, $\\rho_{{sd}}$ = {rho_sd:.0f} mm")
plt.grid(); plt.legend(loc='upper right', fontsize=8)
plt.show()

# COMMENTO AL GRAFICO: i rivelatori D1,D3,D5,D7 (angoli 0/90/180/270)
# giacciono sugli assi cartesiani (i "confini" fra quadranti), mentre
# D2,D4,D6,D8 (45/135/225/315) ne sono le bisettrici: la disposizione
# copre quindi in modo uniforme le 8 direzioni, permettendo di
# campionare la perturbazione indotta da un'inclusione indipendentemente
# dalla sua direzione azimutale rispetto alla sorgente. Il cerchio
# tratteggiato evidenzia che tutti i rivelatori sono equidistanti dalla
# sorgente (stessa rho_sd), a differenza di un'ipotetica disposizione
# irregolare.


# %% =============================================================
# STEP 2 - Griglia spaziale (discretizzazione in voxel)
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# A differenza del mezzo infinito (dove il dominio da discretizzare
# sarebbe simmetrico anche in z), qui il piano z=0 e' l'interfaccia
# fisica aria/mezzo (ipotesi 3): lo spazio dei voxel si estende quindi
# SOLO per z >= 0. In x e y, essendo la sorgente nell'origine e i
# rivelatori disposti simmetricamente intorno ad essa, il dominio e'
# anch'esso simmetrico rispetto all'origine e comprende coordinate
# negative, con la STESSA estensione lungo x e y (ipotesi 3).
#
# NOTA DI PROGRAMMAZIONE: i voxel sono cubi di lato "step"; le coordinate
# generate con np.arange partono da mezzo passo dal bordo del dominio
# (es. -x_ext/2 + step/2), in modo che ogni coordinata rappresenti il
# CENTRO del voxel, non il suo spigolo - convenzione comoda per valutare
# le formule analitiche (che assumono un punto, non un volume) al centro
# di ciascuna cella. np.meshgrid con indexing='ij' produce tre array 3D
# (X,Y,Z) della stessa forma, che affiancati e "appiattiti" con
# column_stack/flatten danno la lista (N_vox x 3) delle coordinate di
# ogni voxel: r_V[m] = (x,y,z) del voxel m-esimo. Questo mapping
# (indice lineare m <-> voxel 3D) e' quello che verra' riusato, nella
# prossima parte, per costruire per colonne la matrice W e il vettore A.

step = 4.0                        # lato del voxel [mm]
x_ext = 64.0                      # estensione totale del dominio in x,y [mm]
z_ext = 64.0                      # estensione massima in profondita' [mm]

x_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
y_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
z_coords = np.arange(0 + step/2, z_ext, step)     # SOLO z >= 0 (ipotesi 3)

X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
r_V = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
N_vox = r_V.shape[0]
V_vox = step**3

print("="*55)
print("GRIGLIA SPAZIALE (VOXEL)")
print("="*55)
print(f"Step voxel: {step} mm | Volume voxel: {V_vox} mm^3")
print(f"Estensione x,y: [{x_coords[0]-step/2:.0f}, {x_coords[-1]+step/2:.0f}] mm "
      f"({x_coords.size} voxel/lato)")
print(f"Estensione z:   [0, {z_coords[-1]+step/2:.0f}] mm ({z_coords.size} voxel/lato)")
print(f"Numero totale di voxel N_vox = {N_vox}")
print("="*55 + "\n")

# --- vista dall'alto: griglia + geometria sorgente-rivelatori ---
plt.figure(figsize=(5, 5))
plt.scatter(X[:, :, 0].flatten(), Y[:, :, 0].flatten(), s=4, c='lightgray',
            label='Voxel (proiezione xy)')
plt.scatter(0, 0, marker='*', s=250, c='red', label='Sorgente')
plt.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', s=80, c='royalblue',
            label='Rivelatori')
plt.gca().set_aspect('equal')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title('Griglia spaziale e geometria sorgente-rivelatori (vista dall\'alto)')
plt.grid(); plt.legend(loc='upper right', fontsize=8)
plt.show()

# --- vista laterale (piano xz, sezione a y più vicino a 0): mostra il
#     vincolo z >= 0 imposto dall'interfaccia fisica ---
iy0 = np.argmin(np.abs(y_coords))
plt.figure(figsize=(6, 4))
plt.scatter(X[:, iy0, :].flatten(), Z[:, iy0, :].flatten(), s=4, c='lightgray',
            label='Voxel (sezione $y\\approx$%.0f mm)' % y_coords[iy0])
plt.scatter(0, 0, marker='*', s=200, c='red', label='Sorgente (interfaccia)')
plt.axhline(0, color='k', linewidth=1.2, label='Interfaccia aria/mezzo (z=0)')
plt.axhline(-zb, color='gray', linestyle='--', linewidth=1, label='Bordo estrapolato')
plt.gca().invert_yaxis()
plt.xlabel('x [mm]'); plt.ylabel('z [mm]')
plt.title('Sezione laterale della griglia: vincolo z $\\geq$ 0')
plt.grid(); plt.legend(loc='lower right', fontsize=7)
plt.show()

# COMMENTO AI GRAFICI: la vista dall'alto e' analoga a quella dello
# Step 1, con in piu' la nuvola di voxel proiettata sul piano xy: si
# vede come il dominio copra ampiamente la zona coperta dai rivelatori
# (rho_sd = 12 mm), lasciando margine per rilevare inclusioni anche
# fuori dal cerchio sorgente-rivelatori. La sezione laterale rende
# esplicita la differenza principale rispetto a un mezzo infinito: i
# voxel esistono SOLO al di sotto dell'interfaccia (z >= 0); il bordo
# estrapolato (tratteggiato), che si trova a z = -zb, cade FUORI dal
# dominio fisico dei voxel, come deve essere: e' un artificio
# matematico del metodo delle immagini (Parte 1), non una regione dello
# spazio reale.


# %% =============================================================
# STEP 3 - Discretizzazione temporale (finestre / time gates)
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Una misura reale di time-resolved DOT non campiona la DTOF
# (distribution of times of flight) a tempo continuo, ma la integra in
# un numero finito di finestre temporali (time gates), ciascuna di
# ampiezza tipica ~1 ns:
#
#   M_gate = int_{t_g0}^{t_g1} phi(t) dt
#
# Discretizziamo quindi l'asse dei tempi con passo dt (che deve essere
# fine abbastanza da risolvere la salita della DTOF, tipicamente
# dell'ordine di decine di ps) e definiamo N_gates finestre contigue,
# qui di ampiezza fissa 1 ns, distanziate anch'esse di 1 ns.
#
# NOTA DI PROGRAMMAZIONE: gate_indices converte i confini "fisici" (in
# ns) di ciascuna finestra negli indici interi dell'array t corrispondenti,
# tramite np.searchsorted: cosi' l'integrazione (che verra' fatta nella
# prossima parte con la regola dei trapezi) potra' operare direttamente
# per slicing su t[i0:i1], senza dover ricalcolare il confine ogni volta.

dt = 0.05                          # passo di campionamento [ns]
t = np.arange(0.1, 10.0, dt)       # asse dei tempi [ns]
N_gates = 8
gates = [(1.0 + i, 2.0 + i) for i in range(N_gates)]     # bordi delle finestre [ns]
gate_indices = [(np.searchsorted(t, g[0]), np.searchsorted(t, g[1])) for g in gates]

print("="*55)
print("FRAMEWORK TEMPORALE")
print("="*55)
print(f"Passo di campionamento dt = {dt} ns | Numero di campioni: {t.size}")
print(f"Numero di finestre temporali (time gates): {N_gates}")
for k, (g0, g1) in enumerate(gates):
    print(f"  Gate {k}: {g0:.1f} - {g1:.1f} ns")
print("="*55 + "\n")

# --- fluenza omogenea semi-infinita al rivelatore D1, con i gate sovrapposti ---
def fluence_inf(mu_a0, mu_s0, n, t, r):
    D = 1/(3*mu_s0); v = 3e2/n
    return (v/(4*pi*D*v*t)**(3/2))*exp(-(r**2/(4*D*v*t)) - mu_a0*v*t)

d_show = 0
r_d_s  = np.linalg.norm(r_d0[d_show] - r_s0)
r_d_si = np.linalg.norm(r_d0[d_show] - r_si)
phi0_demo = fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s) - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(t, phi0_demo)
axs[0].set_xlim(0, 2.5)
axs[0].set_xlabel('t (ns)'); axs[0].set_ylabel(r'$\phi_{semi-inf}$')
axs[0].set_title('Zoom lineare vicino al picco'); axs[0].grid(alpha=0.3)
axs[1].semilogy(t, phi0_demo)
axs[1].set_xlabel('t (ns)'); axs[1].set_ylabel(r'$\phi_{semi-inf}$')
axs[1].set_title('Scala log, intero range'); axs[1].grid(alpha=0.3)
for (g0, g1) in gates:
    axs[0].axvspan(g0, g1, color='gray', alpha=0.15)
    axs[1].axvspan(g0, g1, color='gray', alpha=0.15)
fig.suptitle(f'DTOF omogenea al rivelatore D1 ($\\rho_{{sd}}$={rho_sd:.0f} mm) '
             'e finestre temporali (bande grigie)')
plt.tight_layout()
plt.show()

# COMMENTO AL GRAFICO: nel pannello lineare si vede il picco della DTOF,
# raggiunto entro il primo ns, seguito dalla coda a decadimento
# esponenziale mostrata in scala log nel pannello destro. I gate scelti
# (1-9 ns, bande grigie) cadono interamente nella coda, ben oltre il
# picco: e' li' che la forma della curva e' dominata dall'assorbimento
# (mu_a), rendendo i gate tardivi piu' sensibili a piccole variazioni di
# mu_a (il segnale utile per la Parte 3) rispetto ai gate precoci, che
# sono invece piu' sensibili a mu_s' e alla geometria vicino alla
# sorgente.

print(f"Dimensione del problema diretto: {N_det} rivelatori x {N_gates} time-gates = "
      f"{N_det*N_gates} misurazioni indipendenti, per {N_vox} incognite (voxel).")
print("\nQuesto completa il framework spazio-temporale (griglia di voxel, geometria")
print("sorgente-rivelatori, finestre temporali). Il file DOT_Part2_W_A_M_senza_rumore.py")
print("prosegue da qui costruendo la matrice di sensitivita' W, il vettore dei voxel A")
print("e il vettore delle misure M = W . A per il problema diretto, senza rumore.")

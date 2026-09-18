# -*- coding: utf-8 -*-
"""
DOT Project - Parte 2 (prosecuzione): matrice W, vettore voxel A,
vettore delle misure M - problema diretto, senza rumore, mezzo
semi-infinito.

Prosegue da DOT_Part1_mezzo_semiinfinito.py (fluenza e contrasto per
una coppia sorgente-rivelatore) e da
DOT_Part2_griglia_spaziotemporale.py (griglia di voxel, geometria a 8
rivelatori, finestre temporali). Per comodita' le grandezze di quei due
file vengono ridefinite qui sotto, cosi' che questo file sia eseguibile
da solo.

Obiettivo: costruire la matrice di sensitivita' (Jacobiano) W, tale che
per una qualunque distribuzione di perturbazioni di assorbimento nello
spazio dei voxel, rappresentata dal vettore A (dmu_a in ogni voxel), il
vettore delle misure (contrasto, per rivelatore e per time-gate) sia

    M = W . A

Qui W e A sono costruiti in modo che questa relazione sia ESATTA nel
limite dell'approssimazione di Born al prim'ordine (perturbazione
piccola, gia' usata in Parte 1): non c'e' rumore di alcun tipo, ne'
altre approssimazioni aggiuntive (es. IRF strumentale), che si possono
introdurre in un secondo momento sopra a questa stessa struttura.
"""

# %% =============================================================
# STEP 1 - Richiamo delle grandezze di Parte 1 e della griglia
# ==================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp, pi, sqrt

# --- proprieta' ottiche di base e bordo estrapolato (Parte 1, Step 3) ---
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

# --- geometria a 8 rivelatori a raggiera (Parte 2, Step 1) ---
rho_sd = 12.0
angoli = np.arange(0, 360, 45)
det_xy = np.array([[rho_sd*np.cos(np.radians(a)),
                     rho_sd*np.sin(np.radians(a))] for a in angoli])
N_det = det_xy.shape[0]
r_d0 = np.column_stack((det_xy, np.full(N_det, z0)))
r_di = np.column_stack((det_xy, np.full(N_det, -z0 - 2*zb)))

# --- griglia spaziale, z>=0 (Parte 2, Step 2) ---
step = 4.0
x_ext = 64.0
z_ext = 64.0
x_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
y_coords = np.arange(-x_ext/2 + step/2, x_ext/2, step)
z_coords = np.arange(0 + step/2, z_ext, step)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
r_V = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
N_vox = r_V.shape[0]
V_vox = step**3

# --- discretizzazione temporale, time gates (Parte 2, Step 3) ---
dt = 0.05
t = np.arange(0.1, 10.0, dt)
N_gates = 8
gates = [(1.0 + i, 2.0 + i) for i in range(N_gates)]
gate_indices = [(np.searchsorted(t, g[0]), np.searchsorted(t, g[1])) for g in gates]

print("="*55)
print("RICHIAMO: N_det =", N_det, "| N_vox =", N_vox, "| N_gates =", N_gates)
print(f"Misurazioni indipendenti: {N_det*N_gates}  |  Incognite (voxel): {N_vox}")
print("="*55 + "\n")


# %% =============================================================
# STEP 2 - Vettore dei voxel A: mappatura di un'inclusione nello
#          spazio discretizzato
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Il vettore A e' la rappresentazione discreta della perturbazione
# fisica: A[m] = delta_mu_a nel voxel m-esimo (0 dove il mezzo e'
# omogeneo). Per un'inclusione sferica di volume assegnato V_incl,
# centrata in (xp,yp,zp), il raggio equivalente e'
#
#   r_eq = (3 V_incl / 4 pi)^(1/3)
#
# La regola di mappatura (identica, concettualmente, a quella usata nel
# progetto originale) e': un voxel appartiene all'inclusione se il suo
# CENTRO ricade dentro la sfera, |r_voxel - r_incl| < r_eq. Il volume
# della sfera non e' in generale un multiplo esatto di V_vox, quindi il
# "volume effettivo" discretizzato,
#
#   V_eff = V_vox * (numero di voxel selezionati)
#
# approssima V_incl con un errore dell'ordine di un voxel di superficie
# della sfera - inevitabile quando si passa da una geometria continua a
# una griglia discreta, e tanto piu' piccolo quanto piu' fine e' step.
#
# NOTA DI PROGRAMMAZIONE: `mask` e' un array booleano 3D (stessa forma
# di X,Y,Z) che vale True nei voxel dentro la sfera; moltiplicato per
# dmu_a da' la rappresentazione 3D "A_rep" (utile per i grafici),
# mentre A_rep.flatten() da' il vettore 1D A nello stesso ordine lineare
# di r_V (fondamentale: A[m] deve riferirsi allo stesso voxel di
# r_V[m], altrimenti "M = W @ A" mescolerebbe le informazioni).

dmu_a = 0.01           # variazione di assorbimento dell'inclusione [mm^-1]
V_incl = 1000.0        # volume dell'inclusione [mm^3]
r_eq = (V_incl*3/(4*pi))**(1/3)

xp, yp, zp = 15.0, 10.0, 20.0     # posizione dell'inclusione [mm]

mask = (X - xp)**2 + (Y - yp)**2 + (Z - zp)**2 < r_eq**2
A_rep = mask*dmu_a                # rappresentazione 3D di A (per i grafici)
A = A_rep.flatten()               # vettore voxel 1D (stesso ordine di r_V)
V_eff = V_vox*int(np.sum(mask))

print("="*55)
print("VETTORE VOXEL A (inclusione sferica)")
print("="*55)
print(f"Posizione: xp={xp} mm, yp={yp} mm, zp={zp} mm | V_incl={V_incl} mm^3 "
      f"-> r_eq={r_eq:.2f} mm")
print(f"Voxel selezionati: {int(np.sum(mask))} | Volume effettivo: {V_eff} mm^3")
print(f"delta_mu_a nell'inclusione: {dmu_a} mm^-1")
print("="*55 + "\n")

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
             f"Step {step} mm, volume effettivo {V_eff} mm$^3$")
plt.show()

# --- sezione 2D nel piano z piu' vicino a zp, con sorgente e rivelatori ---
iz = np.argmin(np.abs(z_coords - zp))
plt.figure()
im = plt.imshow(A_rep[:, :, iz].T, origin='lower',
                 extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                         y_coords[0]-step/2, y_coords[-1]+step/2),
                 vmin=0, vmax=dmu_a)
plt.scatter(0, 0, marker='*', c='red', s=120, label='Sorgente')
plt.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', label='Rivelatori')
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f"Sezione del vettore A nello spazio voxel\n"
          f"z $\\approx$ {z_coords[iz]:.0f} mm, step {step} mm")
plt.legend(fontsize=8)
plt.colorbar(im, label=r'$\delta\mu_a$ [$mm^{-1}$]')
plt.grid(); plt.show()

# COMMENTO AI GRAFICI: la vista 3D mostra il "grumo" di voxel attivati
# dall'inclusione (asse z invertito per leggere la profondita' verso il
# basso, come nella sezione laterale di Parte 2). La sezione 2D, presa
# al piano z piu' vicino a zp=20mm, mostra l'inclusione (xp=15,yp=10) in
# posizione ASIMMETRICA rispetto alla sorgente (stella, nell'origine) e
# ai rivelatori (cerchi, sul cerchio di raggio 12mm): a differenza del
# progetto originale (dove l'inclusione era spesso posta sull'asse
# sorgente-rivelatore per simmetria), qui la geometria a 8 rivelatori
# rende ogni posizione "vista" in modo diverso da ciascun rivelatore -
# proprio l'informazione che permettera', nella Parte 3, di localizzarla
# per triangolazione.


# %% =============================================================
# STEP 3 - Costruzione della matrice di sensitivita' (Jacobiano) W
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Generalizzando la formula del contrasto di Parte 1 (Step 4) a un
# generico voxel m e rivelatore d, la sensitivita' della misura al
# time-gate g e':
#
#   W[dg,m] = ( int_{gate g} dphi_semiinf(voxel m, rivelatore d, t) dt ) /
#             ( int_{gate g} phi_semiinf(rivelatore d, t) dt )
#
# dove dphi_semiinf e' la somma a 4 immagini gia' derivata in Parte 1,
# qui valutata per il volume di riferimento V_vox e delta_mu_a UNITARIO
# (dmu_a=1): siccome dphi_semiinf e' lineare in delta_mu_a (Born al
# prim'ordine), W non dipende dal valore di A - e' una proprieta' del
# solo sistema (geometria + ottica), esattamente come un Jacobiano.
# Questo e' cio' che rende vera, per costruzione, la relazione
#
#   M = W . A
#
# qualunque sia la distribuzione A (somma di piu' inclusioni, inclusioni
# di forma qualsiasi, ecc.), non solo per la singola inclusione sferica
# dello Step 2: la linearita' del prim'ordine di Born si eredita
# direttamente nella struttura MATRICE x VETTORE.
#
# NOTA DI PROGRAMMAZIONE (vettorizzazione): per ogni rivelatore d,
# calcoliamo le 4 distanze voxel-sorgente(i) e voxel-rivelatore(i) come
# array di lunghezza N_vox in un colpo solo (np.linalg.norm con axis=1),
# poi valutiamo dphi_semiinf per TUTTI i voxel e TUTTI i tempi
# contemporaneamente, sfruttando il broadcasting: passando le distanze
# come colonna (N_vox,1) e i tempi come riga (1,N_t) si ottiene una
# matrice (N_vox, N_t) in una sola chiamata, invece di un ciclo Python
# su ciascun voxel (che con N_vox=4096 sarebbe migliaia di volte piu'
# lento). L'integrazione nei gate resta poi un semplice np.trapz per
# slicing sull'asse dei tempi.

def deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r, r1, n=n_idx):
    """Perturbazione di fluenza (Born, I ordine) in mezzo infinito, fra
    un punto sorgente(-immagine) a distanza r dal voxel e un punto
    rivelatore(-immagine) a distanza r1 dal voxel. Supporta il
    broadcasting di numpy: r, r1, t possono essere scalari o array de
    forme compatibili (qui: r,r1 colonna (N_vox,1), t riga (1,N_t))."""
    D = 1/(3*mu_s0)
    v = 3e2/n
    return (-(v**2)/((4*pi*D*v)**(5/2)*t**(3/2))*exp(-mu_a0*v*t)
            * dmu_a*V*(1/r + 1/r1)*exp(-(r+r1)**2/(4*D*v*t)))

def fluence_inf(mu_a0, mu_s0, n, t, r):
    D = 1/(3*mu_s0); v = 3e2/n
    return (v/(4*pi*D*v*t)**(3/2))*exp(-(r**2/(4*D*v*t)) - mu_a0*v*t)

# integrazione trapezoidale: np.trapz e' stato rinominato np.trapezoid
# nelle versioni piu' recenti di numpy (>=2.0) ed e' stato rimosso dal
# vecchio nome; questo fallback rende il codice eseguibile con entrambe
# le versioni, senza dover fissare una versione minima di numpy.
_trapz = getattr(np, "trapezoid", None) or np.trapz

r_sv  = np.linalg.norm(r_V - r_s0, axis=1)     # sorgente reale -> ogni voxel
r_siv = np.linalg.norm(r_V - r_si, axis=1)     # sorgente immagine -> ogni voxel

W = np.zeros((N_det*N_gates, N_vox))
tt = t[None, :]                                # riga (1, N_t), per il broadcasting

print("="*55)
print(f"Costruzione della matrice W: {N_det*N_gates} misurazioni x {N_vox} voxel")
print("="*55)

for d in range(N_det):
    r_vd  = np.linalg.norm(r_V - r_d0[d], axis=1)   # voxel -> rivelatore reale
    r_vdi = np.linalg.norm(r_V - r_di[d], axis=1)   # voxel -> rivelatore immagine

    # somma a 4 immagini (Parte 1, Step 4), per tutti i voxel e tutti i
    # tempi in un colpo solo: shape (N_vox, N_t)
    dphi_vt = (deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vd[:, None])
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_sv[:, None],  r_vdi[:, None])
             - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vd[:, None])
             + deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, r_siv[:, None], r_vdi[:, None]))

    # fluenza omogenea (senza perturbazioni) al rivelatore d, mezzo
    # semi-infinito: shape (N_t,)
    r_d_s  = np.linalg.norm(r_d0[d] - r_s0)
    r_d_si = np.linalg.norm(r_d0[d] - r_si)
    phi0_t = fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s) - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si)

    for g, (i0, i1) in enumerate(gate_indices):
        int_dphi = _trapz(dphi_vt[:, i0:i1], dx=dt, axis=1)   # (N_vox,)
        int_phi0 = _trapz(phi0_t[i0:i1], dx=dt)                # scalare
        W[d*N_gates + g, :] = int_dphi / int_phi0

print(f"Matrice W costruita: shape = {W.shape}\n")

# --- sezioni 2D di W per il rivelatore D1, a diverse profondita' e gate ---
# NOTA: usiamo un vmin/vmax CONDIVISO fra tutti i pannelli (non lasciamo
# che imshow normalizzi ogni pannello sul proprio min/max locale):
# altrimenti, anche se |W| crolla di ordini di grandezza fra un pannello
# e l'altro, ogni pannello verrebbe "stirato" sull'intera scala di
# colori e il crollo di sensitivita' con la profondita' - l'informazione
# fisica piu' importante di questo grafico - resterebbe nascosto.
W_3d = W.reshape(N_det, N_gates, x_coords.size, y_coords.size, z_coords.size)
d_show = 0
z_show = [10.0, 20.0, 30.0]
g_show = [0, 3, 7]
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
                                 f"({gates[g][0]:.0f}-{gates[g][1]:.0f} ns)", fontsize=9, pad=8)
        if c_i == 0: axs[r_i, c_i].set_ylabel('y [mm]')
        if r_i == 2: axs[r_i, c_i].set_xlabel('x [mm]')
fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, right=0.86)
fig.suptitle(r'Sezioni di $\log_{10}|W|$ nello spazio voxel - rivelatore D1'
             '\n(scala colore condivisa fra tutti i pannelli)')
fig.colorbar(im, ax=axs, shrink=0.7, label=r'$\log_{10}|W|$')
plt.show()

# --- confronto fra tutti gli 8 rivelatori, a profondita' e gate fissati ---
z_fix = 20.0
g_fix = 5
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
             f"({gates[g_fix][0]:.0f}-{gates[g_fix][1]:.0f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()

# COMMENTO AI GRAFICI: nella prima figura si vede, per il rivelatore D1,
# come la regione di alta sensitivita' sia una "banana" stretta fra
# sorgente e rivelatore al gate precoce (colonna di sinistra), tanto
# piu' confinata in superficie quanto piu' e' bassa la profondita' z
# considerata (il pannello z=30mm/gate0 e' quasi ovunque scuro: a
# profondita' elevate, quasi nessun fotone e' ancora arrivato al
# rivelatore in una finestra cosi' precoce). Ai gate tardivi (colonna
# di destra) la sensitivita' non resta confinata alla banana, ma si
# ESPANDE fino a occupare gran parte del dominio, ed e' significativa
# anche a profondita' notevoli: e' l'effetto, gia' incontrato nella
# Parte 2 (Step 3) per la DTOF omogenea, per cui a tempi lunghi il
# "cono" di diffusione dei fotoni si e' allargato enormemente - ed e'
# esattamente il motivo per cui, in DOT time-resolved, i gate tardivi
# sono quelli piu' informativi per rivelare inclusioni profonde
# (nonostante il rapporto segnale/rumore piu' sfavorevole, non incluso
# in questo modello ideale). La seconda figura mostra che tutti gli 8
# rivelatori, ruotati di 45 gradi l'uno rispetto all'altro, hanno la
# STESSA forma di sensitivita' orientata secondo la propria direzione
# azimutale: conseguenza diretta della simmetria di rotazione della
# geometria a raggiera, che garantisce copertura uniforme di tutte le
# direzioni intorno alla sorgente.


# %% =============================================================
# STEP 4 - Vettore delle misure M = W . A (problema diretto, senza rumore)
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Con W (Step 3) e A (Step 2) costruiti, il problema diretto e' un
# semplice prodotto matrice-vettore:
#
#   M = W . A          M in R^(N_det*N_gates),  A in R^(N_vox)
#
# M[dg] e' il contrasto (adimensionale) previsto al rivelatore d, nel
# time-gate g, per la distribuzione di assorbimento A. Non essendoci
# alcuna sorgente di rumore aggiunta (ne' rumore di conteggio poissoniano
# sui fotoni, ne' rumore strumentale), M e' la misura "ideale": la
# useremo, nella Parte 3, come termine noto del problema inverso, prima
# di eventualmente aggiungere rumore realistico sopra di essa.
#
# NOTA DI PROGRAMMAZIONE: `.reshape(N_det, N_gates)` sfrutta il fatto che
# le righe di W sono state riempite nell'ordine d*N_gates+g (Step 3):
# la stessa convenzione di indicizzazione va sempre rispettata quando si
# passa da un indice "piatto" (per l'algebra lineare) a due indici
# "fisici" (rivelatore, gate) per l'interpretazione e i grafici.

M_ideal = (W @ A).reshape(N_det, N_gates)

print("="*55)
print("VETTORE DELLE MISURE M = W . A (ideale, senza rumore)")
print("="*55)
for d in range(N_det):
    print(f"  D{d+1}: " + " ".join(f"{val:+.4f}" for val in M_ideal[d, :]))
print("="*55 + "\n")

plt.figure(figsize=(6, 5))
im = plt.imshow(M_ideal, aspect='auto', cmap='RdBu_r',
                 vmin=-np.abs(M_ideal).max(), vmax=np.abs(M_ideal).max())
plt.yticks(range(N_det), [f'D{d+1}' for d in range(N_det)])
plt.xticks(range(N_gates), [f'{g[0]:.0f}-{g[1]:.0f}' for g in gates], rotation=45)
plt.xlabel('time gate [ns]'); plt.ylabel('rivelatore')
plt.title('Vettore delle misure M = W . A (ideale, mezzo semi-infinito)')
plt.colorbar(im, label='C (-)')
plt.tight_layout(); plt.show()

# COMMENTO AL GRAFICO: i rivelatori piu' vicini alla proiezione xy
# dell'inclusione (xp=15,yp=10, quindi principalmente D2 a 45 gradi,
# ma anche D1 e D3) mostrano il contrasto maggiore in valore assoluto,
# mentre i rivelatori diametralmente opposti (D5, D6) ne vedono uno
# quasi nullo: e' esattamente il pattern di asimmetria azimutale
# atteso per un'inclusione fuori dall'asse. Il segno negativo di M
# (una diminuzione di fluenza, non un aumento) riflette la fisica di
# un'inclusione ASSORBENTE: piu' fotoni vengono assorbiti dal voxel
# perturbato, meno ne arrivano al rivelatore. Lungo i gate, |M| cresce
# poi lentamente con il tempo (i gate tardivi, piu' vicini alla coda
# della DTOF, integrano un contributo via via maggiore dei fotoni che
# hanno interagito con l'inclusione) - proprieta' che verra' sfruttata
# nella Parte 3 per vincolare meglio il problema inverso.

print("Problema diretto completato: M = W . A, senza rumore.")
print("La prossima parte (Parte 3) affrontera' l'inversione del problema")
print("(pseudo-inversa/regolarizzazione di W) per ricostruire A da M, con")
print("ed eventualmente senza rumore aggiunto sulla misura.")

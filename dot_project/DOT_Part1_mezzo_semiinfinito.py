# -*- coding: utf-8 -*-
"""
DOT Project - Parte 1 (revisione: mezzo semi-infinito)
Fluenza e contrasto per sorgente impulsiva in un mezzo torbido
semi-infinito, con condizione al contorno estrapolata (metodo delle
immagini).

Adattato dal notebook originale "DOT Project - Part 1" di
Lorenzo Lazzari (10532056), che trattava il caso di mezzo INFINITO con
sorgente e rivelatore coincidenti (r = 0).

Cosa cambia rispetto all'originale (nuove ipotesi):
 1) Il mezzo non e' piu' infinito ma SEMI-INFINITO: esiste un'interfaccia
    fisica piana z = 0 fra l'aria (indice di rifrazione 1) e il mezzo
    diffondente (indice n). Per rispettare la condizione al contorno
    all'interfaccia si usa il metodo delle immagini con BORDO
    ESTRAPOLATO (extrapolated boundary condition), ricavato
    esplicitamente nello Step 3.
 2) Sorgente e rivelatore NON sono piu' coincidenti: sono separati da una
    distanza rho_sd sul piano dell'interfaccia (nella Parte 2, rho_sd =
    12 mm, con 8 rivelatori disposti a raggiera). Di conseguenza la
    fluenza va valutata a r != 0, e nella formula del contrasto compare
    la geometria completa sorgente-voxel-rivelatore.
 3) La sorgente e' nell'origine (0,0,0), sul piano fisico z = 0.

Per il resto la struttura ricalca fedelmente l'originale: fluenza
omogenea, poi contrasto in tempo, in spazio, e mappe 2D.
"""

# %% =============================================================
# STEP 1 - Import e parametri fisici / geometrici
# ==================================================================
# Nessuna differenza logica rispetto all'originale in questo blocco:
# importiamo le stesse librerie (numpy per il calcolo vettoriale,
# matplotlib per i grafici) e definiamo le stesse proprieta' ottiche di
# base. Le due aggiunte sono: l'indice di rifrazione n = 1.4 (tessuto
# biologico tipico) al posto di n = 1 (che renderebbe il mezzo
# otticamente indistinguibile dall'aria, senza riflessione interna
# all'interfaccia: e' proprio questo mismatch a rendere necessaria, allo
# Step 3, la condizione al contorno estrapolata); e la distanza
# sorgente-rivelatore rho_sd, non piu' nulla.

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, pi, sqrt

# proprieta' ottiche di base (bulk) del mezzo
mu_a0 = 0.01              # assorbimento di base [mm^-1]
mu_s0 = 1.00              # scattering ridotto di base [mm^-1]
mu_s = np.linspace(0.5, 2, 5)      # per lo studio parametrico
mu_a = np.linspace(0, 0.04, 5)     # per lo studio parametrico
n_idx = 1.4               # indice di rifrazione relativo mezzo/aria

Nsample = 1024
t = np.linspace(0.05, 10, Nsample)   # asse dei tempi [ns]

# geometria sorgente-rivelatore: nella Parte 2 il sistema avra' 8
# rivelatori disposti a raggiera; qui, per la teoria "a coppia singola",
# ne consideriamo uno rappresentativo, alla stessa distanza dalla
# sorgente che verra' poi usata per tutti gli 8 nella Parte 2
rho_sd = 12.0             # distanza sorgente-rivelatore [mm]


# %% =============================================================
# STEP 2 - Fluenza in un mezzo INFINITO omogeneo (richiamo)
# ==================================================================
# Richiamiamo la soluzione dell'equazione di diffusione per una sorgente
# impulsiva isotropa in un mezzo infinito e omogeneo: e' il "mattone" di
# base con cui costruiremo, nella prossima sezione, la soluzione per il
# mezzo semi-infinito (metodo delle immagini: una soluzione in mezzo
# semi-infinito si costruisce SEMPRE come somma/differenza di soluzioni
# in mezzo infinito).
#
#   phi_inf(r,t) = v / (4 pi D v t)^(3/2) * exp( -r^2/(4 D v t) - mu_a v t )
#
# con D = 1/(3 mu_s') coefficiente di diffusione, v = c0/n velocita'
# della luce nel mezzo, r distanza fra sorgente e punto di campo. A
# differenza dell'originale NON possiamo piu' porre r = 0 (sorgente e
# rivelatore coincidenti): con rho_sd = 12 mm la formula generale va
# mantenuta per intero, e va valutata a r = rho_sd.

def fluence_inf(mu_a0, mu_s0, n, t, r):
    """Fluenza in mezzo infinito omogeneo, sorgente impulsiva in r=0.
    mu_a0, mu_s0: assorbimento/scattering ridotto [mm^-1]; n: indice di
    rifrazione; t: array dei tempi [ns]; r: distanza sorgente-campo [mm]."""
    D = 1/(3*mu_s0)
    v = 3e2/n                      # velocita' della luce nel mezzo [mm/ns]
    return (v/(4*pi*D*v*t)**(3/2))*exp(-(r**2/(4*D*v*t)) - mu_a0*v*t)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = [5, 5]

# --- variazione di mu_s' a mu_a fissato ---
plt.figure()
for i in range(5):
    y = fluence_inf(mu_a0, mu_s[i], n_idx, t, rho_sd)
    plt.semilogy(t, y)
plt.title(r"Fluenza in mezzo infinito ($\phi_\infty$) - $\mu_a$ = %.2f $mm^{-1}$, "
          r"$\rho_{sd}$ = %.0f mm" % (mu_a0, rho_sd))
plt.xlabel('t (ns)'); plt.ylabel(r'$\phi_\infty$ ($mm^{-2}ns^{-1}$)')
plt.legend(mu_s, title=r"$\mu_s'$ ($mm^{-1}$)")
plt.grid(); plt.show()

# --- variazione di mu_a a mu_s' fissato ---
plt.figure()
for i in range(5):
    z = fluence_inf(mu_a[i], mu_s0, n_idx, t, rho_sd)
    plt.semilogy(t, z)
plt.title(r"Fluenza in mezzo infinito ($\phi_\infty$) - $\mu_s'$ = %.2f $mm^{-1}$, "
          r"$\rho_{sd}$ = %.0f mm" % (mu_s0, rho_sd))
plt.xlabel('t (ns)'); plt.ylabel(r'$\phi_\infty$ ($mm^{-2}ns^{-1}$)')
plt.legend(mu_a, title=r"$\mu_a$ ($mm^{-1}$)")
plt.grid(); plt.show()

# COMMENTO AI GRAFICI: la forma delle due famiglie di curve e' identica
# a quella dell'originale (il mezzo e', in questa sezione, ancora
# infinito): mu_a governa la pendenza del decadimento esponenziale nella
# coda (in scala log e' una retta di pendenza -mu_a v), mu_s' governa
# soprattutto l'istante e l'ampiezza del picco (attraverso D=1/(3mu_s')),
# con un effetto molto piu' debole sulla coda a lungo tempo. L'unica
# differenza numerica rispetto all'originale e' che ora r = rho_sd =
# 12 mm invece di r = 0: le curve partono da un valore molto piu' basso
# e mostrano una salita iniziale (il fotone impiega un tempo finito per
# diffondere dalla sorgente al rivelatore), invece di partire gia' dal
# picco per t->0 come nel caso r = 0.


# %% =============================================================
# STEP 3 - Mezzo SEMI-INFINITO: bordo estrapolato e metodo delle immagini
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# Il mezzo occupa ora solo il semispazio z >= 0; il piano z = 0 e'
# l'interfaccia fisica con l'aria (indice 1). L'equazione di diffusione
# resta la stessa risolta allo Step 2, ma va risolta rispettando una
# condizione al contorno che tenga conto della riflessione interna
# parziale del "fotone diffuso" dovuta al mismatch di indice di
# rifrazione mezzo/aria (n != 1).
#
# L'approccio standard (Haskell et al. 1994; Patterson et al. 1989) e'
# la PARTIAL-CURRENT BOUNDARY CONDITION, che si approssima imponendo
# fluenza nulla non sul piano fisico z=0, ma su un piano esteso
# (extrapolated boundary) posto a z = -zb, con
#
#   zb = 2 A D ,       A = (1 + Reff)/(1 - Reff)
#
# dove Reff e' il coefficiente di riflessione interna efficace (frazione
# di fotoni diffusi retroriflessi all'interfaccia invece di uscire),
# funzione del solo indice di rifrazione relativo tramite il fit
# empirico di Egan/Groenhuis:
#
#   Reff(n) = -1.4399/n^2 + 0.7099/n + 0.6681 + 0.0636 n
#
# Per n=1 (nessun mismatch) si ha Reff=0, A=1, zb=2D: il caso limite
# "trasparente". Per n=1.4 (tessuto tipico) i fotoni sono parzialmente
# intrappolati (Reff>0, A>1): il bordo estrapolato si allontana
# dall'interfaccia fisica, cioe' il mezzo "sembra" un po' piu' esteso di
# quanto sia in realta'.
#
# La condizione phi(z=-zb)=0 si impone SENZA risolvere una nuova
# equazione, sovrapponendo alla sorgente reale (positiva) una sorgente
# "immagine" NEGATIVA, speculare rispetto al piano z=-zb. Siccome
# l'equazione di diffusione e' lineare, phi_inf(r-r_s) resta soluzione
# anche traslando/cambiando segno alla sorgente, e la LORO DIFFERENZA si
# annulla identicamente sul piano equidistante dalle due sorgenti (il
# bordo esteso), per costruzione:
#
#   phi_semiinf(r,t) = phi_inf(|r - r_s0|,t) - phi_inf(|r - r_si|,t)
#
# La sorgente reale non e' messa esattamente sull'interfaccia (dove la
# formula puntiforme diverge e non e' fisicamente accurata vicino
# all'ingresso del fascio), ma "reiniettata" isotropicamente a un
# cammino libero di trasporto di profondita' sotto la superficie:
#
#   z0 = 1/mu_s'                   (sorgente reale, in (0,0,z0))
#   r_si = (0,0,-z0-2 zb)          (sorgente immagine, speculare a -zb)
#
# Per simmetria dello stesso principio, la grandezza misurata (nella
# convenzione semplificata di questo progetto, la fluenza stessa, non
# il flusso uscente che sarebbe piu' rigoroso ma piu' oneroso da
# derivare) viene valutata nel punto "coniugato" al rivelatore reale,
# posto anch'esso alla profondita' z0 sotto la sua posizione (x_d,y_d)
# sull'interfaccia: si veda la Parte 2 per la geometria completa a 8
# rivelatori.

def A_coefficient(n):
    """Coefficiente A(n) del bordo estrapolato, da Reff(n) (fit di
    Egan/Groenhuis per l'indice di rifrazione relativo mezzo/aria)."""
    Reff = -1.4399/n**2 + 0.7099/n + 0.6681 + 0.0636*n
    return Reff, (1 + Reff)/(1 - Reff)

D = 1/(3*mu_s0)
v = 3e2/n_idx
Reff, A_coeff = A_coefficient(n_idx)
z0 = 1/mu_s0                      # profondita' sorgente reale [mm]
zb = 2*A_coeff*D                  # distanza bordo estrapolato - interfaccia [mm]

r_s0 = np.array([0.0, 0.0, z0])              # sorgente reale
r_si = np.array([0.0, 0.0, -z0 - 2*zb])      # sorgente immagine

print("="*55)
print("BORDO ESTRAPOLATO (metodo delle immagini)")
print("="*55)
print(f"Indice di rifrazione relativo n = {n_idx}")
print(f"Reff(n) = {Reff:.4f}")
print(f"A(n)    = {A_coeff:.4f}")
print(f"z0 (profondita' sorgente reale)      = {z0:.3f} mm")
print(f"zb (distanza bordo esteso - z=0)     = {zb:.3f} mm")
print(f"Sorgente reale    r_s0 = {r_s0} mm")
print(f"Sorgente immagine r_si = {r_si} mm")
print("="*55 + "\n")

# Rivelatore rappresentativo, sul piano z=0 a distanza rho_sd dalla
# sorgente (posto per convenzione lungo l'asse x): la sua posizione
# "attiva" e quella "immagine" seguono la stessa logica della sorgente.
r_d0 = np.array([rho_sd, 0.0, z0])
r_di = np.array([rho_sd, 0.0, -z0 - 2*zb])

def fluence_semiinf(r, rs, ri, mu_a0, mu_s0, n, t):
    """Fluenza in mezzo semi-infinito nel punto r, dovuta alla coppia
    sorgente reale (rs) + sorgente immagine (ri): differenza di due
    fluenze in mezzo infinito (Step 2), per costruzione."""
    dist_s = np.linalg.norm(r - rs)
    dist_i = np.linalg.norm(r - ri)
    return (fluence_inf(mu_a0, mu_s0, n, t, dist_s)
            - fluence_inf(mu_a0, mu_s0, n, t, dist_i))

phi_inf_only = fluence_inf(mu_a0, mu_s0, n_idx, t, np.linalg.norm(r_d0 - r_s0))
phi_semiinf = fluence_semiinf(r_d0, r_s0, r_si, mu_a0, mu_s0, n_idx, t)

plt.figure()
plt.semilogy(t, phi_inf_only, '--', label='solo sorgente reale (no bordo)')
plt.semilogy(t, phi_semiinf, label=r'$\phi_{semi-inf}$ = reale $-$ immagine')
plt.xlabel('t (ns)'); plt.ylabel(r'$\phi$ ($mm^{-2}ns^{-1}$)')
plt.title(f"Effetto del bordo estrapolato sulla fluenza al rivelatore\n"
          f"($\\rho_{{sd}}$ = {rho_sd:.0f} mm, n = {n_idx})")
plt.legend(); plt.grid(); plt.show()

# COMMENTO AL GRAFICO: le due curve coincidono quasi perfettamente per
# tempi brevi/intermedi (il fotone non ha ancora "sentito" il bordo) e
# si separano progressivamente: la sorgente immagine (negativa) sottrae
# fluenza, cioe' rende conto della perdita netta di fotoni attraverso
# l'interfaccia aria/mezzo (che nel mezzo infinito non esisterebbe). La
# correzione cresce con il tempo perche' a tempi lunghi il "cono" di
# diffusione dei fotoni si allarga e interessa sempre di piu' anche la
# regione vicina al bordo esteso.

# --- ripetiamo lo studio parametrico dello Step 2 con la formula corretta ---
# NOTA DI PROGRAMMAZIONE: z0, zb e quindi anche r_s0/r_si dipendono da
# mu_s' (attraverso D=1/(3 mu_s') e z0=1/mu_s'): al variare di mu_s' non
# possiamo piu' riusare r_s0/r_si fissi come allo Step 2, ma vanno
# ricalcolati per ogni valore di mu_s' del ciclo.
plt.figure()
for mus_i in mu_s:
    D_i, z0_i = 1/(3*mus_i), 1/mus_i
    zb_i = 2*A_coeff*D_i
    r_s0_i = np.array([0, 0, z0_i])
    r_si_i = np.array([0, 0, -z0_i - 2*zb_i])
    r_d0_i = np.array([rho_sd, 0, z0_i])
    phi = fluence_semiinf(r_d0_i, r_s0_i, r_si_i, mu_a0, mus_i, n_idx, t)
    plt.semilogy(t, phi)
plt.title(r"Fluenza in mezzo semi-infinito ($\phi_{semi-inf}$) - "
          r"$\mu_a$ = %.2f $mm^{-1}$" % mu_a0)
plt.xlabel('t (ns)'); plt.ylabel(r'$\phi_{semi-inf}$ ($mm^{-2}ns^{-1}$)')
plt.legend(mu_s, title=r"$\mu_s'$ ($mm^{-1}$)")
plt.grid(); plt.show()

# mu_a NON compare in z0/zb/r_s0/r_si (che dipendono solo da mu_s'):
# possiamo quindi riusare la stessa geometria (r_s0, r_si, r_d0) fissata
# sopra e variare solo l'esponenziale di assorbimento, come nello Step 2.
plt.figure()
for mua_i in mu_a:
    phi = fluence_semiinf(r_d0, r_s0, r_si, mua_i, mu_s0, n_idx, t)
    plt.semilogy(t, phi)
plt.title(r"Fluenza in mezzo semi-infinito ($\phi_{semi-inf}$) - "
          r"$\mu_s'$ = %.2f $mm^{-1}$" % mu_s0)
plt.xlabel('t (ns)'); plt.ylabel(r'$\phi_{semi-inf}$ ($mm^{-2}ns^{-1}$)')
plt.legend(mu_a, title=r"$\mu_a$ ($mm^{-1}$)")
plt.grid(); plt.show()

# COMMENTO AI GRAFICI: la dipendenza qualitativa da mu_a e mu_s' e' la
# stessa osservata nel mezzo infinito (Step 2): mu_a fissa la pendenza
# della coda, mu_s' l'ampiezza/il ritardo del picco. La differenza
# concettuale e' che, qui, al variare di mu_s' cambia anche la geometria
# stessa del sistema di immagini (z0, zb): la curva non si sposta
# soltanto per l'effetto "diretto" di D nella formula, ma anche perche'
# la sorgente immagine si sposta.


# %% =============================================================
# STEP 4 - Contrasto in tempo (mezzo semi-infinito, formula a 4 immagini)
# ==================================================================
# GIUSTIFICAZIONE TEORICA
# -----------------------
# La perturbazione di fluenza dovuta a una piccola inclusione assorbente
# delta_mu_a, di volume V, in posizione (xp,yp,zp), nel mezzo INFINITO
# (approssimazione di Born al prim'ordine, cammino sorgente->voxel->
# rivelatore) e':
#
#   dphi_inf(r,r1,t) = -v^2/(4 pi D v)^(5/2)/t^(3/2) * exp(-mu_a0 v t) *
#                       delta_mu_a * V * (1/r + 1/r1) * exp(-(r+r1)^2/(4 D v t))
#
# con r = |voxel - sorgente|, r1 = |voxel - rivelatore| (questa e' la
# stessa funzione "deltafluence" dell'originale, che qui rinominiamo
# deltafluence_inf per chiarezza).
#
# Nel mezzo SEMI-INFINITO, sia la sorgente sia il rivelatore vanno
# sostituiti dalla loro coppia (reale + immagine): il termine di
# perturbazione, essendo bilineare in "propagatore sorgente->voxel" e
# "propagatore voxel->rivelatore", si sviluppa in QUATTRO termini (uno
# per ciascuna combinazione reale/immagine di sorgente e rivelatore):
#
#   dphi_semiinf = dphi_inf(r_sv,  r_vd)  - dphi_inf(r_sv,  r_vdi)
#                - dphi_inf(r_siv, r_vd)  + dphi_inf(r_siv, r_vdi)
#
# dove r_sv = |voxel - sorgente reale|, r_siv = |voxel - sorgente
# immagine|, r_vd = |voxel - rivelatore reale|, r_vdi = |voxel -
# rivelatore immagine|. I segni alternati derivano dal fatto che ciascun
# propagatore "reale meno immagine" (Step 3) compare due volte nel
# prodotto sorgente-rivelatore, e i segni si moltiplicano:
# (+,-) x (+,-) = (+,-,-,+).
#
# Il contrasto resta definito come nell'originale, C = delta_phi_0/phi_0,
# ma ora sia il numeratore sia il denominatore vanno calcolati con le
# formule del mezzo semi-infinito.

def deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r, r1, n=n_idx):
    """Perturbazione di fluenza (Born, I ordine) in mezzo infinito,
    per un cammino sorgente(o immagine)->voxel->rivelatore(o immagine)."""
    D = 1/(3*mu_s0)
    v = 3e2/n
    return (-v**2/((4*pi*D*v)**(5/2)*t**(3/2))*exp(-mu_a0*v*t)
            * dmu_a*V*(1/r + 1/r1)*exp(-(r+r1)**2/(4*D*v*t)))

def deltafluence_semiinf(mu_a0, mu_s0, t, dmu_a, V, xp, yp, zp, r_s0, r_si, r_d0, r_di, n=n_idx):
    """Perturbazione di fluenza in mezzo semi-infinito: somma a 4 termini
    (reale/immagine per sorgente x reale/immagine per rivelatore).
    xp, yp, zp possono essere scalari o array (es. meshgrid) grazie al
    broadcasting di numpy: le distanze sono calcolate componente per
    componente, senza impacchettare (xp,yp,zp) in un np.array unico
    (che fallirebbe se le tre componenti hanno forme diverse)."""
    r_sv  = sqrt((xp-r_s0[0])**2 + (yp-r_s0[1])**2 + (zp-r_s0[2])**2)
    r_siv = sqrt((xp-r_si[0])**2 + (yp-r_si[1])**2 + (zp-r_si[2])**2)
    r_vd  = sqrt((xp-r_d0[0])**2 + (yp-r_d0[1])**2 + (zp-r_d0[2])**2)
    r_vdi = sqrt((xp-r_di[0])**2 + (yp-r_di[1])**2 + (zp-r_di[2])**2)
    return (deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r_sv,  r_vd,  n)
            - deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r_sv,  r_vdi, n)
            - deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r_siv, r_vd,  n)
            + deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r_siv, r_vdi, n))

V = 1000.0        # volume dell'inclusione [mm^3]
dmu_a = 0.01      # variazione di assorbimento dell'inclusione [mm^-1]
xp, yp = 0.0, 0.0   # inclusione sotto l'asse sorgente-rivelatore (yp=0),
                    # a xp = 0: nel mezzo semi-infinito, a differenza
                    # dell'originale (dove xp=yp=0 e zp coincideva con la
                    # distanza sorgente-inclusione), zp resta la profondita'
                    # ma la distanza sorgente-inclusione e rivelatore-
                    # inclusione ora differiscono fra loro (rho_sd != 0).

zp_values = np.linspace(5, 30, 6)
plt.figure()
for zp_i in zp_values:
    dphi = deltafluence_semiinf(mu_a0, mu_s0, t, dmu_a, V, xp, yp, zp_i,
                                 r_s0, r_si, r_d0, r_di)
    phi0 = fluence_semiinf(r_d0, r_s0, r_si, mu_a0, mu_s0, n_idx, t)
    Contrast = np.abs(dphi/phi0)
    plt.plot(t, Contrast)
plt.xlabel('t (ns)'); plt.ylabel('C (-)')
plt.grid()
plt.title(r"Contrasto assoluto ($\delta\phi_0/\phi_0$) in tempo - mezzo semi-infinito")
plt.legend([f"{z:.0f}" for z in zp_values], title=r'$z_p$ (mm)', loc=1)
plt.show()

# COMMENTO AL GRAFICO: come nell'originale, il contrasto e' tanto piu'
# grande (e satura piu' rapidamente) quanto piu' l'inclusione e'
# superficiale: un'inclusione vicina alla sorgente/rivelatore intercetta
# una frazione maggiore dei fotoni diffusi. La presenza del bordo
# estrapolato non cambia questa tendenza qualitativa (e' un effetto del
# cammino sorgente-voxel-rivelatore, non della condizione al contorno),
# ma modifica leggermente i valori assoluti rispetto al mezzo infinito,
# soprattutto per inclusioni profonde (dove il peso relativo della
# sorgente immagine, piu' lontana, e' piu' rilevante).


# %% =============================================================
# STEP 5 - Contrasto in spazio (lungo xp)
# ==================================================================
# Stessa analisi dell'originale (variazione lungo xp, a zp e yp fissati,
# normalizzata al valore centrale xp=0), ma con la formula a 4 immagini
# dello Step 4 e rivelatore non coincidente con la sorgente.

xp_line = np.linspace(-60, 60, Nsample)
yp_line = 0.0
zp_line = 20.0
t_snapshots = np.linspace(1, 10, 5)

plt.figure()
for tt in t_snapshots:
    dphi = np.array([deltafluence_semiinf(mu_a0, mu_s0, tt, dmu_a, V, xi, yp_line,
                     zp_line, r_s0, r_si, r_d0, r_di) for xi in xp_line])
    phi0 = fluence_semiinf(r_d0, r_s0, r_si, mu_a0, mu_s0, n_idx, np.array([tt]))[0]
    dphi0_center = deltafluence_semiinf(mu_a0, mu_s0, tt, dmu_a, V, 0.0, yp_line,
                                         zp_line, r_s0, r_si, r_d0, r_di)
    Contrast = np.abs(dphi/phi0) / np.abs(dphi0_center/phi0)
    plt.plot(xp_line, Contrast)
plt.xlabel(r'$x_p$ (mm)'); plt.ylabel('C (-)')
plt.grid()
plt.title(r"Contrasto assoluto normalizzato lungo $x_p$ - mezzo semi-infinito")
plt.legend([f"{tt:.2f}" for tt in t_snapshots], title='istanti (ns)', loc=1)
plt.show()

# COMMENTO AL GRAFICO: il profilo resta di forma gaussiana come
# nell'originale, con larghezza crescente nel tempo (i fotoni "sondano"
# lateralmente una regione via via piu' ampia). A differenza
# dell'originale, il profilo non e' piu' perfettamente simmetrico
# intorno a xp = 0 quando rho_sd != 0: qui, con l'inclusione centrata a
# xp = 0 (punto medio fra sorgente all'origine e rivelatore a xp=rho_sd),
# la asimmetria residua e' piccola ma presente, ed e' dovuta alla
# posizione non simmetrica rispetto al segmento sorgente-rivelatore.


# %% =============================================================
# STEP 6 - Mappe 2D del contrasto (xp,yp)
# ==================================================================
# Come nell'originale: due famiglie di mappe, la prima a tempo fissato e
# profondita' variabile, la seconda a profondita' fissata e tempo
# variabile. In ciascuna mappa mostriamo anche la sorgente (stella) e il
# rivelatore rappresentativo (cerchio), assenti nell'originale perche'
# li' erano sovrapposti nell'origine.

yp_grid = np.linspace(-60, 60, 200)
xp_grid = np.linspace(-60, 60, 200)
Xp, Yp = np.meshgrid(xp_grid, yp_grid)

def contrast_map(t_val, zp_val):
    dphi = deltafluence_semiinf(mu_a0, mu_s0, t_val, dmu_a, V, Xp, Yp, zp_val,
                                 r_s0, r_si, r_d0, r_di)
    phi0 = fluence_semiinf(r_d0, r_s0, r_si, mu_a0, mu_s0, n_idx, np.array([t_val]))[0]
    return np.abs(dphi/phi0)

# --- t fissato, zp variabile ---
t_fix = 10.0
zp_panels = np.linspace(5, 30, 6)
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
fig.suptitle(f'Contrasto assoluto, t = {t_fix:.0f} ns - mezzo semi-infinito')
vmax_1 = max(contrast_map(t_fix, z).max() for z in zp_panels)
for i, zp_i in enumerate(zp_panels):
    C = contrast_map(t_fix, zp_i)
    ax = axs[i//3, i % 3]
    im = ax.imshow(C, interpolation='none', origin='lower',
                    extent=(-60, 60, -60, 60), vmin=0, vmax=vmax_1)
    ax.scatter(0, 0, marker='*', c='red', s=60)
    ax.scatter(rho_sd, 0, marker='o', c='cyan', s=25)
    ax.set(title='$z_p$ = %i mm' % zp_i)
fig.colorbar(im, ax=axs.ravel().tolist())
fig.text(0.5, 0.04, r'$x_p$ (mm)', ha='center')
fig.text(0.04, 0.5, r'$y_p$ (mm)', va='center', rotation='vertical')
plt.show()

# --- zp fissato, t variabile ---
zp_fix = 20.0
t_panels = np.linspace(1, 10, 6)
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
fig.suptitle(f'Contrasto assoluto normalizzato, $z_p$ = {zp_fix} mm - mezzo semi-infinito')
maps = [contrast_map(tt, zp_fix) for tt in t_panels]
norms = [np.abs(deltafluence_semiinf(mu_a0, mu_s0, tt, dmu_a, V, 0.0, 0.0, zp_fix,
                r_s0, r_si, r_d0, r_di)
                / fluence_semiinf(r_d0, r_s0, r_si, mu_a0, mu_s0, n_idx, np.array([tt]))[0])
         for tt in t_panels]
for i, (tt, C, nrm) in enumerate(zip(t_panels, maps, norms)):
    ax = axs[i//3, i % 3]
    im = ax.imshow(C/nrm, interpolation='none', origin='lower',
                    extent=(-60, 60, -60, 60), vmin=0, vmax=1)
    ax.scatter(0, 0, marker='*', c='red', s=60)
    ax.scatter(rho_sd, 0, marker='o', c='cyan', s=25)
    ax.set(title='t = %.2f ns' % tt)
fig.colorbar(im, ax=axs.ravel().tolist())
fig.text(0.5, 0.04, r'$x_p$ (mm)', ha='center')
fig.text(0.04, 0.5, r'$y_p$ (mm)', va='center', rotation='vertical')
plt.show()

# COMMENTO AI GRAFICI: rispetto all'originale (dove le "macchie" di
# contrasto erano centrate simmetricamente sull'origine, dato che
# sorgente e rivelatore coincidevano), qui la macchia di massima
# sensitivita' e' spostata verso il segmento che congiunge sorgente
# (stella rossa, nell'origine) e rivelatore (cerchio ciano, a
# xp=rho_sd): e' proprio questa regione, la piu' "banana-shaped" nella
# letteratura DOT, quella in cui il cammino sorgente-voxel-rivelatore e'
# statisticamente piu' probabile, e dove quindi la sensitivita' a una
# perturbazione e' massima. Questo effetto era invisibile nell'originale
# a causa della geometria degenere (rho_sd = 0).

print("\nParte 1 completata: fluenza e contrasto nel mezzo semi-infinito,")
print("per una singola coppia sorgente-rivelatore. La Parte 2 introduce la")
print("geometria completa a 8 rivelatori e la discretizzazione spazio-")
print("temporale (griglia di voxel, finestre temporali).")

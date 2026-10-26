import os
import struct
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi, exp
from scipy.fft import rfft, irfft, next_fast_len
from scipy import ndimage
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

# CONFRONTO FRA LE VERSIONI MIGLIORI DI QUATTRO METODI DI RICOSTRUZIONE, CON IRF MISURATA:
#   TSVD, Tikhonov, L1 (FISTA) e L1 + variazione totale (L1+TV, ADMM),
# applicati agli STESSI dati simulati (stessa misura rumorosa per tutti i metodi).
#
# Problema diretto, rumore, gate, pesi e calibrazione sono quelli dello scan L1/L1+TV:
# fluenza (x) IRF, fondo stimato e sottratto, vincolo a 0, rumore simula_misura (con il
# rumore del fondo), N_tot nei primi N_gates_normalizzazione gate.
#
# Sistema di unita': lunghezze in mm, tempi in ns, coefficienti ottici in
# mm^-1, velocita' in mm/ns (c = 3e2 mm/ns), concentrazioni in uM.
#
# ASSE DEI TEMPI COMUNE: tutto si calcola sulla griglia dei canali della IRF
# (4096 canali x 3.05 ps): fluenza phi(t'), con t' = 0 al canale 0, IRF sul suo
# asse di canali, convoluzione DTOF[n] = dc sum_m phi[m] IRF[n-m] (l'indice n e'
# il canale dello strumento), fondo in conteggi per canale, gate in canali.
#
# PROBLEMA DIRETTO DELLA MISURA (per ogni coppia, curva non perturbata e
# perturbata, in fotoconteggi per canale):
#   1) fluenza (x) IRF NETTA (senza fondo, area 1) + fondo STAZIONARIO: il fondo della
#      IRF misurata (b_h per canale, rispetto all'area netta) e' costante nel tempo e
#      proporzionale ai fotoni rivelati, quindi nella DTOF vale b_h int phi dt su TUTTI
#      i canali. (Convolvere phi con la IRF grezza, come nelle versioni precedenti,
#      trasformava il fondo in una rampa b_h sum_{m<=n} phi[m] che parte da t' = 0: nella
#      finestra del fondo i fotoni tardivi, soprattutto quelli perturbati dai voxel
#      profondi, non erano ancora arrivati e restava un offset spurio nei gate.)
#   2) fondo STIMATO prima del picco: media dei conteggi per canale della curva
#      nella finestra fondo_finestra_ns che finisce irf_fondo_prima_ns prima del
#      picco della IRF
#   3) fondo SOTTRATTO canale per canale; canali <= 0 posti a 0 (vincolo), gate
#      senza conteggi esclusi (riga di W nulla, contrasto 0)
#   4) conteggi attesi per gate = somma dei canali del gate -> lam0 e M_ideal
#   5) rumore: simula_misura (Poisson su conteggi attesi non perturbati e perturbati;
#      con rumore_fondo = True anche il fondo nei gate e la sua stima)
#   M_ideal = W_irf @ A con W_irf costruita sulle curve convolute e senza fondo.
#
# INVERSIONE: con W_irf (inversione_con_irf = True), la stessa matrice della
# misura ("inverse crime": caso ottimistico); inversione_con_irf = False usa la W
# senza IRF. misura_con_irf = False rigenera il caso senza IRF.
#
# COMPENSAZIONE DI PROFONDITA' (comune ai quattro metodi): incognita x_j = p_j A_j,
#     p_j = max(||w_j||, 1/dmu_a_max)^gamma,   B = W_w diag(1/p),   A_j = x_j / p_j
# (w_j = colonna j di W_w = W / sigma). Senza compensazione la regolarizzazione
# preferisce i voxel superficiali (piu' sensibili).
#
# I QUATTRO METODI (versione migliore di ciascuno, dagli scan precedenti):
#   TSVD      gamma = 0.5; k per discrepanza: il piu' piccolo k con chi^2 <= N_validi.
#             Soglia della regione ADATTIVA: 0.25 se z_picco <= 10 mm, 0.30 oltre
#             (scan del 13-14/10).
#   Tikhonov  gamma = 0.6; min ||W_w A - m_w||^2 + lam^2 ||diag(p) A||^2, lam per
#             discrepanza (chi^2 = N_validi, bisezione in log(lam)). Soglia della
#             regione ADATTIVA 0.22 se z_picco <= 10 mm, 0.28 oltre.
#   L1        gamma = 1.0; min_x 1/2 ||B x - m_w||^2 + lam ||x||_1 (FISTA, passo
#             1/||B||^2); lam per discrepanza, chi^2 = N_validi + 2 sqrt(2 N_validi), con
#             griglia + bisezioni. Soglia della regione 0.20.
#   L1+TV     alpha = 0.95 (ADMM):
#                 min_x 1/2 ||W_w x - m_w||^2 + lam [alpha sum_j p_j |x_j|
#                                                     + (1-alpha) sum_e q_e |(D x)_e|]
#             x = dmu_a FISICO; D = differenze fra voxel adiacenti in x, y, z (5632
#             spigoli); p_j = max(||w_j||, 1/dmu_a_max) (gamma = 1), q_e = sqrt(p_i p_k).
#             L* = 6 (1-alpha)/alpha = 0.32 voxel. ADMM con u = copia di x (L1) e z = copia
#             di D x (TV), penalita' rho_u = c p^2, rho_z = c q^2, Minv = (W_w^T W_w +
#             diag(rho_u) + D^T diag(rho_z) D)^-1 una volta per W_w. lam come la L1.
#             Soglia della regione 0.20.
#
# STIMA DELL'INTENSITA' (mappa stima_per_metodo, una scelta per metodo). Regione
# R (il "cluster") = voxel connessi (26-vicinato) al picco di p_j |A_j| con p_j |A_j| >=
# soglia del picco e stesso segno (z_picco_comp = z di quel picco).
#   "fit"       (default) il metodo di ricostruzione serve SOLO a trovare il cluster R;
#               poi si torna ai dati e si stima una sola dmu_a, uniforme su R, ai minimi
#               quadrati pesati:
#                   a = argmin ||a w_R - m_w||^2 = w_R^T m_w / w_R^T w_R,
#                   w_R = sum_{j in R} w_j   (risposta dei dati a dmu_a = 1 su tutto R)
#                   stima = a N_R V_vox                          [mm^-1 mm^3]
#               Ipotesi: dmu_a omogenea nel cluster e cluster corretto. Toglie lo
#               shrinkage della regolarizzazione (L1, L1+TV), ma dipende dalla forma di R.
#   "integrale" somma dei valori ricostruiti nei voxel del cluster, nessun fit:
#                   stima = V_vox sum_{j in R} A_hat_j           [mm^-1 mm^3]
#   "debias"    debiasing di SCALA: si tiene la forma ricostruita A_hat_R e si stima dai
#               dati solo un fattore beta >= 0 (1 grado di liberta', nessun problema di
#               collinearita'):
#                   beta = (W_w,R A_hat_R)^T m_w / ||W_w,R A_hat_R||^2
#                   stima = beta V_vox sum_{j in R} A_hat_j
#               Toglie lo shrinkage di L1 e L1+TV senza imporre dmu_a uniforme.
# Le tre stime sono comunque salvate in ogni riga (colonne stima_fit_grezza,
# stima_integrale_grezza, stima_debias_grezza, dmu_fit_regione = a, beta_debias);
# rapporto_stima usa quella scelta.
#
# COSA SI CONFRONTA NEL RAPPORTO stima/vero: l'assorbimento COMPLESSIVO (integrale di
# dmu_a sul volume), non il valore di un singolo voxel.
#   - Simulazione: dmu_a(lambda) e' il valore di OGNI voxel dell'inclusione (A = mask *
#     dmu): gli 8 voxel hanno tutti dmu_a = dmu, non dmu/8. Il vero e' quindi
#         vero = dmu * N_voxel_incl * V_vox = dmu * 8 * 64 mm^3
#   - Ricostruzione: la sfocatura distribuisce l'assorbimento su un cluster che di solito
#     ha piu' (TSVD/Tikhonov, decine di voxel) o meno (L1) voxel dell'inclusione, con
#     valori per voxel piu' bassi o piu' alti del vero; l'integrale e' la grandezza che
#     la sfocatura conserva, mentre il valore per voxel dipende dalla dimensione del
#     cluster (col fit: a ~ 8 dmu / N_R se il cluster ha N_R voxel invece di 8).
#     rapporto_stima = 1 significa: assorbimento complessivo stimato = 8 dmu V_vox.
#   Il valore medio per voxel equivalente (stima / (N_voxel_incl V_vox), da confrontare
#   con dmu) ha lo stesso rapporto ed e' nella colonna 'dmu_medio_equivalente'.
#
# "TROVATA": segnale della regione in deviazioni standard del rumore,
# |w_R^T m_w| / ||w_R|| (filtro adattato), >= soglia calibrata su misure di solo rumore
# (quantile 0.99, 1% di falsi positivi), comunque >= soglia_minima_sigma. Per la L1+TV
# la soglia non scende sotto quella della L1 (sul solo rumore la L1+TV resta quasi
# sempre vuota).
#
# MODELLO DEL RUMORE NEI PESI (modello_sigma): "baseline" (sigma_0 dai conteggi
# attesi) o "dati" (sigma dai conteggi osservati, metodo delta).
#
# NB: le soglie di TSVD e Tikhonov sono state scelte con N_tot = 1e8, coppie
# 'primi_secondi_vicini' e 20 gate; qui valgono la configurazione dello scan L1/L1+TV
# (N_tot = 1e7, 'primi_secondi_terzi_vicini', 25 gate, rumore del fondo).

# %% =============================================================
# STEP 0 - Spettri
# ==================================================================
file_spettri = r"C:\Users\marti\Downloads\specific_absorption.xlsx"
foglio_spettri = "new"                     # NON usare "mod": colonna Lipid disallineata
# File Excel di output (riscritto a ogni lambda). Un percorso relativo come
# "risultati_scan.xlsx" finisce nella cartella di lavoro corrente di Python
# (os.getcwd()), che in Spyder, VS Code o Jupyter spesso NON e' la cartella dello
# script. Per evitarlo il file va nella stessa cartella del file degli spettri,
# con data e ora nel nome (nessuna sovrascrittura dei risultati precedenti).
cartella_risultati = os.path.dirname(os.path.abspath(file_spettri))
# Stima dell'intensita' (vedi intestazione): "fit" o "integrale"
# Stima dell'intensita' PER METODO (vedi intestazione): "fit", "integrale" o "debias".
# Configurazione mista: integrale puro per i metodi lineari (TSVD, Tikhonov, senza
# shrinkage) e integrale con debiasing di scala per L1 e L1+TV (che per la penalita'
# sottostimano). Le tre stime sono comunque tutte nel file; questa mappa decide solo
# quale va in rapporto_stima per ciascun metodo.
stima_per_metodo = {"TSVD": "integrale", "Tikhonov": "integrale",
                    "L1": "debias", "L1TV": "debias"}
if set(stima_per_metodo.values()) - {"fit", "integrale", "debias"}:
    raise ValueError("stima_per_metodo: valori ammessi 'fit', 'integrale', 'debias'.")
file_risultati = os.path.join(cartella_risultati,
                              f"risultati_confronto_4metodi_misto_IRF_"
                              f"{time.strftime('%Y%m%d_%H%M')}.xlsx")

composizione = {
    "HHb":   10.0,    # [uM]  emoglobina deossigenata
    "O2Hb":  30.0,    # [uM]  emoglobina ossigenata
    "Lipid":  0.70,   # [-]   frazione in volume di lipide
    "H2O":    0.30,   # [-]   frazione in volume di acqua
}

wl_nm = np.arange(650, 951, 10, dtype=float)   # [nm] 650-950 nm, passo 10, estremi inclusi
N_wl = wl_nm.size

# scattering ridotto: Jacques, Phys. Med. Biol. 58 R37 (2013), Tab. 2, "fatty tissue"
a_jac = 18.4 / 10     # [mm^-1] mu_s' a lambda0 (18.4 cm^-1 -> 1.84 mm^-1)
b_jac = 0.672         # [-]     scattering power
wl0_jac = 500.0       # [nm]    lunghezza d'onda di riferimento


def leggi_spettri(percorso, foglio="new"):
    """Spettri specifici dei cromofori dal file Excel (righe 1-4 = intestazioni)."""
    df = pd.read_excel(percorso, sheet_name=foglio, skiprows=4)
    df = df[pd.to_numeric(df["Lambda"], errors="coerce").notna()]
    return {k: df[k].to_numpy(dtype=float) for k in ("Lambda", "HHb", "O2Hb", "Lipid", "H2O")}


def mua_background(spettri, comp, wl):
    """mu_a [mm^-1] = 0.1 * [C_HHb eps_HHb + C_O2Hb eps_O2Hb + f_lip mua_lip + f_w mua_w]"""
    L = spettri["Lambda"]
    if wl.min() < L.min() or wl.max() > L.max():
        raise ValueError(f"Lunghezze d'onda {wl.min():.0f}-{wl.max():.0f} nm fuori dal "
                         f"range del file ({L.min():.0f}-{L.max():.0f} nm).")
    mua_cm = np.zeros_like(wl)
    for nome, peso in comp.items():
        mua_cm += peso * np.interp(wl, L, spettri[nome])
    return mua_cm / 10


def mus_jacques(wl, a=a_jac, b=b_jac, wl0=wl0_jac):
    """mu_s' [mm^-1] dalla legge di potenza di Jacques."""
    return a * (wl / wl0)**(-b)


# Variazione dell'inclusione rispetto al background:
#   - lipide/acqua COMPLEMENTARI: dH2O = -d_lip
#   - emoglobina: si assegnano SEPARATAMENTE le variazioni di HHb e di O2Hb
#     [uM]. La variazione di emoglobina totale e' la loro somma,
#         d_tHb = d_HHb + d_O2Hb,
#     e la saturazione dell'inclusione SO2_incl = O2Hb_incl / tHb_incl dice se
#     la variazione e' "ossigenata" (SO2 sale), "deossigenata" (SO2 scende) o
#     a SO2 costante (stessa proporzione del background: vedi ripartisci_tHb).
#     dmu_a = 0.1 [d_HHb eps_HHb + d_O2Hb eps_O2Hb + d_lip (mua_lip - mua_H2O)]
#   Le due emoglobine hanno spettri diversi (HHb ha il picco a 760 nm, O2Hb
#   cresce oltre 800 nm): variare l'una o l'altra cambia la FORMA di dmu_a(lambda).
# !!! LIMITI: |d_lip| <= 0.30;  |d_tHb| <= 200 uM !!!
def ripartisci_tHb(d_tHb, comp_bkg):
    """(d_HHb, d_O2Hb) per una variazione d_tHb a SO2 costante (quella del background)."""
    SO2 = comp_bkg["O2Hb"] / (comp_bkg["HHb"] + comp_bkg["O2Hb"])
    return (1 - SO2) * d_tHb, SO2 * d_tHb


def delta_mua_inclusione(spettri, comp_bkg, wl, d_lip=0.0, d_HHb=0.0, d_O2Hb=0.0):
    """dmu_a [mm^-1] (N_wl,) dell'inclusione e sua composizione [dict, con tHb e SO2]."""
    d_tHb = d_HHb + d_O2Hb
    if abs(d_lip) > 0.30:
        print(f"ATTENZIONE: |d_lip| = {abs(d_lip):.2f} > 0.30 (limite: 30%).")
    if abs(d_tHb) > 200:
        print(f"ATTENZIONE: |d_tHb| = {abs(d_tHb):.0f} uM > 200 uM.")
    d_comp = {"HHb": d_HHb, "O2Hb": d_O2Hb, "Lipid": d_lip, "H2O": -d_lip}
    comp_incl = {k: comp_bkg[k] + d_comp[k] for k in comp_bkg}
    if not (0 <= comp_incl["Lipid"] <= 1 and 0 <= comp_incl["H2O"] <= 1):
        raise ValueError(f"Frazioni non fisiche nell'inclusione: lipide "
                         f"{comp_incl['Lipid']:.2f}, acqua {comp_incl['H2O']:.2f}.")
    if comp_incl["HHb"] < 0 or comp_incl["O2Hb"] < 0:
        raise ValueError(f"Concentrazioni negative nell'inclusione: HHb "
                         f"{comp_incl['HHb']:.1f} uM, O2Hb {comp_incl['O2Hb']:.1f} uM.")
    comp_incl["tHb"] = comp_incl["HHb"] + comp_incl["O2Hb"]
    comp_incl["SO2"] = comp_incl["O2Hb"] / comp_incl["tHb"] if comp_incl["tHb"] > 0 else np.nan
    return mua_background(spettri, d_comp, wl), comp_incl


spettri = leggi_spettri(file_spettri, foglio_spettri)
mu_a_bkg = mua_background(spettri, composizione, wl_nm)          # [mm^-1] (N_wl,)
mu_s_bkg = mus_jacques(wl_nm)                                    # [mm^-1] (N_wl,)


# %% =============================================================
# STEP 1 - Geometria sorgenti/rivelatori, griglia, gate
# ==================================================================
n_idx = 1.4           # [-]      indice di rifrazione del mezzo

def A_coefficient(n):
    """Coefficiente A della condizione al contorno estrapolata [-]."""
    Reff = -1.4399/n**2 + 0.7099/n + 0.6681 + 0.0636*n
    return (1 + Reff)/(1 - Reff)

A_coeff = A_coefficient(n_idx)

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
        {"type": "detector", "x": -24, "y": 24},
        {"type": "detector", "x": 24,   "y": 24},
        {"type": "detector", "x": 24,  "y": -24},
        {"type": "detector", "x": -24,  "y": -24},
    ],
}

src_xy = np.array([[p["x"], p["y"]] for p in geometria_base["points"]
                   if p["type"] == "source"], dtype=float)
det_xy = np.array([[p["x"], p["y"]] for p in geometria_base["points"]
                   if p["type"] == "detector"], dtype=float)
N_src, N_det = src_xy.shape[0], det_xy.shape[0]
r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))           # rivelatori a z = 0

# Coppie s-d: per ogni sorgente, i rivelatori fino all'ordine di vicinanza scelto
# (1 = piu' vicini, 2 = secondi, 3 = terzi; distanze uguali entro tol_vicini mm).
modalita_coppie = "primi_secondi_terzi_vicini"
tol_vicini = 0.5
ORDINI_PER_MODALITA = {"primi_vicini": 1, "primi_secondi_vicini": 2,
                       "primi_secondi_terzi_vicini": 3, "tutte": None}
if modalita_coppie not in ORDINI_PER_MODALITA:
    raise ValueError(f"modalita_coppie = '{modalita_coppie}' non valida: usare "
                     f"{list(ORDINI_PER_MODALITA)}.")


def ordini_di_vicinanza(dist, tol):
    livelli = []
    for dd in np.sort(dist):
        if not livelli or dd - livelli[-1] > tol:
            livelli.append(dd)
    livelli = np.array(livelli)
    return np.argmin(np.abs(dist[:, None] - livelli[None, :]), axis=1) + 1


def seleziona_coppie(modalita, tol=tol_vicini):
    """Coppie (sorgente, rivelatore) con ordine di vicinanza <= quello della modalita',
    ordinate per sorgente e, per ogni sorgente, per distanza crescente."""
    ordine_max = ORDINI_PER_MODALITA[modalita]
    coppie = []
    for s in range(N_src):
        dist = np.linalg.norm(det_xy - src_xy[s], axis=1)
        ordini = ordini_di_vicinanza(dist, tol)
        scelti = np.arange(N_det) if ordine_max is None else np.where(ordini <= ordine_max)[0]
        coppie += [(s, int(d)) for d in scelti[np.argsort(dist[scelti])]]
    return coppie


pairs = seleziona_coppie(modalita_coppie)
N_pairs = len(pairs)
rho_pairs = np.array([np.linalg.norm(src_xy[s] - det_xy[d]) for s, d in pairs])

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
Nx, Ny, Nz = x_coords.size, y_coords.size, z_coords.size

# --- IRF misurate (una per lunghezza d'onda) e asse dei tempi comune ---------
file_irf = r"C:\Users\marti\OneDrive - Politecnico di Milano\tesi magistrale-MSI\ottimizzazione\TXTs0012.DAT"
irf_loop = [51]                          # numero di curve nel file
irf_wl_nm = np.linspace(600, 1100, 51)   # lunghezza d'onda di ogni curva [nm]
irf_canali = 4096                        # canali per curva (verificato sull'header)
irf_tipo_dati = "uint32"
irf_ps_per_canale = 3.05                 # [ps] larghezza di un canale
dc = irf_ps_per_canale * 1e-3            # [ns] passo temporale di TUTTO il calcolo
irf_liscia_canali = 5        # media mobile SOLO per trovare picco e salita della IRF

# Fondo: stimato prima del picco della IRF, in conteggi per canale
irf_fondo_prima_ns = 1.0     # la finestra del fondo finisce 1 ns prima del picco della IRF
fondo_finestra_ns = 1.0      # [ns] durata della finestra (None = dal canale 0)

# Riferimento temporale dei gate (lo stesso va usato sulle DTOF reali):
#   "salita" (frazione_salita del picco della IRF), "picco", "baricentro"
riferimento_gate = "salita"
frazione_salita = 0.5
gate_fissi = True            # stessi canali per tutte le lambda (come nella misura reale)
wl_riferimento_gate = 800.0  # [nm] IRF da cui si prende il riferimento comune
t_rif_fisso_ns = None        # [ns] None = dal riferimento della IRF a wl_riferimento_gate

# Gate da t_start, larghezza gate_width, contati dall'istante di riferimento e
# arrotondati ai canali (0.2 ns = 65.57 canali -> 65-66 canali), identici per la
# misura e per la W dell'inversione. Finestra = t_start + N_gates gate_width dal
# riferimento (20 gate -> 4.05 ns): per prolungare il campionamento basta aumentare
# N_gates (es. 25 -> 5.05 ns, 30 -> 6.05 ns).
# N_tot_TPSF si riferisce SEMPRE ai primi N_gates_normalizzazione gate: aggiungendo gate
# i conteggi dei primi 20 restano identici e quelli nuovi sono fotoni in piu' (stessa
# acquisizione, finestra piu' lunga), cosi' il confronto fra finestre e' alla pari.
t_start = 0.05
gate_width = 0.2
N_gates = 25
N_gates_normalizzazione = 20
N_meas = N_pairs * N_gates
gate_validi = np.ones(N_meas, bool)   # aggiornato per ogni lambda in prepara_lambda

# Problema diretto della misura e modello dell'inversione (vedi intestazione)
misura_con_irf = True        # False = misura senza IRF e senza fondo
inversione_con_irf = True    # True = W_irf (IRF nella W); False = W senza IRF

HEADLEN_DAT = 764


def leggi_dat(percorso, loop, canali=None, tipo_dati="uint32"):
    """Legge un file .DAT: array (prod(loop), numbin) di conteggi grezzi."""
    with open(percorso, "rb") as f:
        head = f.read(HEADLEN_DAT)
    if len(head) != HEADLEN_DAT:
        raise ValueError(f"{percorso}: header incompleto ({len(head)} byte).")
    sublen = struct.unpack_from("<i", head, 16)[0]
    numbin = struct.unpack_from("<i", head, 116)[0]
    if not (0 < sublen <= 100_000 and 0 < numbin <= 10_000_000):
        raise ValueError(f"{percorso}: header anomalo (sublen={sublen}, numbin={numbin}).")
    if canali is not None and int(canali) != numbin:
        raise ValueError(f"canali = {canali} ma il file ne ha {numbin}.")
    fmt = {"uint16": "<H", "ushort": "<H", "uint32": "<I", "ulong": "<I"}[tipo_dati.lower()]
    nbyte = numbin * struct.calcsize(fmt)
    nframes = int(np.prod(loop))
    out = []
    with open(percorso, "rb") as f:
        f.read(HEADLEN_DAT)
        for _ in range(nframes):
            if len(f.read(sublen)) != sublen:
                break
            raw = f.read(nbyte)
            if len(raw) != nbyte:
                break
            out.append(np.frombuffer(raw, dtype=np.dtype(fmt)).astype(float))
    if len(out) != nframes:
        raise ValueError(f"{percorso}: lette {len(out)} curve su {nframes}.")
    return np.array(out)


def media_mobile(c, n):
    return np.convolve(c, np.ones(n) / n, mode="same") if n > 1 else c


def attraversa(c, livello, k_da, verso=-1):
    """Tempo [in canali, frazionario] in cui c attraversa livello partendo da k_da
    verso sinistra (verso=-1) o destra (+1); interpolazione lineare."""
    k = k_da
    while 0 < k < c.size - 1 and c[k] > livello:
        k += verso
    k1 = k - verso
    a, b = (k, k1) if verso == -1 else (k1, k)
    return a + (livello - c[a]) / (c[b] - c[a]) if c[b] != c[a] else float(k)


def canali_fondo(k_picco, n_canali):
    """Canali della finestra di stima del fondo, prima del picco."""
    t_ax = np.arange(n_canali) * dc
    t_fine = k_picco * dc - irf_fondo_prima_ns
    t_ini = -np.inf if fondo_finestra_ns is None else t_fine - fondo_finestra_ns
    k = np.where((t_ax >= t_ini) & (t_ax < t_fine))[0]
    if k.size < 20:
        raise ValueError(f"Solo {k.size} canali per stimare il fondo prima del picco: "
                         f"ridurre irf_fondo_prima_ns o fondo_finestra_ns.")
    return k


def prepara_irf(conteggi):
    """IRF GREZZA (con il suo fondo) normalizzata ad area NETTA unitaria:
        h = conteggi / (sum(conteggi - fondo_IRF) dc)
    cosi' la parte di segnale ha area 1 e il fondo resta nella stessa proporzione
    della IRF misurata. Riporta picco, salita, FWHM e il riferimento dei gate."""
    c = np.asarray(conteggi, float)
    K = c.size
    cl = media_mobile(c, irf_liscia_canali)
    k_p = int(np.argmax(cl))
    k_f = canali_fondo(k_p, K)
    fondo = float(np.mean(c[k_f]))
    netti = float(np.sum(c - fondo))
    h = c / (netti * dc)
    picco_l = cl[k_p] - fondo
    k_sal = attraversa(cl - fondo, frazione_salita * picco_l, k_p, -1)
    k_disc = attraversa(cl - fondo, 0.5 * picco_l, k_p, +1)
    k_sal50 = attraversa(cl - fondo, 0.5 * picco_l, k_p, -1)
    t_ax = np.arange(K) * dc
    h_netta = (c - fondo) / (netti * dc)
    info = dict(fondo=fondo, conteggi_netti=netti, k_picco=k_p, t_picco=k_p * dc,
                t_salita=k_sal * dc, t_baricentro=float(np.sum(t_ax * h_netta) * dc),
                fwhm_ps=(k_disc - k_sal50) * irf_ps_per_canale)
    info["t_rif"] = {"salita": info["t_salita"], "picco": info["t_picco"],
                     "baricentro": info["t_baricentro"]}[riferimento_gate]
    return h, h_netta, info


irf_grezze = leggi_dat(file_irf, irf_loop, irf_canali, irf_tipo_dati)
if irf_grezze.shape[0] != irf_wl_nm.size:
    raise ValueError(f"Il file IRF ha {irf_grezze.shape[0]} curve, attese {irf_wl_nm.size}.")
K_canali = irf_grezze.shape[1]
if K_canali != irf_canali:
    raise ValueError(f"La IRF ha {K_canali} canali, attesi {irf_canali}.")
t_s = np.arange(K_canali) * dc                        # asse dello strumento [ns]

if t_rif_fisso_ns is not None:
    t_rif_comune = float(t_rif_fisso_ns)
else:
    _k = int(np.argmin(np.abs(irf_wl_nm - wl_riferimento_gate)))
    t_rif_comune = prepara_irf(irf_grezze[_k])[2]["t_rif"]


def gate_canali(t_rif):
    """Canali dei gate: [t_rif + t_start + g gate_width, ...), arrotondati ai canali
    (canale k nel gate g se bordo_g <= k dc < bordo_g+1)."""
    bordi = t_rif + t_start + gate_width * np.arange(N_gates + 1)
    if bordi[-1] > (K_canali - 1) * dc:
        raise ValueError(f"L'ultimo gate finisce a {bordi[-1]:.2f} ns, oltre la finestra "
                         f"({K_canali} canali x {irf_ps_per_canale} ps = {K_canali*dc:.2f} ns).")
    idx = np.searchsorted(t_s, bordi)
    return bordi, list(zip(idx[:-1].tolist(), idx[1:].tolist()))


def irf_per_lambda(wl):
    """IRF (grezza e netta) della curva con lambda piu' vicina a wl e canali dei gate."""
    k = int(np.argmin(np.abs(irf_wl_nm - wl)))
    h, h_netta, info = prepara_irf(irf_grezze[k])
    info["wl_irf"] = irf_wl_nm[k]
    if gate_fissi:
        info["t_rif"] = t_rif_comune
    info["bordi_gate"], info["gate_idx"] = gate_canali(info["t_rif"])
    info["k_rif"] = int(round(info["t_rif"] / dc))   # canale di t' = 0 per la W senza IRF
    return h, h_netta, info

# NB: in Python '^' e' lo XOR bit a bit (10^8 = 2): la potenza si scrive 1e8 o 10**8
N_tot_TPSF = 1e7        # conteggi senza fondo nei primi N_gates_normalizzazione gate, per coppia s-d
# Rumore del fondo nella misura simulata (vedi simula_misura):
#   True  -> ogni acquisizione (baseline e perturbata) conta segnale + fondo nei gate e
#            sottrae il fondo stimato nella finestra prima del picco: al rumore di Poisson
#            del segnale si aggiungono quello del fondo nel gate e quello della sua stima
#   False -> Poisson sui soli conteggi senza fondo (ottimista per i gate tardivi)
rumore_fondo = True
seme_rumore = 42        # rumore riproducibile (un generatore unico per tutto lo scan)
                        # 42 = stesso rumore degli scan precedenti (confronto diretto);
                        # un altro valore (es. 43) per validare la configurazione scelta

# Sigma usata per pesare le misure (vedi intestazione):
#   "baseline" -> sigma_0 dai conteggi attesi di baseline, pesi una volta per lambda (veloce)
#   "dati"     -> stimata dai conteggi osservati di ogni misura, pesi per caso (realistico,
#                 piu' lento: SVD e fattorizzazione ADMM ripetute per ogni caso)
# "baseline" calcola i pesi (W_w, SVD, fattorizzazione ADMM) UNA volta per lambda (31
# volte): molto piu' veloce. "dati" li ricalcola per OGNI caso (279) + ogni realizzazione
# di calibrazione: ~9x piu' lento nel ciclo principale. Nel regime di Born
# sigma_dati/sigma_0 ~ 1.000 (colonna sigma_dati_su_sigma0), quindi "baseline" e' quasi
# equivalente. Usare "dati" solo per lo studio del rumore eteroschedastico.
modello_sigma = "baseline"
if modello_sigma not in ("dati", "baseline"):
    raise ValueError(f"modello_sigma = '{modello_sigma}' non valido: usare 'dati' o 'baseline'.")


# %% =============================================================
# STEP 2 - Posizioni e variazioni da scansionare
# ==================================================================
# Inclusione 2x2x2: il centro deve stare su un vertice fra voxel
# (x, y multipli di 4 fra -28 e 28; z in {4, 8, ..., 28}).
n_blk = (2, 2, 2)
posizioni = [   # (id, descrizione, (xp, yp, zp) [mm])
    ("P1", "superficiale", (8.0, 20.0,  8.0)),    # blocco z 4-12 mm
    ("P2", "intermedia",   (8.0,  4.0, 16.0)),    # blocco z 12-20 mm
    ("P3", "profonda",     (0.0,  0.0, 20.0)),    # blocco z 16-24 mm
]

# (id, descrizione, d_lip [-], d_HHb [uM], d_O2Hb [uM])
# Background: HHb 10 uM, O2Hb 30 uM (tHb 40 uM, SO2 75%), lipide 70%, acqua 30%.
# Varia UN cromoforo alla volta: HHb +10 uM, O2Hb -10 uM, lipidi +30% (acqua -30%).
variazioni = [
    ("V1", "HHb +10",   0.0,  10.0,   0.0),   # HHb 20 uM,  tHb 50 uM, SO2 60%
    ("V2", "O2Hb -10",  0.0,   0.0, -10.0),   # O2Hb 20 uM, tHb 30 uM, SO2 67%
    # Lipidi +30 punti percentuali di frazione in volume (il limite |d_lip| <= 0.30),
    # acqua complementare (dH2O = -d_lip), emoglobine invariate:
    ("V3", "lipide +30% (acqua -30%)",  0.30, 0.0, 0.0),   # lipide 100%, acqua 0%
]

# Sottoinsieme da scansionare (None = tutte). Per ridurre i tempi, ad esempio:
# posizioni_scan = ["P2"]; variazioni_scan = ["V3"].
posizioni_scan = None
variazioni_scan = None
if posizioni_scan is not None:
    posizioni = [p for p in posizioni if p[0] in posizioni_scan]
if variazioni_scan is not None:
    variazioni = [v for v in variazioni if v[0] in variazioni_scan]
if not posizioni or not variazioni:
    raise ValueError("Nessuna posizione o variazione selezionata: controllare posizioni_scan "
                     "e variazioni_scan.")

L_blk = np.array(n_blk) * step
maschere = {}
for pid, pdesc, (xp, yp, zp) in posizioni:
    m = ((np.abs(X - xp) < L_blk[0]/2) & (np.abs(Y - yp) < L_blk[1]/2) &
         (np.abs(Z - zp) < L_blk[2]/2)).flatten()
    if m.sum() != np.prod(n_blk):
        raise ValueError(f"{pid}: il blocco contiene {m.sum()} voxel invece di "
                         f"{np.prod(n_blk)} (centro non allineato alla griglia).")
    maschere[pid] = m

dmu_var, comp_var = {}, {}
for vid, vdesc, d_lip, d_HHb, d_O2Hb in variazioni:
    dmu_var[vid], comp_var[vid] = delta_mua_inclusione(spettri, composizione, wl_nm,
                                                       d_lip, d_HHb, d_O2Hb)

N_casi = N_wl * len(posizioni) * len(variazioni)
print("="*78)
print(f"SCAN: {N_wl} lambda ({wl_nm[0]:.0f}-{wl_nm[-1]:.0f} nm) x {len(posizioni)} posizioni x "
      f"{len(variazioni)} variazioni = {N_casi} casi, 4 metodi per caso "
      f"(TSVD, Tikhonov, L1, L1+TV)")
print(f"Geometria: {N_pairs} coppie s-d ({modalita_coppie}), {N_gates} gate, "
      f"{N_meas} misure, {N_vox} voxel, N_tot = {N_tot_TPSF:.0e} conteggi/TPSF/coppia")
print(f"Risultati: {file_risultati}")
print(f"Pesi delle misure: modello_sigma = '{modello_sigma}' "
      + ("(sigma stimata dai conteggi osservati, pesi ricalcolati per caso)" if modello_sigma == "dati"
         else "(sigma_0 dai conteggi attesi di baseline, pesi per lambda)"))
print("Posizioni: " + "; ".join(f"{p[0]} {p[2]}" for p in posizioni))
print(f"Variazioni (background: HHb {composizione['HHb']:.0f} uM, O2Hb {composizione['O2Hb']:.0f} uM, "
      f"SO2 {100*composizione['O2Hb']/(composizione['HHb']+composizione['O2Hb']):.0f}%, "
      f"lipide {100*composizione['Lipid']:.0f}%):")
for vid, vdesc, d_lip, d_HHb, d_O2Hb in variazioni:
    ci = comp_var[vid]
    print(f"   {vid} {vdesc:38s} dO2Hb {d_O2Hb:+7.1f}  dHHb {d_HHb:+6.1f}  "
          f"dtHb {d_HHb + d_O2Hb:+7.1f} uM  dlip {d_lip:+.2f}  ->  SO2 incl {100*ci['SO2']:.0f}%, "
          f"dmu_a {dmu_var[vid].min():+.1e}..{dmu_var[vid].max():+.1e} mm^-1")
print("="*78 + "\n")


# %% =============================================================
# FUNZIONI - problema diretto e rumore
# ==================================================================
def deltafluence_inf(mu_a0, mu_s0, t, dmu_a, V, r, r1, n=n_idx):
    D = 1/(3*mu_s0); v = 3e2/n
    return (-(v**2)/((4*pi*D*v)**(5/2)*t**(3/2))*exp(-mu_a0*v*t)
            * dmu_a*V*(1/r + 1/r1)*exp(-(r+r1)**2/(4*D*v*t)))

def fluence_inf(mu_a0, mu_s0, n, t, r):
    D = 1/(3*mu_s0); v = 3e2/n
    return (v/(4*pi*D*v*t)**(3/2))*exp(-(r**2/(4*D*v*t)) - mu_a0*v*t)

def convolvi(curve, H, nfft, n_c, blocco=256):
    """Convoluzione con la IRF sull'asse dei canali:
        DTOF[n] = dc * sum_m phi[m] h[n - m],   n = 0 .. n_c-1
    phi[m] = phi(t' = m dc) (canale 0 = ingresso nel tessuto), H = rfft(h[:n_c], nfft)
    con nfft >= 2 n_c - 1 (convoluzione lineare, senza ricircolo). A blocchi di righe."""
    curve = np.atleast_2d(curve)
    out = np.empty((curve.shape[0], n_c))
    for i in range(0, curve.shape[0], blocco):
        out[i:i + blocco] = irfft(rfft(curve[i:i + blocco], nfft, axis=-1) * H, nfft,
                                  axis=-1)[:, :n_c] * dc
    return out


def matrice_W(mu_a0, mu_s0, h, h_netta, info, coppia_curve=None):
    """Matrici di sensitivita' e frazioni della TPSF nei gate, sui canali dello strumento.
      W_id, frac_id   SENZA IRF: fluenza con t' = 0 nel canale di riferimento k_rif,
                      gate sugli stessi canali
      W_irf, frac_irf CON IRF: fluenza (x) IRF netta + fondo stazionario
                      b_h int phi dt (costante su tutti i canali), fondo stimato prima
                      del picco (media per canale) e sottratto canale per canale
    La normalizzazione tot e' l'area di phi0 nei primi N_gates_normalizzazione gate
    (asse t'): N_tot_TPSF conteggi in quella finestra, qualunque sia N_gates."""
    gate_idx, k_rif = info["gate_idx"], info["k_rif"]
    # fine della finestra di normalizzazione (canale dello strumento), calcolata come in
    # gate_canali: con N_gates = N_gates_normalizzazione coincide con la fine dei gate
    k_norm = int(np.searchsorted(t_s, info["t_rif"] + t_start
                                 + gate_width * N_gates_normalizzazione))
    n_c = max(gate_idx[-1][1], k_norm)                   # canali fino alla fine dell'ultimo gate
    nfft = next_fast_len(2 * n_c - 1)
    Hn = rfft(h_netta[:n_c], nfft)
    # livello di fondo della IRF normalizzata (h = h_netta + b_h): fondo per canale /
    # area netta. Nella DTOF il fondo e' stazionario: b_h * (integrale della fluenza)
    b_h = info["fondo"] / (info["conteggi_netti"] * dc)
    k_f = canali_fondo(info["k_picco"], n_c)
    gate_id = [(i0 - k_rif, i1 - k_rif) for i0, i1 in gate_idx]   # stessi gate sull'asse t'
    if gate_id[0][0] < 1:
        raise ValueError("Il primo gate cade prima di t' = 0: aumentare t_start.")
    tp = np.arange(n_c) * dc
    tp[0] = np.inf                                       # phi(t' = 0) = 0 (niente 1/0)
    tt = tp[None, :]
    D = 1/(3*mu_s0); z0 = 1/mu_s0; zb = 2*A_coeff*D
    r_s0 = np.column_stack((src_xy, np.full(N_src, z0)))
    r_si = np.column_stack((src_xy, np.full(N_src, -z0 - 2*zb)))
    r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))
    r_sv_all  = np.linalg.norm(r_V[None] - r_s0[:, None], axis=2)
    r_siv_all = np.linalg.norm(r_V[None] - r_si[:, None], axis=2)
    r_vd_all  = np.linalg.norm(r_V[None] - r_d0[:, None], axis=2)
    r_vdi_all = np.linalg.norm(r_V[None] - r_di[:, None], axis=2)
    W_id, W_irf = np.zeros((N_meas, N_vox)), np.zeros((N_meas, N_vox))
    frac_id, frac_irf = np.zeros((N_pairs, N_gates)), np.zeros((N_pairs, N_gates))
    fondo_cnt = np.zeros(N_pairs)                        # fondo stimato [conteggi/canale]
    residuo = np.zeros((N_pairs, N_gates))               # fondo rimasto dopo la sottrazione
    n_canali_azzerati = np.zeros(N_pairs, int)           # canali dei gate posti a 0 (vincolo)
    curve = {}
    for p, (s, d) in enumerate(pairs):
        a, b = r_sv_all[s][:, None], r_siv_all[s][:, None]
        c, e = r_vd_all[d][:, None], r_vdi_all[d][:, None]
        dphi = (deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, a, c)
                - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, a, e)
                - deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, b, c)
                + deltafluence_inf(mu_a0, mu_s0, tt, 1.0, V_vox, b, e))
        phi0 = (fluence_inf(mu_a0, mu_s0, n_idx, tp, np.linalg.norm(r_d0[d] - r_s0[s]))
                - fluence_inf(mu_a0, mu_s0, n_idx, tp, np.linalg.norm(r_d0[d] - r_si[s])))
        tot = np.sum(phi0[gate_id[0][0]:k_norm - k_rif]) * dc
        # --- senza IRF (sui canali) -----------------------------------------------
        for g, (i0, i1) in enumerate(gate_id):
            int_phi0 = np.sum(phi0[i0:i1]) * dc
            W_id[p*N_gates + g, :] = np.sum(dphi[:, i0:i1], axis=1) * dc / int_phi0
            frac_id[p, g] = int_phi0 / tot
        # --- con IRF: convoluzione con la IRF netta + fondo stazionario, stima del
        # fondo nella finestra prima del picco, sottrazione canale per canale
        conv0_vero = convolvi(phi0, Hn, nfft, n_c)[0]    # curva senza fondo
        conv0 = conv0_vero + b_h * np.sum(phi0) * dc     # + fondo costante
        # variazione: anche il fondo cambia (di b_h int dphi dt, costante nel tempo)
        conv_d = convolvi(dphi, Hn, nfft, n_c) + b_h * dphi.sum(axis=1, keepdims=True) * dc
        conv_d -= conv_d[:, k_f].mean(axis=1, keepdims=True)
        b0 = conv0[k_f].mean()
        conv0_n = conv0 - b0
        # VINCOLO: dove la curva senza fondo e' <= 0 i conteggi attesi si pongono a 0
        # canale per canale; in quei canali anche la variazione (W) e' nulla, cosi'
        # misura e matrice restano coerenti (nessun fotone -> nessuna sensibilita').
        positivi = conv0_n > 0
        n_canali_azzerati[p] = int(np.sum(~positivi[gate_idx[0][0]:gate_idx[-1][1]]))
        conv0_n = np.where(positivi, conv0_n, 0.0)
        conv_d *= positivi[None, :]
        fondo_cnt[p] = N_tot_TPSF * dc * b0 / tot
        for g, (i0, i1) in enumerate(gate_idx):
            int_phi0 = np.sum(conv0_n[i0:i1]) * dc
            if int_phi0 > 0:
                W_irf[p*N_gates + g, :] = np.sum(conv_d[:, i0:i1], axis=1) * dc / int_phi0
            # else: gate senza conteggi attesi -> riga di W nulla, frac = 0 (gate escluso)
            frac_irf[p, g] = int_phi0 / tot
            vero_g = np.sum(conv0_vero[i0:i1]) * dc
            residuo[p, g] = int_phi0 / vero_g - 1 if vero_g > 0 else np.nan
        if p == coppia_curve:                            # conteggi per canale, per i grafici
            sc = N_tot_TPSF * dc / tot
            curve = dict(conv0=sc * conv0, fondo=sc * b0, conv0_n=sc * conv0_n,
                         conv0_vero=sc * conv0_vero,
                         ideale=sc * np.concatenate((np.zeros(k_rif), phi0[:n_c - k_rif])),
                         k_f=k_f, n_c=n_c)
    return dict(W_id=W_id, frac_id=frac_id, W_irf=W_irf, frac_irf=frac_irf,
                fondo_cnt=fondo_cnt, residuo=residuo, curve=curve,
                n_canali_azzerati=n_canali_azzerati, n_canali_fondo=int(k_f.size),
                n_canali_gate=np.array([i1 - i0 for i0, i1 in gate_idx]))

def misura_valida(M_est):
    """Gate senza conteggi attesi (vincolo a 0 dopo la sottrazione del fondo): nessun
    fotone, nessuna informazione -> contrasto posto a 0 (la riga di W e' nulla).
    Senza questo simula_misura darebbe M = 0/(0+1) - 1 = -1 in quei gate."""
    M_est = M_est.copy()
    M_est.reshape(-1)[~gate_validi] = 0.0
    return M_est


def con_fondo():
    """True se la misura simulata include il rumore del fondo (solo misura con IRF:
    senza IRF non c'e' fondo)."""
    return rumore_fondo and misura_con_irf


def simula_misura(lam0, M_ideal, rng):
    """Baseline + misura perturbata (Poisson). Restituisce M stimato e la sua
    deviazione standard stimata dai conteggi osservati (metodo delta).
    Senza rumore del fondo: N0 ~ Poisson(lam0), N ~ Poisson(lam0 (1+M)),
        Var(N/N0) ~ N/N0^2 + N^2/N0^3   (+1 per evitare divisioni per zero).
    Con rumore del fondo (con_fondo()), per ogni acquisizione e coppia:
        lordi_g  ~ Poisson(S_g + b n_g)          conteggi del gate g (segnale + fondo)
        finestra ~ Poisson(b n_f)                UNA finestra del fondo per curva
        netti_g  = lordi_g - finestra n_g / n_f  (fondo stimato e sottratto)
        Var(netti_g) ~ lordi_g + finestra (n_g / n_f)^2
    (b = fondo per canale della coppia, n_g, n_f = canali del gate e della finestra).
    La stima del fondo e' comune a tutti i gate della curva (errori correlati fra i
    gate); la sigma restituita e' quella del singolo gate."""
    if not con_fondo():
        n0 = rng.poisson(lam0).astype(float)
        npert = rng.poisson(np.clip(lam0 * (1 + M_ideal), 0, None)).astype(float)
        M_est = npert / (n0 + 1) - 1
        sig_hat = np.sqrt((npert + 1) / (n0 + 1)**2 + (npert + 1)**2 / (n0 + 1)**3)
        return M_est, sig_hat
    b = fondo_canale[:, None]                            # (N_pairs, 1) conteggi/canale
    c = (n_canali_gate / n_canali_fondo)[None, :]        # (1, N_gates) = n_g / n_f
    def acquisizione(S):
        lordi = rng.poisson(S + b * n_canali_gate[None, :]).astype(float)
        finestra = rng.poisson(b * n_canali_fondo).astype(float)
        return lordi - finestra * c, lordi + finestra * c**2
    n0, v0 = acquisizione(lam0)
    npert, vp = acquisizione(np.clip(lam0 * (1 + M_ideal), 0, None))
    d0 = np.maximum(n0, 0) + 1
    M_est = npert / d0 - 1
    sig_hat = np.sqrt((vp + 1) / d0**2 + (np.maximum(npert, 0) + 1)**2 * (v0 + 1) / d0**4)
    return M_est, sig_hat


def sigma_teorica(lam0, M_ideal):
    """Deviazione standard TEORICA di M dai conteggi attesi (stesso modello di
    simula_misura): con V_f = varianza del fondo sottratto nel gate,
        Var(M) = [(lam0+1)(1+M) + V_f + (1+M)^2 (lam0+1 + V_f)] / (lam0+1)^2
    V_f = b n_g (1 + n_g/n_f) con rumore del fondo, 0 senza."""
    if con_fondo():
        n_g = n_canali_gate[None, :]
        V_f = fondo_canale[:, None] * n_g * (1 + n_g / n_canali_fondo)
    else:
        V_f = 0.0
    L = lam0 + 1
    return np.sqrt((L * (1 + M_ideal) + V_f + (1 + M_ideal)**2 * (L + V_f)) / L**2)


def chi2_bersaglio():
    """Numero di gate VALIDI. I gate esclusi dal vincolo (nessun conteggio atteso:
    riga di W nulla, contrasto posto a 0) non portano rumore al chi^2; contarli
    renderebbe il bersaglio troppo alto (troppa regolarizzazione). Senza gate
    esclusi vale N_meas."""
    return int(np.sum(gate_validi))


# %% =============================================================
# PARAMETRI DEI QUATTRO METODI (versione migliore di ciascuno)
# ==================================================================
# Compensazione di profondita': p_j = max(||w_j||, 1/dmu_a_max)^gamma. Il minimo
# 1/dmu_a_max evita di amplificare senza limite i voxel non osservabili.
dmu_a_max = 0.05      # [mm^-1]

# --- TSVD e Tikhonov (configurazione ottimale degli scan del 13-14/10) ---------------
# gamma <= 0.4 sottostima in profondita' (P3 -20/-40%), gamma >= 0.7 sovrastima (P2-P3
# fino a x2): 0.5-0.65 e' la finestra utile.
gamma_tsvd = 0.5
gamma_tik = 0.6
# SOGLIA ADATTIVA della regione: dipende dalla profondita' RICOSTRUITA z_picco_comp del
# picco di p_j |A_j| (nota anche con dati reali). Lista di (z_max, soglia): si usa la
# prima con z_picco_comp <= z_max. In profondita' la sfocatura cresce e serve una
# soglia piu' alta per non includere le code (sovrastima).
regola_soglia = {
    "TSVD":     [(10.0, 0.25), (np.inf, 0.30)],
    "Tikhonov": [(10.0, 0.22), (np.inf, 0.28)],
}

# --- L1 (FISTA) ----------------------------------------------------------------------
# gamma = 1.0 dagli scan L1 del 15/10 (con gamma < 1 la L1 sceglie spesso un supporto
# parziale e sottostima di ~1/3 a 16-20 mm); soglia della regione 0.20.
gamma_L1 = 1.0
soglia_L1 = 0.20
# lam / lam_max (griglia, da lam grande). Si parte comunque da f = 1 (immagine vuota,
# vedi cerca_lam_discrepanza): anche se 0.6 e' gia' accettato si biseca fra 1 e 0.6.
fattori_lam = (0.6, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4)
# Se gia' l'immagine vuota rispetta il bersaglio (||m_w||^2 <= N + 2 sqrt(2N)) la
# discrepanza non sceglie lam:
#   None  -> si restituisce l'immagine vuota (f = 1, "nessuna inclusione richiesta dai dati")
#   0.3   -> ricostruzione a lam = 0.3 lam_max, come le versioni precedenti dello scan
# I casi si riconoscono dalla colonna 'compatibile_con_zero' del foglio 'dati'.
f_se_vuota = 0.3
n_bisezioni_lam = 3   # bisezioni in log(lam) dopo la griglia (risoluzione ~15%)
tol_fista = 1e-3      # arresto: variazione relativa di x < tol_fista
n_iter_max = 2000

# --- L1 + TV (ADMM) ------------------------------------------------------------------
alpha_tv = 0.95       # peso relativo L1/TV (1 = sola L1); L* = 6(1-alpha)/alpha = 0.32 voxel
soglia_L1TV = 0.20    # soglia della regione (frazione del picco di p|A|)
rho_rel_tv = 1.0      # costante c delle penalita' rho_u = c p^2, rho_z = c q^2
tol_admm = 1e-3       # arresto: residuo primale e variazione di u < tol ||x||
n_iter_admm = 500     # iterazioni massime per ogni lam
if not 0 < alpha_tv <= 1:
    raise ValueError("alpha_tv deve stare in (0, 1] (alpha = 0, sola TV, renderebbe "
                     "lam_max infinito).")

# --- tabella dei metodi ----------------------------------------------------------------
# (metodo, versione, tipo, gamma, soglia della regione (numero o regola));
# stima per metodo secondo stima_per_metodo
METODI_DEF = [
    ("TSVD", f"{stima_per_metodo['TSVD']}_adattivo_g{gamma_tsvd:g}",
     f"{stima_per_metodo['TSVD']}_adattivo", gamma_tsvd, regola_soglia["TSVD"]),
    ("Tikhonov", f"{stima_per_metodo['Tikhonov']}_adattivo_g{gamma_tik:g}",
     f"{stima_per_metodo['Tikhonov']}_adattivo", gamma_tik, regola_soglia["Tikhonov"]),
    ("L1", f"{stima_per_metodo['L1']}_g{gamma_L1:g}_s{soglia_L1:g}", stima_per_metodo['L1'],
     gamma_L1, soglia_L1),
    ("L1TV", f"{stima_per_metodo['L1TV']}_a{alpha_tv:g}_s{soglia_L1TV:g}",
     stima_per_metodo['L1TV'], 1.0, soglia_L1TV),
]
METODI = tuple(m[0] for m in METODI_DEF)
gamma_svd = tuple(sorted({gamma_tsvd, gamma_tik}))   # gamma per cui serve la SVD

# --- soglie di significativita' (in unita' di deviazioni standard del rumore) --
quantile_falsi_positivi = 0.99
soglia_minima_sigma = 2.0                     # nessuna soglia 'trovata' sotto 2 sigma
lambda_calibrazione = (650.0, 800.0, 950.0)   # [nm] lambda usate per la calibrazione
n_realizzazioni_rumore = 40                   # misure di solo rumore per lambda (120 in tutto)
# Calibrazione con i pesi di baseline sig0 (una SVD e una fattorizzazione ADMM per
# lambda invece che per realizzazione): nel regime di Born sigma_dati/sigma_0 ~ 1, e su
# misure di solo rumore M_ideal = 0. False = pesi dai dati anche nella calibrazione.
calibrazione_veloce_sig0 = True
soglia_trovata = {m: 3.0 for m in METODI}     # sovrascritte dalla calibrazione

_DESCR_STIMA = {"fit": "fit (dmu_a uniforme sul cluster, a N_R V_vox)",
               "integrale": "integrale spaziale (V_vox sum_R A_hat)",
               "debias": "integrale + debiasing di scala (x beta sui dati)"}
print("Metodi a confronto (intensita' per metodo, confrontata con dmu x N_voxel_incl x V_vox):")
for _m in METODI:
    print(f"   {_m:9s} stima: {_DESCR_STIMA[stima_per_metodo[_m]]}")
print(f"   TSVD      gamma {gamma_tsvd}, k per discrepanza, "
      f"soglia adattiva {regola_soglia['TSVD']}")
print(f"   Tikhonov  gamma {gamma_tik}, lam per discrepanza, "
      f"soglia adattiva {regola_soglia['Tikhonov']}")
print(f"   L1        gamma {gamma_L1}, FISTA, soglia {soglia_L1}")
print(f"   L1TV      alpha {alpha_tv} (L* = {6*(1-alpha_tv)/alpha_tv:.2f} voxel), ADMM, "
      f"soglia {soglia_L1TV}\n")


# %% =============================================================
# FUNZIONI - ricostruzione
# ==================================================================
# --- TSVD / Tikhonov (SVD di B = W_w diag(1/p)) --------------------------------------
def tsvd_disc(c, m_w, s, Vt, target):
    """TSVD con k per discrepanza: il piu' piccolo k con chi^2 <= target."""
    chi2 = float(m_w @ m_w) - np.cumsum(c**2)
    ok = np.where(chi2 <= target)[0]
    k = int(ok[0]) + 1 if ok.size else s.size
    return Vt[:k].T @ (c[:k] / s[:k]), k, float(chi2[k - 1])


def tikhonov_disc(c, m_w, s, Vt, target):
    """Tikhonov con lam per discrepanza (bisezione in log(lam)): chi^2 = target."""
    res_fuori = max(float(m_w @ m_w) - np.sum(c**2), 0.0)
    chi2 = lambda lam: res_fuori + np.sum(((1 - s**2/(s**2 + lam**2)) * c)**2)
    lo, hi = np.log(1e-10*s[0]), np.log(10*s[0])
    if chi2(np.exp(lo)) > target:
        lam = np.exp(lo)
    elif chi2(np.exp(hi)) < target:
        lam = np.exp(hi)
    else:
        for _ in range(60):
            mid = 0.5*(lo + hi)
            if chi2(np.exp(mid)) > target:
                hi = mid
            else:
                lo = mid
        lam = np.exp(0.5*(lo + hi))
    f = s**2/(s**2 + lam**2)
    return Vt.T @ (f * c / s), lam, float(chi2(lam))


# --- L1 (FISTA) -------------------------------------------------------------------
def norma2_quadra(B, tol=1e-6, n_iter=300, margine=1.01):
    """||B||_2^2 (autovalore massimo di B^T B) col metodo delle potenze (+1%)."""
    x = np.random.default_rng(0).standard_normal(B.shape[1])
    x /= np.linalg.norm(x)
    val = 0.0
    for _ in range(n_iter):
        y = B.T @ (B @ x)
        val_new = np.linalg.norm(y)
        x = y / val_new
        if abs(val_new - val) <= tol * val_new:
            break
        val = val_new
    return margine * val_new

def fista_l1(B, m, lam, Lip, x0=None):
    x = np.zeros(B.shape[1]) if x0 is None else x0.copy()
    y, tk = x.copy(), 1.0
    for _ in range(n_iter_max):
        z = y - B.T @ (B @ y - m) / Lip
        x_new = np.sign(z) * np.maximum(np.abs(z) - lam / Lip, 0)
        t_new = (1 + np.sqrt(1 + 4*tk*tk)) / 2
        y = x_new + (tk - 1) / t_new * (x_new - x)
        if np.linalg.norm(x_new - x) <= tol_fista * max(np.linalg.norm(x_new), 1e-30):
            return x_new
        x, tk = x_new, t_new
    return x

def cerca_lam_discrepanza(risolvi, stato, sol_vuota, chi2_vuota):
    """Scelta di lam per discrepanza, comune a L1 e L1+TV (ogni metodo ha il suo lam).
    risolvi(f, stato) risolve il problema con lam = f lam_max partendo da stato (warm
    start) e restituisce (soluzione, stato finale, chi2). Bersaglio chi^2 = N_validi +
    2 sqrt(2 N_validi) (gate validi).
    Punto di partenza f = 1 (lam = lam_max): immagine vuota (sol_vuota), chi2_vuota =
    ||m_w||^2, nota senza risolvere nulla.
      - chi2_vuota <= bersaglio: i dati sono compatibili con "nessuna inclusione" e la
        discrepanza non sceglie lam -> immagine vuota con f = 1 (f_se_vuota = None)
        oppure ricostruzione a f = f_se_vuota;
      - altrimenti f = 1 e' il primo f rifiutato: griglia fattori_lam finche' chi2 <=
        bersaglio, poi n_bisezioni_lam bisezioni in log(f) fra l'ultimo f rifiutato e il
        primo accettato (anche quando e' accettato gia' il primo f della griglia).
    Restituisce (soluzione, f = lam/lam_max, chi2)."""
    n_att = chi2_bersaglio()
    target = n_att + 2*np.sqrt(2*n_att)
    if chi2_vuota <= target:
        if f_se_vuota is None:
            return sol_vuota, 1.0, chi2_vuota
        sol, _, chi2 = risolvi(f_se_vuota, stato)
        return sol, f_se_vuota, chi2
    # warm start sempre dallo stato dell'ultimo f RIFIUTATO (lam piu' grande, soluzione
    # piu' sparsa): si procede da lam grandi a lam piccoli anche nelle bisezioni,
    # invece di ripartire dalla soluzione piu' densa di f_acc
    f_rif, stato_rif = 1.0, stato
    for f in fattori_lam:
        sol, stato_nuovo, chi2 = risolvi(f, stato_rif)
        if chi2 <= target:
            break
        f_rif, stato_rif = f, stato_nuovo
    else:
        return sol, f, chi2                 # nessun f accettato: il piu' piccolo della griglia
    f_acc, sol_acc, chi2_acc = f, sol, chi2
    for _ in range(n_bisezioni_lam):
        f_mid = np.sqrt(f_rif * f_acc)
        sol_mid, stato_mid, chi2_mid = risolvi(f_mid, stato_rif)
        if chi2_mid <= target:
            f_acc, sol_acc, chi2_acc = f_mid, sol_mid, chi2_mid
        else:
            f_rif, stato_rif = f_mid, stato_mid
    return sol_acc, f_acc, chi2_acc


def ricostruzione_L1(m_w, B, Lip):
    """L1 compensata con lam per discrepanza (cerca_lam_discrepanza); FISTA, warm
    start da x. Restituisce (x, f = lam/lam_max, chi2)."""
    lam_max = np.abs(B.T @ m_w).max()
    def risolvi(f, x0):
        x = fista_l1(B, m_w, f * lam_max, Lip, x0=x0)
        return x, x, float(np.sum((B @ x - m_w)**2))
    vuota = np.zeros(B.shape[1])
    return cerca_lam_discrepanza(risolvi, vuota, vuota, float(m_w @ m_w))


# --- L1 + TV (ADMM) ----------------------------------------------------------------
def matrice_differenze():
    """D_tv sparsa (N_spigoli x N_vox): una riga per ogni coppia di voxel adiacenti lungo
    x, y o z, (D_tv x)_e = x[piu_e] - x[meno_e] (piu = voxel a coordinata maggiore).
    Indici dei voxel j = (ix*Ny + iy)*Nz + iz, come in r_V. 16x16x8 -> 5632 spigoli."""
    idx = np.arange(N_vox).reshape(Nx, Ny, Nz)
    piu, meno = [], []
    for asse in range(3):
        n = idx.shape[asse]
        piu.append(np.take(idx, range(1, n), axis=asse).ravel())
        meno.append(np.take(idx, range(n - 1), axis=asse).ravel())
    piu, meno = np.concatenate(piu), np.concatenate(meno)
    ne = piu.size
    righe = np.r_[np.arange(ne), np.arange(ne)]
    D = sp.csr_matrix((np.r_[np.ones(ne), -np.ones(ne)], (righe, np.r_[piu, meno])),
                      shape=(ne, N_vox))
    return D, piu, meno


D_tv, spigolo_piu, spigolo_meno = matrice_differenze()


def soft(v, soglia):
    """Soglia morbida componente per componente: sign(v) max(|v| - soglia, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - soglia, 0.0)


def admm_l1tv(lam, alpha, Wtm, stato):
    """ADMM per min 1/2||W_w x - m_w||^2 + lam[alpha sum p|x| + (1-alpha) sum q|D x|]
    (variabili globali Minv_tv, peso_tv, peso_spigoli_tv, rho_u_tv, rho_z_tv da
    prepara_pesi). stato = (u, z, a, b) di partenza (warm start). Restituisce u
    (zeri esatti), lo stato finale e il numero di iterazioni.
    Minv_tv e' in float32: il vettore va convertito in float32 PRIMA del prodotto,
    altrimenti NumPy riconverte tutta la matrice in float64 a ogni iterazione."""
    u, z, a, b = (v.copy() for v in stato)
    s1 = lam * alpha * peso_tv / rho_u_tv                 # soglia sui voxel (L1)
    s2 = lam * (1 - alpha) * peso_spigoli_tv / rho_z_tv   # soglia sugli spigoli (TV)
    for it in range(1, n_iter_admm + 1):
        # passo in x: Tikhonov generalizzato, (W^T W + P_u + D^T P_z D) x = ...
        rhs = Wtm + rho_u_tv * (u - a) + D_tv.T @ (rho_z_tv * (z - b))
        x = (Minv_tv @ rhs.astype(np.float32)).astype(np.float64)
        Dx = D_tv @ x
        u_old, z_old = u, z
        u = soft(x + a, s1)                               # passo in u (L1)
        z = soft(Dx + b, s2)                              # passo in z (TV: azzera i salti piccoli)
        a = a + x - u                                     # moltiplicatori scalati
        b = b + Dx - z
        # arresto relativo a ||x||; il minimo scala_min_tv evita di non fermarsi mai
        # quando la soluzione e' l'immagine vuota (x -> 0)
        scala = max(np.linalg.norm(x), scala_min_tv)
        res_primale = np.sqrt(np.sum((x - u)**2) + np.sum((Dx - z)**2))
        scala_z = max(np.linalg.norm(Dx), scala_min_tv)
        if (res_primale <= tol_admm * scala and np.linalg.norm(u - u_old) <= tol_admm * scala
                and np.linalg.norm(z - z_old) <= tol_admm * scala_z):
            break
    return u, (u, z, a, b), it


def ricostruzione_L1TV(m_w, alpha):
    """L1+TV con lam per discrepanza (cerca_lam_discrepanza, stessa procedura della L1
    ma lam proprio), con lam = f lam_max e lam_max = max_j |(W_w^T m_w)_j| / (alpha p_j)
    (maggiorazione: immagine vuota garantita). Warm start di (u, z, a, b) fra un lam
    e il successivo. Restituisce (A, f = lam/lam_max, chi2, iterazioni ADMM totali)."""
    global scala_min_tv
    Wtm = W_w.T @ m_w
    lam_max = np.abs(Wtm / (alpha * peso_tv)).max()
    # scala della soluzione non sparsa (vettore in float32 come in admm_l1tv)
    scala_min_tv = 1e-3 * np.linalg.norm(Minv_tv @ Wtm.astype(np.float32))
    n_it = 0
    def risolvi(f, stato):
        nonlocal n_it
        u, stato, it = admm_l1tv(f * lam_max, alpha, Wtm, stato)
        n_it += it
        return u, stato, float(np.sum((W_w @ u - m_w)**2))
    stato0 = (np.zeros(N_vox), np.zeros(D_tv.shape[0]), np.zeros(N_vox), np.zeros(D_tv.shape[0]))
    u, f, chi2 = cerca_lam_discrepanza(risolvi, stato0, np.zeros(N_vox), float(m_w @ m_w))
    return u, f, chi2, n_it


# --- regione connessa (26-vicinato) -------------------------------------------
# Due voxel sono connessi se si toccano per faccia, spigolo o vertice (cubo 3x3x3).
# NB: senza structure, ndimage.label userebbe solo le 6 facce (regioni piu' piccole).
VICINATO_26 = np.ones((3, 3, 3), bool)

def regione_connessa(candidati, j):
    """Maschera (N_vox,) dei voxel candidati connessi al voxel j."""
    etich, _ = ndimage.label(candidati.reshape(Nx, Ny, Nz), structure=VICINATO_26)
    etich = etich.ravel()
    return etich == etich[j]


# --- metriche ----------------------------------------------------------------
def esito(A_hat, sel, stima, vero, segnale, soglia, mask, r_true):
    """Metriche per l'insieme di voxel 'sel' (la regione). Se il segnale e'
    sotto soglia l'inclusione e' 'non trovata': posizione, errore e stima sono
    vuoti (restano nelle colonne *_grezzo/_grezza)."""
    w = np.abs(A_hat[sel])
    r_bar = (w[:, None] * r_V[sel]).sum(0) / w.sum()
    jj = np.where(sel)[0][np.argmax(w)]
    err = float(np.linalg.norm(r_bar - r_true))
    trovata = bool(segnale >= soglia)
    nan = np.nan
    rap = stima / vero if vero != 0 else nan
    return dict(trovata=trovata, segnale_sigma=float(segnale),
                x_rec=r_bar[0] if trovata else nan, y_rec=r_bar[1] if trovata else nan,
                z_rec=r_bar[2] if trovata else nan,
                x_bar=float(r_bar[0]), y_bar=float(r_bar[1]),
                z_baricentro_grezzo=float(r_bar[2]), z_picco_grezzo=float(r_V[jj][2]),
                errore_baricentro_mm=err if trovata else nan, errore_baricentro_grezzo_mm=err,
                errore_picco_mm=float(np.linalg.norm(r_V[jj] - r_true)) if trovata else nan,
                stima=stima if trovata else nan, stima_grezza=stima, valore_vero=vero,
                rapporto_stima=rap if trovata else nan, rapporto_stima_grezzo=rap,
                segno_corretto=bool(trovata and np.sign(stima) == np.sign(vero)),
                dmu_picco=float(A_hat[jj]),
                voxel_regione=int(sel.sum()), voxel_in_comune=int((sel & mask).sum()))

def esito_vuoto(vero):
    """Metriche per un'immagine tutta a zero (dati compatibili con 'nessuna inclusione'):
    stesse colonne di esito, inclusione non trovata, stima grezza 0, nessun voxel."""
    nan = np.nan
    return dict(trovata=False, segnale_sigma=0.0, x_rec=nan, y_rec=nan, z_rec=nan,
                x_bar=nan, y_bar=nan,
                z_baricentro_grezzo=nan, z_picco_grezzo=nan,
                errore_baricentro_mm=nan, errore_baricentro_grezzo_mm=nan,
                errore_picco_mm=nan, stima=nan, stima_grezza=0.0, valore_vero=vero,
                rapporto_stima=nan, rapporto_stima_grezzo=0.0 if vero != 0 else nan,
                segno_corretto=False, dmu_picco=0.0, voxel_regione=0, voxel_in_comune=0)

def metriche_regione(A_hat, peso, m_w, mask, r_true, dmu_vero, soglia_cl, stima_scelta,
                     soglia=np.inf):
    """Intensita' sulla regione connessa al picco (il cluster). Picco e regione si
    scelgono sull'immagine pesata peso*A_hat (peso = p_j della ricostruzione
    compensata): voxel con peso*A_hat >= soglia_cl del picco, stesso segno, connessi al
    picco. Due stime dell'assorbimento complessivo [mm^-1 mm^3]:
        fit:       a = w_R^T m_w / w_R^T w_R (dmu_a uniforme su R, dai dati), a N_R V_vox
        integrale: V_vox sum_R A_hat_j (valori ricostruiti, nessun fit)
        debias:    beta V_vox sum_R A_hat_j, beta = scala della forma A_hat_R sui dati
    stima = quella indicata da stima_scelta; vero = dmu_vero N_voxel_incl V_vox
    (dmu_vero = dmu_a di OGNI voxel dell'inclusione).
    soglia_cl: numero (soglia fissa) o regola [(z_max, soglia), ...] (adattiva, sulla
    profondita' del picco di p|A|)."""
    vero = float(dmu_vero * mask.sum() * V_vox)
    pa = peso * A_hat
    if not np.any(pa):           # immagine vuota (lam >= lam_max): nessuna regione
        return dict(esito_vuoto(vero), soglia_usata=np.nan, z_picco_comp=np.nan,
                    dmu_medio_equivalente=0.0, dmu_fit_regione=np.nan, beta_debias=np.nan,
                    stima_fit_grezza=0.0, stima_integrale_grezza=0.0,
                    stima_debias_grezza=0.0)
    j = int(np.argmax(np.abs(pa)))
    sgn = np.sign(pa[j]) if pa[j] != 0 else 1.0
    z_pk = float(r_V[j][2])                          # profondita' del picco di p|A|
    if not np.isscalar(soglia_cl):                   # regola adattiva [(z_max, soglia), ...]
        soglia_cl = next(sv for zm, sv in soglia_cl if z_pk <= zm)
    candidati = sgn * pa >= soglia_cl * np.abs(pa[j])
    regione = regione_connessa(candidati, j)
    w_R = W_w[:, regione].sum(axis=1)                # firma della regione nei dati
    # FIT: dmu_a uniforme su R che meglio spiega i dati pesati (gate esclusi: riga di
    # W_w nulla e m_w = 0, non contano)
    a_fit = float((w_R @ m_w) / (w_R @ w_R))
    stima_fit = a_fit * regione.sum() * V_vox                        # [mm^-1 mm^3]
    stima_int = float(A_hat[regione].sum() * V_vox)                  # [mm^-1 mm^3]
    # DEBIAS DI SCALA: stessa forma A_hat_R, ampiezza dai dati (beta >= 0)
    firma = W_w[:, regione] @ A_hat[regione]
    n2 = float(firma @ firma)
    beta = max(float(firma @ m_w) / n2, 0.0) if n2 > 0 else 0.0
    stima_deb = beta * stima_int
    stima = {"fit": stima_fit, "integrale": stima_int, "debias": stima_deb}[stima_scelta]
    # SEGNALE PER "TROVATA" (filtro adattato): quanto i DATI somigliano alla firma
    # della regione w_R, in deviazioni standard del rumore.
    segnale = float(abs(w_R @ m_w) / np.linalg.norm(w_R))
    met = esito(A_hat, regione, stima, vero, segnale, soglia, mask, r_true)
    met["soglia_usata"] = float(soglia_cl)
    met["z_picco_comp"] = z_pk
    met["dmu_fit_regione"] = a_fit           # dmu_a uniforme stimata sul cluster [mm^-1]
    met["stima_fit_grezza"] = stima_fit
    met["stima_integrale_grezza"] = stima_int
    met["beta_debias"] = beta
    met["stima_debias_grezza"] = stima_deb
    # valore medio per voxel equivalente: stima / volume VERO dell'inclusione,
    # direttamente confrontabile con dmu_vero (stesso rapporto stima/vero)
    met["dmu_medio_equivalente"] = stima / (mask.sum() * V_vox)
    return met


def tutte_le_stime(ric, m_w, mask, r_true, dmu, soglie=None):
    """[(metodo, versione, tipo, gamma, soglia_cl, nome_par, par, chi2, chi2_target,
    tempo, metriche)], un elemento per metodo."""
    out = []
    for metodo, versione, tipo, g, s_cl in METODI_DEF:
        sog = np.inf if soglie is None else soglie[metodo]
        A, nome_par, par, chi2, chi2_t, p, t_sol = ric[metodo]
        met = metriche_regione(A, p, m_w, mask, r_true, dmu, s_cl, stima_per_metodo[metodo], sog)
        met["alpha_tv"] = alpha_tv if metodo == "L1TV" else np.nan
        out.append((metodo, versione, tipo, g, s_cl, nome_par, par, chi2, chi2_t, t_sol, met))
    return out

def classe_snr(snr):
    """Classe del SNR atteso, utile come filtro nei pivot: sotto ~7 l'inclusione
    e' sotto il rumore e i rapporti dmu_a stimato/vero non hanno significato."""
    for lim, nome in [(7, "1: <7 (sotto il rumore)"), (20, "2: 7-20 (limite)"),
                      (100, "3: 20-100"), (np.inf, "4: >100")]:
        if snr < lim:
            return nome


# %% =============================================================
# SALVATAGGIO EXCEL (funzione)
# ==================================================================
def salva_excel(righe, percorso):
    """Fogli: 'dati' (formato lungo, una riga per caso e metodo: ideale per i
    pivot), 'confronto' (una riga per caso, metodi affiancati), 'irf', 'spettri',
    'posizioni', 'variazioni', 'configurazione'."""
    df = pd.DataFrame(righe)
    chiavi = ["lambda_nm", "pos_id", "var_id"]
    wide = df.pivot_table(index=chiavi, columns="metodo",
                          values=["trovata", "errore_baricentro_mm", "rapporto_stima",
                                  "rapporto_stima_grezzo", "voxel_in_comune", "segno_corretto"],
                          aggfunc="first")
    wide.columns = [f"{v}_{m}" for v, m in wide.columns]
    info = df.drop_duplicates(chiavi)[chiavi + ["pos_desc", "x_vero", "y_vero", "z_vero",
                                                "var_desc", "d_lip", "d_HHb", "d_O2Hb", "d_tHb",
                                                "SO2_incl", "mu_a0",
                                                "mu_s0", "dmu_a_vero", "dmu_su_mua0",
                                                "SNR_atteso", "classe_SNR",
                                                "sigma_dati_su_sigma0", "compatibile_con_zero"]]
    wide = info.merge(wide.reset_index(), on=chiavi)
    spettri_df = pd.DataFrame({"lambda_nm": wl_nm, "mu_a_bkg [mm^-1]": mu_a_bkg,
                               "mu_s'_bkg [mm^-1]": mu_s_bkg,
                               **{f"dmu_a {v[0]} {v[1]} [mm^-1]": dmu_var[v[0]] for v in variazioni}})
    pos_df = pd.DataFrame([{"pos_id": p[0], "descrizione": p[1], "x": p[2][0], "y": p[2][1],
                            "z": p[2][2]} for p in posizioni])
    var_df = pd.DataFrame([{"var_id": v[0], "descrizione": v[1], "d_lip": v[2],
                            "d_HHb": v[3], "d_O2Hb": v[4], "d_tHb": v[3] + v[4],
                            **{f"incl_{k}": comp_var[v[0]][k]
                               for k in ("HHb", "O2Hb", "tHb", "SO2", "Lipid", "H2O")}}
                           for v in variazioni])
    testo_pesi = ("sigma stimata dai conteggi osservati di ogni misura (metodo delta); "
                  "pesi, SVD e fattorizzazione ADMM ricalcolati per caso"
                  if modello_sigma == "dati" else
                  "sigma_0 dai conteggi attesi di baseline; pesi per lambda")
    testo_pesi += ("; rumore del fondo incluso (conteggi grezzi del gate + stima del fondo "
                   "nella finestra prima del picco)" if con_fondo() else "; senza rumore del fondo")
    conf_df = pd.DataFrame({"parametro": ["composizione background", "scattering",
                                          "N_tot_TPSF", "coppie", "gate", "voxel",
                                          "dmu_a_max (compensazione)",
                                          "TSVD", "Tikhonov", "L1 (FISTA)", "L1+TV (ADMM)",
                                          "seme_rumore", "modello_sigma", "pesi",
                                          "soglie 'trovata' (calibrate)", "stima dell'integrale"],
                            "valore": [str(composizione),
                                       f"Jacques fatty: a={a_jac} mm^-1, b={b_jac}, l0={wl0_jac} nm",
                                       N_tot_TPSF, f"{N_pairs} ({modalita_coppie})",
                                       f"{N_gates} x {gate_width} ns da {t_start} ns dopo il "
                                       f"riferimento '{riferimento_gate}' della IRF "
                                       f"({t_rif_comune:.3f} ns), arrotondati ai canali da "
                                       f"{irf_ps_per_canale} ps; misura con IRF: {misura_con_irf}, "
                                       f"inversione con IRF: {inversione_con_irf}; fondo: media "
                                       f"dei canali in {fondo_finestra_ns} ns che finiscono "
                                       f"{irf_fondo_prima_ns} ns prima del picco della IRF; "
                                       f"file IRF {file_irf}; N_tot nei primi "
                                       f"{N_gates_normalizzazione} gate; rumore del fondo nella "
                                       f"misura: {con_fondo()}",
                                       f"{Nx}x{Ny}x{Nz}, lato {step} mm",
                                       f"{dmu_a_max}",
                                       f"gamma {gamma_tsvd}; k per discrepanza (chi^2 <= N_validi); "
                                       f"soglia adattiva {regola_soglia['TSVD']}",
                                       f"gamma {gamma_tik}; lam per discrepanza (chi^2 = N_validi); "
                                       f"soglia adattiva {regola_soglia['Tikhonov']}",
                                       f"gamma {gamma_L1}; lam per discrepanza: chi^2 = N_validi + "
                                       f"2 sqrt(2 N_validi); partenza da lam/lam_max = 1; se gia' "
                                       f"compatibile con i dati: f = {f_se_vuota}; griglia "
                                       f"{fattori_lam} + {n_bisezioni_lam} bisezioni; tol {tol_fista}, "
                                       f"max {n_iter_max} iterazioni; soglia "
                                       f"{soglia_L1}",
                                       f"min 1/2||W_w x - m||^2 + lam[alpha sum p|x| + (1-alpha) sum "
                                       f"q|Dx|], p = max(||w_j||, 1/dmu_a_max), q = sqrt(p_i p_k); "
                                       f"alpha {alpha_tv}; rho = {rho_rel_tv} p^2, q^2; tol {tol_admm}, "
                                       f"max {n_iter_admm} iterazioni per lam; lam come la L1; "
                                       f"soglia {soglia_L1TV}",
                                       seme_rumore, modello_sigma, testo_pesi,
                                       "trovata se il segnale della regione >= soglia: " + ", ".join(
                                           f"{m} {s:.2f}" for m, s in soglia_trovata.items())
                                       + f" sigma (quantile {quantile_falsi_positivi} del solo rumore)",
                                       "regione connessa al picco di p|A| (>= soglia del picco); "
                                       "p = max(||w_j||, 1/dmu_a_max)^gamma; stima per tutti i metodi: "
                                       + "per metodo -> " + ", ".join(f"{m}: {stima_per_metodo[m]}"
                                                                     for m in METODI)
                                       + "; vero = "
                                       "dmu x N_voxel_incl x V_vox (dmu = valore di ogni voxel "
                                       "dell'inclusione)"]})
    with pd.ExcelWriter(percorso, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="dati", index=False)
        wide.to_excel(xw, sheet_name="confronto", index=False)
        pd.DataFrame(righe_irf).to_excel(xw, sheet_name="irf", index=False)
        spettri_df.to_excel(xw, sheet_name="spettri", index=False)
        pos_df.to_excel(xw, sheet_name="posizioni", index=False)
        var_df.to_excel(xw, sheet_name="variazioni", index=False)
        conf_df.to_excel(xw, sheet_name="configurazione", index=False)
        aggiungi_grafici_spettri(xw.book["spettri"], n_righe=len(spettri_df),
                                 n_var=len(variazioni))
    return df


def aggiungi_grafici_spettri(ws, n_righe, n_var):
    """Grafici Excel nativi nel foglio 'spettri': mu_a e mu_s' di background e
    dmu_a di ogni variazione in funzione di lambda (si aggiornano se si
    modificano i dati del foglio)."""
    from openpyxl.chart import ScatterChart, Reference, Series
    x = Reference(ws, min_col=1, min_row=2, max_row=n_righe + 1)
    def grafico(titolo, y_label, colonne, ancora):
        ch = ScatterChart()
        ch.title, ch.style = titolo, 13
        ch.x_axis.title, ch.y_axis.title = "lambda [nm]", y_label
        ch.x_axis.scaling.min, ch.x_axis.scaling.max = float(wl_nm[0]), float(wl_nm[-1])
        ch.x_axis.delete = False; ch.y_axis.delete = False
        ch.x_axis.tickLblPos = "low"               # etichette di lambda in basso anche se y < 0
        ch.height, ch.width = 8, 16
        for c in colonne:
            y = Reference(ws, min_col=c, min_row=1, max_row=n_righe + 1)
            s = Series(y, x, title_from_data=True)
            s.marker.symbol = "circle"; s.marker.size = 4
            s.smooth = False                       # segmenti fra i punti, nessuna interpolazione
            ch.series.append(s)
        ws.add_chart(ch, ancora)
    grafico("Assorbimento di background", "mu_a [mm^-1]", [2], "H2")
    grafico("Scattering ridotto di background (Jacques)", "mu_s' [mm^-1]", [3], "H19")
    grafico("Variazione di assorbimento dell'inclusione", "dmu_a [mm^-1]",
            list(range(4, 4 + n_var)), "H36")


# %% =============================================================
# PREPARAZIONE PER LAMBDA, PESI e CALIBRAZIONE DELLE SOGLIE SUL RUMORE
# ==================================================================
def prepara_lambda(mu_a0, mu_s0, wl, coppia_curve=None):
    """W dell'inversione, W della misura e conteggi attesi di baseline per una
    lambda. Con modello_sigma = 'baseline' prepara subito anche i pesi.
      W      -> inversione: W_irf (con IRF, fondo sottratto), salvo inversione_con_irf = False
      W_mis  -> misura: M_ideal = W_mis @ A (con IRF e fondo sottratto, salvo
                misura_con_irf = False); lam0 dalle curve della misura."""
    global W, W_mis, lam0, sig0, diag_irf, fondo_canale, n_canali_gate, n_canali_fondo
    global gate_validi
    h, h_netta, info = irf_per_lambda(wl)
    F = matrice_W(mu_a0, mu_s0, h, h_netta, info, coppia_curve)
    # fondo per il rumore della misura (simula_misura, sigma_teorica)
    fondo_canale = F["fondo_cnt"]                        # (N_pairs,) conteggi/canale
    n_canali_gate, n_canali_fondo = F["n_canali_gate"], F["n_canali_fondo"]
    W_mis = F["W_irf"] if misura_con_irf else F["W_id"]
    frac_gates = F["frac_irf"] if misura_con_irf else F["frac_id"]
    W = F["W_irf"] if inversione_con_irf else F["W_id"]
    # Gate senza conteggi attesi (tutti i canali posti a 0 dal vincolo): riga di W
    # nulla e misura neutra (M = 0, vedi misura_valida), cioe' gate escluso di fatto.
    gate_validi = (frac_gates > 0).flatten()
    # anche con inversione_con_irf = False (W_id ha quelle righe non nulle): altrimenti
    # l'inversione cercherebbe W A = 0 nei gate esclusi
    W = W.copy()
    W[~gate_validi, :] = 0.0
    if not gate_validi.all():
        print(f"   {wl:.0f} nm: {np.sum(~gate_validi)} gate su {N_meas} senza conteggi attesi "
              f"dopo la sottrazione del fondo (posti a 0, esclusi): coppie "
              f"{sorted(set((np.where(~gate_validi)[0] // N_gates).tolist()))}")
    diag_irf = dict(lambda_nm=wl, wl_irf=info["wl_irf"], fondo_irf=info["fondo"],
                    conteggi_netti_irf=info["conteggi_netti"], fwhm_irf_ps=info["fwhm_ps"],
                    t_picco_irf_ns=info["t_picco"], t_salita_irf_ns=info["t_salita"],
                    t_rif_gate_ns=info["t_rif"], canale_rif=info["k_rif"],
                    fondo_conteggi_canale=float(np.median(F["fondo_cnt"])),
                    residuo_fondo_mediano=float(np.nanmedian(F["residuo"])),
                    residuo_fondo_max=float(np.nanmax(np.abs(F["residuo"]))),
                    residuo_fondo_ultimo_gate=float(np.nanmedian(F["residuo"][:, -1])),
                    canali_azzerati_tot=int(F["n_canali_azzerati"].sum()),
                    gate_esclusi=int(np.sum(~gate_validi)))
    diag_irf["_info"], diag_irf["_curve"] = info, F["curve"]
    lam0 = N_tot_TPSF * frac_gates                       # conteggi attesi di baseline
    # sigma dai conteggi attesi (M = 0): sqrt(2/(lam0+1)) senza rumore del fondo
    sig0 = sigma_teorica(lam0, 0.0).flatten()
    # fondo nell'ultimo gate rispetto al segnale (mediana sulle coppie): > 1 -> gate
    # dominato dal fondo, prolungare la finestra serve a poco
    ng_u = n_canali_gate[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        diag_irf["fondo_su_segnale_ultimo_gate"] = float(np.nanmedian(
            np.where(lam0[:, -1] > 0, fondo_canale * ng_u / lam0[:, -1], np.nan)))
    if modello_sigma == "baseline":
        prepara_pesi(sig0)


tempi_prep = {m: 0.0 for m in METODI}   # [s] preparazione delle matrici (ultima prepara_pesi)


def prepara_pesi(sig):
    """W_w = W/sigma e le matrici di ciascun metodo (dipendono solo da W_w):
      TSVD/Tikhonov  per ogni gamma: p_j e SVD di W_w diag(1/p) (modi con s_i > 0)
      L1             p_j, B = W_w diag(1/p) e ||B||^2 (passo di FISTA)
      L1+TV          p_j e q_e, penalita' rho_u, rho_z e Minv_tv
    tempi_prep: tempo di preparazione per metodo (la SVD di gamma condivisa e' contata
    per il metodo che la usa)."""
    global W_w, svd_comp, L1_comp, peso_tv, peso_spigoli_tv, rho_u_tv, rho_z_tv, Minv_tv
    W_w = W / sig[:, None]
    p_base = np.maximum(np.linalg.norm(W_w, axis=0), 1 / dmu_a_max)   # max(||w_j||, 1/dmu_a_max)
    svd_comp, t_svd = {}, {}
    for g in gamma_svd:
        t0 = time.perf_counter()
        p = p_base**g
        U, s, Vt = np.linalg.svd(W_w / p, full_matrices=False)
        # solo i modi con s_i > 0: le righe nulle dei gate esclusi danno s_i = 0 e
        # dividerebbero per zero se la discrepanza non si raggiunge prima
        k_pos = int(np.sum(s > s[0] * np.finfo(float).eps * max(W_w.shape)))
        svd_comp[g] = (p, U[:, :k_pos], s[:k_pos], Vt[:k_pos])
        t_svd[g] = time.perf_counter() - t0
    tempi_prep["TSVD"], tempi_prep["Tikhonov"] = t_svd[gamma_tsvd], t_svd[gamma_tik]
    t0 = time.perf_counter()
    p = p_base**gamma_L1
    B = W_w / p
    L1_comp = (p, B, norma2_quadra(B))
    tempi_prep["L1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    peso_tv = p_base                                                  # p_j (gamma = 1)
    peso_spigoli_tv = np.sqrt(peso_tv[spigolo_piu] * peso_tv[spigolo_meno])   # q_e
    rho_u_tv = rho_rel_tv * peso_tv**2
    rho_z_tv = rho_rel_tv * peso_spigoli_tv**2
    K = (W_w.T @ W_w + np.diag(rho_u_tv)
         + (D_tv.T @ sp.diags(rho_z_tv) @ D_tv).toarray())
    # K definita positiva (rho_u > 0). Inversa calcolata in float64 e MEMORIZZATA in
    # float32: meta' memoria letta a ogni iterazione ADMM, errore relativo ~1e-6 sul
    # passo in x (rumore dei dati ~1e-3), verificato anche sui voxel profondi
    Minv_tv = cho_solve(cho_factor(K), np.eye(N_vox)).astype(np.float32)
    tempi_prep["L1TV"] = time.perf_counter() - t0


def pesi_misura(sig_hat):
    """Sigma con cui pesare questa misura. Con modello_sigma = 'dati' ricalcola i pesi."""
    if modello_sigma == "dati":
        prepara_pesi(sig_hat)
        return sig_hat
    return sig0


iter_admm = 0      # iterazioni ADMM dell'ultima ricostruzione L1+TV (diagnostica)


def ricostruisci_tutto(m_w):
    """I quattro metodi sugli stessi dati pesati m_w:
    {metodo: (A_hat, nome_par, par, chi2, chi2_bersaglio, peso p_j, tempo [s])}.
    Il tempo e' quello della sola soluzione (la preparazione e' in tempi_prep)."""
    global iter_admm
    ric = {}
    n_att = chi2_bersaglio()
    target_L1 = n_att + 2*np.sqrt(2*n_att)
    # --- TSVD ---------------------------------------------------------------------
    t0 = time.perf_counter()
    p, U, s, Vt = svd_comp[gamma_tsvd]
    x, k, chi2 = tsvd_disc(U.T @ m_w, m_w, s, Vt, n_att)
    ric["TSVD"] = (x / p, "k", k, chi2, n_att, p, time.perf_counter() - t0)
    # --- Tikhonov -----------------------------------------------------------------
    t0 = time.perf_counter()
    p, U, s, Vt = svd_comp[gamma_tik]
    x, lam, chi2 = tikhonov_disc(U.T @ m_w, m_w, s, Vt, n_att)
    ric["Tikhonov"] = (x / p, "lam/s0", lam / s[0], chi2, n_att, p, time.perf_counter() - t0)
    # --- L1 -----------------------------------------------------------------------
    t0 = time.perf_counter()
    p, B, Lip = L1_comp
    x, f, chi2 = ricostruzione_L1(m_w, B, Lip)
    ric["L1"] = (x / p, "lam/lam_max", f, chi2, target_L1, p, time.perf_counter() - t0)
    # --- L1 + TV ------------------------------------------------------------------
    t0 = time.perf_counter()
    A, f, chi2, iter_admm = ricostruzione_L1TV(m_w, alpha_tv)
    ric["L1TV"] = (A, "lam/lam_max", f, chi2, target_L1, peso_tv, time.perf_counter() - t0)
    return ric


def calibra_soglie():
    """Segnale 'trovato' da ogni metodo su misure di SOLO rumore (dmu_a = 0): la
    soglia e' il suo quantile_falsi_positivi."""
    rng_cal = np.random.default_rng(seme_rumore + 1000)
    pid0 = posizioni[0][0]
    segnali = {m: [] for m in METODI}
    for wl_c in lambda_calibrazione:
        i = int(np.argmin(np.abs(wl_nm - wl_c)))
        prepara_lambda(float(mu_a_bkg[i]), float(mu_s_bkg[i]), wl_nm[i])
        if calibrazione_veloce_sig0 and modello_sigma == "dati":
            prepara_pesi(sig0)
        for _ in range(n_realizzazioni_rumore):
            M0, sig_hat = simula_misura(lam0, np.zeros_like(lam0), rng_cal)
            M0 = misura_valida(M0)
            sig = sig0 if calibrazione_veloce_sig0 else pesi_misura(sig_hat.flatten())
            m_w = M0.flatten() / sig
            for metodo, *_, met in tutte_le_stime(
                    ricostruisci_tutto(m_w), m_w, maschere[pid0], np.zeros(3), 0.0):
                segnali[metodo].append(met["segnale_sigma"])
    soglie = {m: max(float(np.quantile(v, quantile_falsi_positivi)), soglia_minima_sigma)
              for m, v in segnali.items()}
    # L1+TV: non sotto la soglia della L1 (sul solo rumore resta quasi sempre vuota e la
    # calibrazione da sola darebbe una soglia troppo permissiva)
    soglie["L1TV"] = max(soglie["L1TV"], soglie["L1"])
    return soglie, segnali


# %% =============================================================
# CONTROLLO DELLA IRF E DELLA MISURA SIMULATA (tabella per lambda e grafico)
# ==================================================================
_, _, _inf = irf_per_lambda(wl_nm[0])
_ng = np.array([i1 - i0 for i0, i1 in _inf["gate_idx"]])
print("="*78)
print(f"IRF: {file_irf}: {irf_grezze.shape[0]} curve x {K_canali} canali da "
      f"{irf_ps_per_canale} ps (finestra {K_canali * dc:.3f} ns)")
print(f"Gate: {N_gates} x {gate_width} ns = {gate_width / dc:.2f} canali -> {_ng.min()}-{_ng.max()} "
      f"canali, dal canale {_inf['gate_idx'][0][0]} ({_inf['bordi_gate'][0]:.3f} ns) al canale "
      f"{_inf['gate_idx'][-1][1]} ({_inf['bordi_gate'][-1]:.3f} ns); t' = 0 della W senza IRF "
      f"al canale {_inf['k_rif']}")
print(f"Misura con IRF: {misura_con_irf}   inversione con IRF: {inversione_con_irf}")
print(f"{'lambda':>7s} {'fondoIRF':>9s} {'FWHM':>6s} {'t_salita':>9s} {'t_picco':>8s}")
for wl in wl_nm:
    _, _, inf = irf_per_lambda(wl)
    print(f"{wl:7.0f} {inf['fondo']:9.2f} {inf['fwhm_ps']:4.0f}ps {inf['t_salita']:7.3f}ns "
          f"{inf['t_picco']:6.3f}ns")
print("="*78 + "\n")

grafico_controllo = True     # grafico della DTOF simulata (fondo stimato e sottratto)
wl_grafico = 800.0
coppia_grafico = 0
if grafico_controllo:
    _i = int(np.argmin(np.abs(wl_nm - wl_grafico)))
    prepara_lambda(float(mu_a_bkg[_i]), float(mu_s_bkg[_i]), wl_nm[_i], coppia_grafico)
    C, inf = diag_irf["_curve"], diag_irf["_info"]
    tg = t_s[:C["n_c"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a_, logy in zip(ax, (True, False)):
        for g, (b0_, b1_) in enumerate(zip(inf["bordi_gate"][:-1], inf["bordi_gate"][1:])):
            a_.axvspan(b0_, b1_, color="0.85" if g % 2 else "0.93", zorder=0)
        a_.axvspan(tg[C["k_f"][0]], tg[C["k_f"][-1]], color="C1", alpha=0.15,
                   label="finestra del fondo")
        a_.plot(tg, C["conv0"], color="0.4", lw=1, label="fluenza (x) IRF grezza")
        a_.axhline(C["fondo"], color="C1", ls="--", lw=1,
                   label=f"fondo stimato {C['fondo']:.2f} conteggi/canale")
        a_.plot(tg, C["conv0_n"], color="C3", lw=1.4, label="fondo sottratto (misura)")
        a_.plot(tg, C["conv0_vero"], color="k", lw=1, ls=":", label="fluenza (x) IRF senza fondo")
        a_.plot(tg, C["ideale"], color="C0", lw=1, ls="--", label="senza IRF (W dell'inversione)")
        a_.set(xlabel="t strumento [ns]", ylabel="conteggi attesi per canale")
        a_.grid(alpha=0.3, ls=":")
        if logy:
            a_.set_yscale("log")
            a_.set_ylim(max(C["fondo"] * 0.1, 1e-3), 3 * C["conv0"].max())
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{wl_nm[_i]:.0f} nm, coppia {coppia_grafico} (rho = "
                    f"{rho_pairs[coppia_grafico]:.0f} mm), N_tot = {N_tot_TPSF:.0e}")
    ax[1].set_xlim(tg[C["k_f"][0]], inf["bordi_gate"][3])
    ax[1].set_ylim(-3 * max(C["fondo"], 1e-3), 30 * max(C["fondo"], 1e-3))
    ax[1].set_title("zoom sulla finestra del fondo (scala lineare)")
    plt.tight_layout()
    plt.show()
    print(f"Fondo stimato a {wl_nm[_i]:.0f} nm: mediana {diag_irf['fondo_conteggi_canale']:.2f} "
          f"conteggi/canale; fondo rimasto nei gate dopo la sottrazione (rispetto alla curva "
          f"senza fondo): mediana {100*diag_irf['residuo_fondo_mediano']:+.2f}%, ultimo gate "
          f"{100*diag_irf['residuo_fondo_ultimo_gate']:+.2f}%, max {100*diag_irf['residuo_fondo_max']:.2f}%\n")


t_cal = time.time()
soglia_trovata, segnali_nulli = calibra_soglie()
print("="*78)
print(f"CALIBRAZIONE SULLE MISURE DI SOLO RUMORE ({len(lambda_calibrazione)} lambda x "
      f"{n_realizzazioni_rumore} realizzazioni, modello_sigma = '{modello_sigma}', "
      f"{time.time() - t_cal:.0f} s)")
for m, s_ in soglia_trovata.items():
    v = np.array(segnali_nulli[m])
    print(f"   {m:9s} segnale da solo rumore: mediana {np.median(v):4.1f}, max {v.max():4.1f} "
          f"sigma -> soglia 'trovata' = {s_:.2f} sigma")
print("="*78 + "\n")


# %% =============================================================
# CICLO: lambda (esterno) x posizione x variazione (interno)
# ==================================================================
rng = np.random.default_rng(seme_rumore)
righe = []
righe_irf = []
t_inizio = time.time()
n_fatti = 0

for i_wl, wl in enumerate(wl_nm):
    mu_a0, mu_s0 = float(mu_a_bkg[i_wl]), float(mu_s_bkg[i_wl])

    # --- una volta per lambda: W e conteggi attesi (e pesi, se 'baseline') --
    prepara_lambda(mu_a0, mu_s0, wl)
    diag = {k: v for k, v in diag_irf.items() if not k.startswith("_")}
    righe_irf.append(diag)

    print("#"*78)
    print(f"# LAMBDA = {wl:.0f} nm ({i_wl+1}/{N_wl}): mu_a0 = {mu_a0:.4f} mm^-1, "
          f"mu_s' = {mu_s0:.3f} mm^-1, conteggi di baseline per gate "
          f"{lam0.min():.0f}-{lam0.max():.0f}")
    print(f"# IRF {diag['wl_irf']:.0f} nm, FWHM {diag['fwhm_irf_ps']:.0f} ps; fondo stimato "
          f"{diag['fondo_conteggi_canale']:.2f} conteggi/canale; fondo rimasto nei gate "
          f"{100*diag['residuo_fondo_mediano']:+.2f}% (ultimo gate "
          f"{100*diag['residuo_fondo_ultimo_gate']:+.2f}%); canali posti a 0 "
          f"{diag['canali_azzerati_tot']}, gate esclusi {diag['gate_esclusi']}; fondo/segnale "
          f"nell'ultimo gate (mediana) {diag['fondo_su_segnale_ultimo_gate']:.2f}")
    print("#"*78)

    for pid, pdesc, r_pos in posizioni:
        mask = maschere[pid]
        r_true = np.array(r_pos)
        for vid, vdesc, d_lip, d_HHb, d_O2Hb in variazioni:
            dmu = float(dmu_var[vid][i_wl])
            A = mask * dmu
            M_ideal = (W_mis @ A).reshape(N_pairs, N_gates)  # misura: con IRF, fondo sottratto
            M_noisy, sig_hat = simula_misura(lam0, M_ideal, rng)
            M_noisy = misura_valida(M_noisy)
            sig = pesi_misura(sig_hat.flatten())         # 'dati': pesi ricalcolati qui
            m_w = M_noisy.flatten() / sig

            # SNR atteso: norma del segnale ideale divisa per la sigma TEORICA
            # (conteggi attesi, include l'effetto dell'inclusione sul rumore)
            sig_teo = sigma_teorica(lam0, M_ideal).flatten()
            SNR = float(np.linalg.norm(M_ideal.flatten() / sig_teo))

            base = dict(lambda_nm=wl, pos_id=pid, pos_desc=pdesc,
                        x_vero=r_pos[0], y_vero=r_pos[1], z_vero=r_pos[2],
                        var_id=vid, var_desc=vdesc, d_lip=d_lip, d_HHb=d_HHb, d_O2Hb=d_O2Hb,
                        d_tHb=d_HHb + d_O2Hb, SO2_incl=comp_var[vid]["SO2"],
                        mu_a0=mu_a0, mu_s0=mu_s0, dmu_a_vero=dmu,
                        dmu_su_mua0=dmu / mu_a0, SNR_atteso=SNR,
                        classe_SNR=classe_snr(SNR),
                        sigma_dati_su_sigma0=float(np.median(sig_hat.flatten() / sig0)),
                        # dati compatibili con l'immagine vuota: lam di L1 e L1+TV non
                        # scelto dalla discrepanza (vedi f_se_vuota)
                        chi2_immagine_vuota=float(m_w @ m_w),
                        compatibile_con_zero=bool(m_w @ m_w <= chi2_bersaglio()
                                                  + 2*np.sqrt(2*chi2_bersaglio())),
                        misura_con_irf=misura_con_irf, inversione_con_irf=inversione_con_irf,
                        fwhm_irf_ps=diag["fwhm_irf_ps"],
                        fondo_conteggi_canale=diag["fondo_conteggi_canale"],
                        residuo_fondo_mediano=diag["residuo_fondo_mediano"],
                        N_gates=N_gates, rumore_fondo=con_fondo(),
                        fondo_su_segnale_ultimo_gate=diag["fondo_su_segnale_ultimo_gate"])
            ric = ricostruisci_tutto(m_w)
            stampe_caso = []
            for (metodo, versione, tipo, gamma, s_cl, nome_par, par, chi2_fin, chi2_t,
                 t_sol, met) in tutte_le_stime(ric, m_w, mask, r_true, dmu, soglia_trovata):
                righe.append(dict(base, metodo=metodo, versione=versione, tipo=tipo,
                                  gamma=gamma,
                                  soglia_cl=float(s_cl) if np.isscalar(s_cl) else np.nan,
                                  metodo_versione=f"{metodo} {versione}",
                                  grandezza="integrale [mm^-1 mm^3]",
                                  parametro_reg=nome_par, valore_reg=par,
                                  chi2_finale=chi2_fin, chi2_atteso=chi2_t,
                                  tempo_soluzione_s=t_sol,
                                  tempo_preparazione_s=tempi_prep[metodo],
                                  iterazioni_admm=iter_admm if metodo == "L1TV" else np.nan,
                                  **met))
                if met["trovata"]:
                    esito_txt = (f"errore {met['errore_baricentro_mm']:5.1f} mm  "
                                 f"stima/vero {met['rapporto_stima']:6.2f}")
                else:
                    esito_txt = "NON TROVATA" + " "*21
                testo = (f"     {metodo:8s} {nome_par}={par:<9.3g} soglia {met['soglia_usata']:.2f}  "
                         f"{esito_txt}  segnale {met['segnale_sigma']:7.1f} sigma"
                         f"  voxel {met['voxel_regione']:3d} (in comune "
                         f"{met['voxel_in_comune']}/{int(mask.sum())})  {t_sol:5.2f} s")
                stampe_caso.append(testo)

            # --- stampa compatta del confronto ---------------------------------
            n_fatti += 1
            print(f"{wl:.0f} nm | {pid} {r_pos} | {vid} {vdesc:38s} | dmu_a = {dmu:+.2e} "
                  f"({100*dmu/mu_a0:+.0f}% di mu_a0) | SNR atteso {SNR:7.1f}")
            for testo in stampe_caso:
                print(testo)
            print(f"     iterazioni ADMM (L1+TV): {iter_admm}")

    # --- salvataggio dopo ogni lambda (lo scan resta salvato se si interrompe)
    salva_excel(righe, file_risultati)
    trascorso = time.time() - t_inizio
    stima = trascorso / n_fatti * (N_casi - n_fatti)
    print(f"\n>>> {n_fatti}/{N_casi} casi, {trascorso/60:.1f} min trascorsi, "
          f"~{stima/60:.0f} min rimanenti. Salvato in '{file_risultati}'.\n")


# %% =============================================================
# RIEPILOGO FINALE: confronto fra i quattro metodi
# ==================================================================
df = salva_excel(righe, file_risultati)
print("="*78)
print(f"SCAN COMPLETATO in {(time.time() - t_inizio)/60:.1f} min "
      f"(modello_sigma = '{modello_sigma}'). Risultati in '{file_risultati}'")
print("="*78)
pd.set_option("display.width", 160)
# mediane solo sui casi con SNR atteso >= 20: sotto, l'inclusione e' confusa col
# rumore e i rapporti stimato/vero non hanno significato (tutti i casi sono
# comunque nel file Excel, filtrabili con la colonna classe_SNR)
ril = df[df["SNR_atteso"] >= 20].copy()
n_righe_caso = len(METODI)
print(f"\nCasi con SNR atteso >= 20: {ril.shape[0] // n_righe_caso} su {df.shape[0] // n_righe_caso}")

def per_metodo(tab):
    """Righe nell'ordine TSVD, Tikhonov, L1, L1TV."""
    return tab.reindex([m for m in METODI if m in tab.index])

print("\nRapporto mediano stima / vero (per posizione, SNR atteso >= 20):")
print(per_metodo(ril.pivot_table(index="metodo", columns="pos_id", values="rapporto_stima",
                                 aggfunc="median")).round(2))
print("\nErrore mediano sul baricentro [mm] (SNR atteso >= 20):")
print(per_metodo(ril.pivot_table(index="metodo", columns="pos_id", values="errore_baricentro_mm",
                                 aggfunc="median")).round(1))
print("\nVoxel della regione in comune con l'inclusione (mediana su 8, SNR atteso >= 20):")
print(per_metodo(ril.pivot_table(index="metodo", columns="pos_id", values="voxel_in_comune",
                                 aggfunc="median")).round(1))
print("\nFrazione di casi con inclusione TROVATA (tutti i casi):")
print(per_metodo(df.pivot_table(index="metodo", columns="pos_id", values="trovata",
                                aggfunc="mean")).round(2))
print("\nNumero di casi per classe di SNR atteso:")
print((df.groupby("classe_SNR").size() // n_righe_caso).to_string())
print(f"\nRapporto mediano sigma stimata dai dati / sigma_0 di baseline: "
      f"{df['sigma_dati_su_sigma0'].median():.3f}")

# --- Riepilogo per inclusione (posizione x variazione): rapporto GREZZO (non dipende
# da 'trovata'), SNR >= 20
r = ril["rapporto_stima_grezzo"]
ril["entro_20"] = (r - 1).abs() <= 0.2
# segno sbagliato o stima nulla (r <= 0) NON vanno scartati (la mediana ignorerebbe i NaN
# e premierebbe i metodi che falliscono): ricevono la penalita' massima ln(10), cioe'
# "errore di un fattore >= 10"; anche |ln r| e' limitato a ln(10)
abs_ln_r_max = np.log(10)
ril["abs_ln_r"] = np.where(r > 0, np.minimum(np.abs(np.log(r.where(r > 0))), abs_ln_r_max),
                           abs_ln_r_max)
ril["segno_ok"] = r > 0
riep = (ril.groupby(["metodo", "pos_id", "var_id"])
        .agg(n_lambda=("rapporto_stima_grezzo", "size"),
             rapporto_mediano=("rapporto_stima_grezzo", "median"),
             q25=("rapporto_stima_grezzo", lambda s: s.quantile(0.25)),
             q75=("rapporto_stima_grezzo", lambda s: s.quantile(0.75)),
             frazione_entro_20=("entro_20", "mean"),
             abs_ln_r_mediano=("abs_ln_r", "median"),
             segno_corretto=("segno_ok", "mean"),
             errore_baricentro_mm=("errore_baricentro_grezzo_mm", "median"),
             voxel_regione=("voxel_regione", "median"),
             voxel_in_comune=("voxel_in_comune", "median"))
        .reset_index())
trov = (df.groupby(["metodo", "pos_id", "var_id"])["trovata"].mean()
        .rename("frazione_trovata_tutte_lambda").reset_index())
riep = riep.merge(trov, on=["metodo", "pos_id", "var_id"])

# --- CONFRONTO FINALE: una riga per metodo -----------------------------------------
# accuratezza della stima (SNR >= 20, media sulle inclusioni), localizzazione,
# rivelazione (tutti i casi) e costo (tempo per caso = preparazione + soluzione)
confronto = (riep.groupby("metodo")
             .agg(abs_ln_r_medio=("abs_ln_r_mediano", "mean"),
                  abs_ln_r_peggiore=("abs_ln_r_mediano", "max"),
                  frazione_entro_20_media=("frazione_entro_20", "mean"),
                  frazione_entro_20_minima=("frazione_entro_20", "min"),
                  segno_corretto_minimo=("segno_corretto", "min"),
                  errore_baricentro_mm_medio=("errore_baricentro_mm", "mean"),
                  voxel_in_comune_medio=("voxel_in_comune", "mean"),
                  voxel_regione_medio=("voxel_regione", "mean")))
confronto["frazione_trovata"] = df.groupby("metodo")["trovata"].mean()
confronto["tempo_medio_per_caso_s"] = (df.assign(t=df["tempo_soluzione_s"]
                                                 + df["tempo_preparazione_s"])
                                       .groupby("metodo")["t"].mean())
confronto["soglia_trovata_sigma"] = pd.Series(soglia_trovata)
confronto = per_metodo(confronto).reset_index()

print("\n" + "="*78)
print("CONFRONTO FRA I QUATTRO METODI (versione migliore di ciascuno)")
print("   abs_ln_r: |ln(stima/vero)| (0 = stima esatta; 0.18 = +-20%; segno sbagliato o stima "
      "nulla = ln(10) = 2.30), media sulle inclusioni")
print("="*78)
print(confronto.round(3).to_string(index=False))
print("\nRapporto mediano stima/vero GREZZO per inclusione (SNR atteso >= 20):")
print(per_metodo(riep.pivot_table(index="metodo", columns=["pos_id", "var_id"],
                                  values="rapporto_mediano")).round(2).to_string())
print("\nFrazione di lambda entro +-20% per inclusione (SNR atteso >= 20):")
print(per_metodo(riep.pivot_table(index="metodo", columns=["pos_id", "var_id"],
                                  values="frazione_entro_20")).round(2).to_string())
with pd.ExcelWriter(file_risultati, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as xw:
    confronto.to_excel(xw, sheet_name="confronto_metodi", index=False)
    riep.to_excel(xw, sheet_name="riepilogo_inclusioni", index=False)
print(f"\nFogli 'confronto_metodi' e 'riepilogo_inclusioni' aggiunti a '{file_risultati}'.")


# %% =============================================================
# STIMA DEI CROMOFORI DALLO SPETTRO RICOSTRUITO (unmixing spettrale)
# ==================================================================
# Per ogni metodo, posizione e variazione si ha uno spettro ricostruito dmu_a(lambda),
# cioe' l'integrale spaziale di ogni lambda diviso per il volume dell'inclusione
# (colonna 'dmu_medio_equivalente' = V_vox sum_R A_hat / (N_voxel_incl V_vox)). Lo
# spettro e' una combinazione lineare degli spettri specifici dei cromofori:
#     dmu_a(lambda) = 0.1 [d_HHb eps_HHb + d_O2Hb eps_O2Hb + d_lip mua_lip + d_H2O mua_H2O]
# (stessa formula di mua_background, [mm^-1]) e le variazioni si stimano ai minimi
# quadrati su tutte le lambda selezionate:
#     min_c || E c - dmu_a_ric ||^2,   E[lambda, k] = 0.1 * spettro_k(lambda)
#   vincolo_lipide_acqua = True  -> incognite d_HHb, d_O2Hb, d_lip con d_H2O = -d_lip
#                                   (stessa ipotesi delle variazioni simulate; 3 incognite)
#   vincolo_lipide_acqua = False -> 4 incognite indipendenti (d_H2O libera)
# Lambda usate: con solo_lambda_trovate = True solo quelle in cui il metodo ha 'trovato'
# l'inclusione (criterio basato sui dati, disponibile anche con misure reali); le altre
# hanno una stima dominata dal rumore. Servono almeno (incognite + 1) lambda.
# NB: dividere per il volume VERO dell'inclusione e' possibile solo in simulazione; con
# dati reali si ottengono gli integrali di concentrazione (colonne *_integrale, in
# uM mm^3 e mm^3 per le frazioni), da dividere per un volume stimato.
# Riga di controllo "VERO": lo stesso unmixing sullo spettro vero dmu_a_vero deve
# restituire le variazioni simulate (verifica della base spettrale).
# Coerenza spaziale: il cluster e' cercato in modo indipendente a ogni lambda e a basso
# SNR puo' saltare su un artefatto lontano. Si escludono le lambda il cui baricentro
# dista piu' di tol_coerenza_mm dalla posizione di riferimento (mediana dei baricentri
# della meta' delle lambda con segnale_sigma piu' alto: criterio basato sui dati).
# None = nessun controllo.
vincolo_lipide_acqua = True
solo_lambda_trovate = True
tol_coerenza_mm = 8.0


def base_spettrale(wl, vincolo=vincolo_lipide_acqua):
    """Matrice E (N_lambda, N_cromofori) [mm^-1 per uM o per frazione] e nomi delle
    incognite. Con vincolo la colonna del lipide e' (mua_lip - mua_H2O)."""
    L = spettri["Lambda"]
    s = {k: np.interp(wl, L, spettri[k]) / 10 for k in ("HHb", "O2Hb", "Lipid", "H2O")}
    if vincolo:
        return np.column_stack((s["HHb"], s["O2Hb"], s["Lipid"] - s["H2O"])), \
               ["d_HHb", "d_O2Hb", "d_lip"]
    return np.column_stack((s["HHb"], s["O2Hb"], s["Lipid"], s["H2O"])), \
           ["d_HHb", "d_O2Hb", "d_lip", "d_H2O"]


def stima_cromofori(wl, dmu_ric, vincolo=vincolo_lipide_acqua, comp_bkg=composizione):
    """Variazioni dei cromofori dallo spettro di assorbimento ricostruito.
    wl [nm] (N,), dmu_ric [mm^-1] (N,): spettro dmu_a dell'inclusione.
    Restituisce un dict con d_HHb, d_O2Hb [uM], d_tHb [uM], d_lip, d_H2O [-], SO2
    dell'inclusione, residuo relativo del fit e numero di condizione della base."""
    wl, dmu_ric = np.asarray(wl, float), np.asarray(dmu_ric, float)
    E, nomi = base_spettrale(wl, vincolo)
    ok = np.isfinite(dmu_ric)
    if ok.sum() < E.shape[1] + 1:
        return dict({n: np.nan for n in ("d_HHb", "d_O2Hb", "d_tHb", "d_lip", "d_H2O",
                                         "SO2_incl_stimata", "residuo_relativo",
                                         "numero_condizione")}, n_lambda_usate=int(ok.sum()))
    c, *_ = np.linalg.lstsq(E[ok], dmu_ric[ok], rcond=None)
    out = dict(zip(nomi, c))
    if vincolo:
        out["d_H2O"] = -out["d_lip"]
    out["d_tHb"] = out["d_HHb"] + out["d_O2Hb"]
    hhb, o2hb = comp_bkg["HHb"] + out["d_HHb"], comp_bkg["O2Hb"] + out["d_O2Hb"]
    out["SO2_incl_stimata"] = o2hb / (hhb + o2hb) if hhb + o2hb > 0 else np.nan
    res = E[ok] @ c - dmu_ric[ok]
    out["residuo_relativo"] = float(np.linalg.norm(res) / np.linalg.norm(dmu_ric[ok]))
    out["numero_condizione"] = float(np.linalg.cond(E[ok]))
    out["n_lambda_usate"] = int(ok.sum())
    return out


def tabella_cromofori(df, solo_trovate=solo_lambda_trovate):
    """Una riga per metodo x posizione x variazione (piu' la riga 'VERO' per ogni
    posizione e variazione): variazioni stimate, vere ed errori."""
    var_info = {v[0]: dict(d_lip=v[2], d_HHb=v[3], d_O2Hb=v[4]) for v in variazioni}
    righe_c = []
    for (metodo, pid, vid), g in df.groupby(["metodo", "pos_id", "var_id"]):
        g = g.sort_values("lambda_nm")
        sel = g["trovata"].to_numpy(bool) if solo_trovate else np.ones(len(g), bool)
        # coerenza spaziale fra le lambda (baricentri grezzi: definiti anche se non trovata)
        n_scartate = 0
        if tol_coerenza_mm is not None and sel.sum() >= 2:
            xyz = g[["x_bar", "y_bar", "z_baricentro_grezzo"]].to_numpy(float)
            seg = g["segnale_sigma"].to_numpy(float)
            forti = sel & (seg >= np.median(seg[sel]))
            r_ref = np.nanmedian(xyz[forti], axis=0)
            lontane = np.linalg.norm(xyz - r_ref, axis=1) > tol_coerenza_mm
            n_scartate = int((sel & lontane).sum())
            sel = sel & ~lontane
        V_incl = maschere[pid].sum() * V_vox
        dmu_ric = np.where(sel, g["stima_grezza"].to_numpy(float) / V_incl, np.nan)
        casi = [(metodo, dmu_ric)]
        if metodo == METODI[0]:     # controllo: unmixing dello spettro VERO (una volta)
            casi.append(("VERO", g["dmu_a_vero"].to_numpy(float)))
        for nome, spettro in casi:
            st = stima_cromofori(g["lambda_nm"].to_numpy(float), spettro)
            vi = var_info[vid]
            riga = dict(metodo=nome, pos_id=pid, var_id=vid, **st,
                        lambda_scartate_coerenza=n_scartate if nome != "VERO" else 0,
                        d_HHb_vero=vi["d_HHb"], d_O2Hb_vero=vi["d_O2Hb"],
                        d_tHb_vero=vi["d_HHb"] + vi["d_O2Hb"], d_lip_vero=vi["d_lip"],
                        d_H2O_vero=-vi["d_lip"], SO2_incl_vera=comp_var[vid]["SO2"],
                        volume_incl_mm3=V_incl)
            for k in ("d_HHb", "d_O2Hb", "d_tHb", "d_lip", "d_H2O"):
                riga[f"errore_{k}"] = riga[k] - riga[f"{k}_vero"]
                riga[f"{k}_integrale"] = riga[k] * V_incl     # [uM mm^3] o [mm^3]
            righe_c.append(riga)
    tab = pd.DataFrame(righe_c)
    ordine = {m: i for i, m in enumerate(("VERO",) + METODI)}
    return tab.sort_values(["pos_id", "var_id", "metodo"],
                           key=lambda s: s.map(ordine) if s.name == "metodo" else s
                           ).reset_index(drop=True)


crom = tabella_cromofori(df)
print("\n" + "="*78)
print("STIMA DEI CROMOFORI DALLO SPETTRO RICOSTRUITO "
      f"({'d_H2O = -d_lip' if vincolo_lipide_acqua else '4 cromofori liberi'}; "
      f"{'solo lambda con inclusione trovata' if solo_lambda_trovate else 'tutte le lambda'}"
      + (f", coerenti entro {tol_coerenza_mm:g} mm" if tol_coerenza_mm is not None else "") + ")")
print("   d_HHb, d_O2Hb, d_tHb in uM; d_lip, d_H2O in frazione di volume; "
      "riga VERO = unmixing dello spettro simulato")
print("="*78)
col_stampa = ["pos_id", "var_id", "metodo", "n_lambda_usate", "lambda_scartate_coerenza",
              "d_HHb", "d_HHb_vero", "d_O2Hb", "d_O2Hb_vero", "d_tHb", "d_tHb_vero",
              "d_lip", "d_lip_vero", "SO2_incl_stimata", "SO2_incl_vera", "residuo_relativo"]
print(crom[col_stampa].round(3).to_string(index=False))
# errore assoluto medio per metodo (esclusa la riga di controllo)
err_crom = (crom[crom["metodo"] != "VERO"]
            .assign(**{f"|errore_{k}|": lambda t, k=k: t[f"errore_{k}"].abs()
                       for k in ("d_HHb", "d_O2Hb", "d_tHb", "d_lip")})
            .groupby("metodo")[[f"|errore_{k}|" for k in ("d_HHb", "d_O2Hb", "d_tHb", "d_lip")]]
            .mean())
err_crom = per_metodo(err_crom)
print("\nErrore assoluto medio sulle variazioni dei cromofori (uM; frazione per d_lip):")
print(err_crom.round(3).to_string())
with pd.ExcelWriter(file_risultati, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as xw:
    crom.to_excel(xw, sheet_name="cromofori", index=False)
    err_crom.reset_index().to_excel(xw, sheet_name="errore_cromofori", index=False)
print(f"\nFogli 'cromofori' e 'errore_cromofori' aggiunti a '{file_risultati}'.")

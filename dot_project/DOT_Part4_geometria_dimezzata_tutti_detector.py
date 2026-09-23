import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp

# =============================================================================
# STEP 1 - Richiamo delle grandezze di Parte 1 e della griglia
# =============================================================================
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

# --- GEOMETRIA CON DISTANZE DIMEZZATE: si riprende il caso "tutti i
#     detector" (nessun pruning primi/secondi vicini) ma si dimezza la scala
#     dell'intero array sorgenti-rivelatori: passo base 6 mm anziche' 12 mm
#     (quindi anello sorgenti a raggio 6 mm, rivelatori fino a 12 mm invece
#     che 24 mm). Riduce le distanze sorgente-rivelatore per aumentare
#     l'ampiezza del segnale (leva "opzione 2": avvicinare sorgenti e
#     rivelatori), a differenza del pruning ("opzione 3") che agiva solo sul
#     condizionamento senza toccare l'SNR. La posizione della perturbazione
#     (xp,yp,zp) NON viene riscalata: resta al suo posto originale, cosi' da
#     testare l'effetto dell'avvicinamento su un bersaglio fisico fisso.
SCALE = 0.5   # fattore di scala della geometria sorgenti/rivelatori (1.0 = originale)

src_xy = SCALE * np.array([
    [0.0,  12.0],
    [-12.0, 0.0],
    [12.0,  0.0],
    [0.0, -12.0],
])
N_src = src_xy.shape[0]

det_xy = SCALE * np.array([
    [0.0,   0.0],
    [-12.0, 12.0], [0.0, 24.0], [12.0, 12.0],
    [12.0, -12.0], [-12.0, -12.0], [0.0, -24.0],
    [24.0,  0.0],  [-24.0, 0.0],
    [-12.0, 24.0], [12.0, 24.0], [12.0, -24.0], [-12.0, -24.0],
])
N_det = det_xy.shape[0]

# posizioni reale + immagine (metodo delle immagini, contorno estrapolato)
# per OGNI sorgente e OGNI rivelatore
r_s0_all = np.column_stack((src_xy, np.full(N_src, z0)))            # sorgenti reali
r_si_all = np.column_stack((src_xy, np.full(N_src, -z0 - 2*zb)))    # sorgenti immagine

r_d0 = np.column_stack((det_xy, np.full(N_det, 0.0)))       # rivelatori: superficie fisica z=0
r_di = np.column_stack((det_xy, np.full(N_det, -2*zb)))     # rivelatori: immagine

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

# --- sorgenti/rivelatori sono sui nodi del reticolo dei voxel? -------------
# Il reticolo (passo 4 mm, centri a x,y = ..., -6,-2,2,6,10,... cioe' 2 mod 4)
# NON e' stato pensato per allinearsi alle coordinate di sorgenti/rivelatori,
# e infatti con SCALE=0.5 in generale non lo fa (verificato sotto). Questo
# NON e' un problema: sorgenti e rivelatori non sono nodi del reticolo dei
# voxel, sono punti fisici continui usati per valutare in forma chiusa le
# funzioni di Green (fluence_inf, deltafluence_inf) alla distanza reale
# r = |r_voxel - r_sorgente/rivelatore|. Il reticolo discretizza SOLO
# l'incognita A (la perturbazione di assorbimento), non le posizioni di
# sorgenti/rivelatori, quindi nessuna interpolazione o snapping e' richiesta.
# L'unico caso in cui l'allineamento potrebbe creare un problema numerico e'
# una distanza r=0 esatta (singolarita' 1/r in deltafluence_inf): qui non
# puo' succedere perche' la quota z e' SEMPRE diversa fra voxel e sorgenti/
# rivelatori (voxel a z >= step/2 = 2 mm; rivelatori a z=0; sorgenti alla
# profondita' virtuale z0=1 mm), quindi anche una coincidenza (x,y) esatta
# lascerebbe comunque r > 0.
print("Allineamento sorgenti/rivelatori al reticolo (solo x,y, informativo):")
for name, arr, prefix in [("sorgenti", src_xy, "S"), ("rivelatori", det_xy, "D")]:
    for i, p in enumerate(arr):
        on_x = np.any(np.isclose(x_coords, p[0]))
        on_y = np.any(np.isclose(y_coords, p[1]))
        stato = "SUL reticolo" if (on_x and on_y) else "fuori reticolo"
        print(f"  {prefix}{i+1} ({p[0]:+.1f}, {p[1]:+.1f}) mm: {stato} "
              f"(x_on_grid={on_x}, y_on_grid={on_y})")
print("-> nessun problema: la posizione di sorgenti/rivelatori e' continua, "
      "non richiede coincidenza con i nodi del reticolo (vedi commento sopra).\n")

# --- verifica regime diffusivo -------------------------------------------
mfp = 1/mu_s0   # libero cammino medio di trasporto [mm]
sds_min = np.min(np.linalg.norm(det_xy[None, :, :] - src_xy[:, None, :], axis=-1))
print(f"Libero cammino medio di trasporto: {mfp:.2f} mm")
print(f"Separazione sorgente-rivelatore minima nella nuova geometria: {sds_min:.2f} mm "
      f"({sds_min/mfp:.1f} liberi cammini medi)")
print("-> tipicamente si richiede una separazione di almeno 5-10 liberi cammini medi "
      "per restare in un regime affidabilmente diffusivo.\n")

dt = 0.05
t = np.arange(0.1, 10.0, dt)

# --- TIME GATING A FINESTRE FISSE (12 gate da 0.2 ns) -----------------------
t_start = 1.0
gate_width = 0.20
N_gates = 12

gates = [(t_start + gate_width*i, t_start + gate_width*(i + 1)) for i in range(N_gates)]
gate_indices = [(np.searchsorted(t, g[0]), np.searchsorted(t, g[1])) for g in gates]

for g_i, (i0, i1) in enumerate(gate_indices):
    if i1 - i0 < 2:
        raise ValueError(f"Gate {g_i} ({gates[g_i]}) troppo stretto rispetto a dt={dt}: "
                          f"contiene solo {i1 - i0} campioni.")

N_meas = N_src * N_det * N_gates

print("="*55)
print("RICHIAMO: N_src =", N_src, "| N_det =", N_det,
      "| N_vox =", N_vox, "| N_gates =", N_gates)
print(f"Misurazioni indipendenti: {N_meas} (= N_src x N_det x N_gates)")
print(f"Incognite (voxel): {N_vox}")
print(f"Gate fissi da {gate_width} ns, range {gates[0][0]:.1f}-{gates[-1][1]:.1f} ns")
print("="*55 + "\n")


# =============================================================================
# STEP 2 - Vettore dei voxel A: perturbazione cubica di volume fisico esplicito
# =============================================================================
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

# --- sezione 2D nel piano z del voxel perturbato, con TUTTE le sorgenti e
#     i rivelatori della nuova configurazione -------------------------------
plt.figure()
im = plt.imshow(A_rep[:, :, iz_p].T, origin='lower',
                 extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                         y_coords[0]-step/2, y_coords[-1]+step/2),
                 vmin=0, vmax=dmu_a)
plt.scatter(src_xy[:, 0], src_xy[:, 1], marker='*', c='red', s=140,
            label=f'Sorgenti ({N_src})', zorder=3)
plt.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', s=40,
            label=f'Rivelatori ({N_det})', zorder=3)
plt.xlabel('x [mm]'); plt.ylabel('y [mm]')
plt.title(f"Sezione del vettore A nello spazio voxel\n"
          f"z = {zp:.0f} mm, step {step} mm - configurazione multi-sorgente")
plt.legend(fontsize=8)
plt.colorbar(im, label=r'$\delta\mu_a$ [$mm^{-1}$]')
plt.grid(); plt.show()


# =============================================================================
# STEP 3 - Costruzione della matrice di sensitivita' (Jacobiano) W
#          Ora W ha N_src*N_det*N_gates righe: ogni riga corrisponde a UNA
#          combinazione (sorgente attiva s, rivelatore d, gate g), esattamente
#          come nell'acquisizione reale con switch sequenziale delle sorgenti.
# =============================================================================
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

def meas_row(s, d, g):
    """Indice di riga in W per la tripletta (sorgente s, rivelatore d, gate g)."""
    return s*N_det*N_gates + d*N_gates + g

for s in range(N_src):
    r_s0 = r_s0_all[s]
    r_si = r_si_all[s]
    r_sv  = np.linalg.norm(r_V - r_s0, axis=1)     # sorgente s reale -> ogni voxel
    r_siv = np.linalg.norm(r_V - r_si, axis=1)     # sorgente s immagine -> ogni voxel

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
            int_dphi = np.sum(dphi_vt[:, i0:i1], axis=1) * dt
            int_phi0 = np.sum(phi0_t[i0:i1]) * dt
            W[meas_row(s, d, g), :] = int_dphi / int_phi0

print(f"Matrice W costruita: shape = {W.shape}  "
      f"({N_src} sorgenti x {N_det} rivelatori x {N_gates} gate)")

W_5d = W.reshape(N_src, N_det, N_gates, x_coords.size, y_coords.size, z_coords.size)
Wlog_vmin, Wlog_vmax = -4, 1

# --- sezioni 2D di W per UNA coppia (sorgente, rivelatore) fissata,
#     a diverse profondita' e gate ------------------------------------------
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
        axs[r_i, c_i].scatter(*src_xy[s_show], marker='*', c='red', s=40)
        axs[r_i, c_i].scatter(*det_xy[d_show], marker='o', c='lime', s=20)
        axs[r_i, c_i].set_title(f"z={z_coords[iz]:.0f} mm, gate {g} "
                                 f"({gates[g][0]:.1f}-{gates[g][1]:.1f} ns)", fontsize=9, pad=8)
        if c_i == 0: axs[r_i, c_i].set_ylabel('y [mm]')
        if r_i == 2: axs[r_i, c_i].set_xlabel('x [mm]')
fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.88, right=0.86)
fig.suptitle(r'Sezioni di $\log_{10}|W|$ nello spazio voxel - '
             f'sorgente S{s_show+1}, rivelatore D{d_show+1}'
             '\n(scala colore condivisa fra tutti i pannelli, 12 gate da 0.2 ns)')
fig.colorbar(im, ax=axs, shrink=0.7, label=r'$\log_{10}|W|$')
plt.show()

# --- confronto fra TUTTI i rivelatori, sorgente e profondita'/gate fissati -
z_fix = 18.0
g_fix = 3
iz = np.argmin(np.abs(z_coords - z_fix))
n_cols = 5
n_rows = int(np.ceil(N_det / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 3.0*n_rows))
axs = axs.flatten()
for d in range(N_det):
    Wsec = np.log10(np.clip(np.abs(W_5d[s_show, d, g_fix, :, :, iz]), 1e-8, None)).T
    im = axs[d].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[d].scatter(*src_xy[s_show], marker='*', c='cyan', s=40)
    axs[d].scatter(*det_xy[d], marker='o', c='lime', s=25)
    axs[d].set_title(f'D{d+1}', fontsize=9)
for extra in range(N_det, n_rows*n_cols):
    axs[extra].axis('off')
fig.suptitle(f"Sensitivita' " r"$\log_{10}|W|$" f" per ciascun rivelatore, sorgente S{s_show+1}\n"
             f"z = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()

# --- NUOVO: confronto fra TUTTE le sorgenti, rivelatore fissato -----------
# Mostra perche' accendere sorgenti diverse aiuta: ogni sorgente "illumina"
# il volume da un'angolazione diversa, cambiando la mappa di sensitivita'
# vista dallo STESSO rivelatore.
d_fix = 0
fig, axs = plt.subplots(1, N_src, figsize=(3.4*N_src, 3.6))
if N_src == 1: axs = [axs]
for s in range(N_src):
    Wsec = np.log10(np.clip(np.abs(W_5d[s, d_fix, g_fix, :, :, iz]), 1e-8, None)).T
    im = axs[s].imshow(Wsec, origin='lower', vmin=Wlog_vmin, vmax=Wlog_vmax,
                extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                        y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[s].scatter(*src_xy[s], marker='*', c='red', s=45)
    axs[s].scatter(*det_xy[d_fix], marker='o', c='lime', s=25)
    axs[s].set_title(f'S{s+1} attiva', fontsize=9)
fig.suptitle(f"Sensitivita' " r"$\log_{10}|W|$" f" viste dal rivelatore D{d_fix+1} al variare della sorgente attiva\n"
             f"z = {z_coords[iz]:.0f} mm, gate {g_fix} "
             f"({gates[g_fix][0]:.1f}-{gates[g_fix][1]:.1f} ns)")
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()


# =============================================================================
# Helper di plotting per grandezze nello spazio delle misure (N_src x N_det x
# N_gates), usato per M ideale/rumoroso e per i modi u_i della SVD.
# =============================================================================
def plot_measurement_matrix(ax, mat2d, title, cmap='RdBu_r', vlim=None):
    """mat2d: array (N_src*N_det, N_gates), righe raggruppate per sorgente."""
    if vlim is None:
        vlim = np.nanmax(np.abs(mat2d))
    im = ax.imshow(mat2d, aspect='auto', cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(N_gates))
    ax.set_xticklabels([f'{g[0]:.1f}-{g[1]:.1f}' for g in gates], rotation=90, fontsize=6)
    ax.set_xlabel('time gate [ns]')
    for s in range(1, N_src):
        ax.axhline(s*N_det - 0.5, color='k', linewidth=0.8)
    tick_pos = [s*N_det + N_det/2 - 0.5 for s in range(N_src)]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f'S{s+1}\n(D1..D{N_det})' for s in range(N_src)], fontsize=7)
    ax.set_title(title)
    return im


# =============================================================================
# STEP 4 - Vettore delle misure M = W . A (problema diretto, senza rumore)
# =============================================================================
M_ideal = (W @ A).reshape(N_src, N_det, N_gates)

print("VETTORE DELLE MISURE M = W . A (ideale, senza rumore)")
for s in range(N_src):
    print(f"-- Sorgente S{s+1} --")
    for d in range(N_det):
        print(f"  D{d+1}: " + " ".join(f"{val:+.4f}" for val in M_ideal[s, d, :]))

fig, ax = plt.subplots(figsize=(8, 0.42*N_src*N_det + 1.5))
plot_measurement_matrix(ax, M_ideal.reshape(N_src*N_det, N_gates),
                         'Vettore delle misure M = W . A (ideale, mezzo semi-infinito)\n'
                         f'{N_src} sorgenti x {N_det} rivelatori, 12 gate fissi da 0.2 ns')
fig.colorbar(ax.images[0], ax=ax, label='C (-)')
plt.tight_layout(); plt.show()


# =============================================================================
# STEP 5 - Aggiunta di rumore poissoniano (shot noise) al vettore delle misure
# =============================================================================
# Ogni coppia (sorgente attiva, rivelatore) e' un'acquisizione TCSPC a se'
# stante (la sorgente s e' accesa, tutti i rivelatori misurano la propria
# DTOF, poi si passa alla sorgente s+1): lo shot noise va quindi applicato
# in modo indipendente per ciascuna tripletta (s, d, gate).

np.random.seed(42)

Phi0_int = np.zeros((N_src, N_det, N_gates))
for s in range(N_src):
    r_s0 = r_s0_all[s]; r_si = r_si_all[s]
    for d in range(N_det):
        r_d_s = np.linalg.norm(r_d0[d] - r_s0)
        r_d_si = np.linalg.norm(r_d0[d] - r_si)
        phi0_t = (fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_s)
                  - fluence_inf(mu_a0, mu_s0, n_idx, t, r_d_si))
        for g, (i0, i1) in enumerate(gate_indices):
            Phi0_int[s, d, g] = np.sum(phi0_t[i0:i1]) * dt

# budget fotonico per ciascuna acquisizione (sorgente, rivelatore) - stesso
# ordine di grandezza per ogni coppia, come nel caso a sorgente singola
N_tot = 1e6
N0_ideal = N_tot * Phi0_int / Phi0_int.sum(axis=2, keepdims=True)
Npert_ideal = N0_ideal * (1 + M_ideal)

N0_noisy = np.random.poisson(N0_ideal)
Npert_noisy = np.random.poisson(np.clip(Npert_ideal, 0, None))

with np.errstate(divide='ignore', invalid='ignore'):
    M_noisy = np.where(N0_noisy > 0,
                        (Npert_noisy - N0_noisy) / N0_noisy,
                        np.nan)

print(f"Budget fotonico: N_tot = {N_tot:.0e} conteggi per coppia sorgente-rivelatore")
print(f"Conteggi minimi/massimi di baseline per gate: "
      f"{N0_ideal.min():.1f} / {N0_ideal.max():.1f}")

vmax = np.nanmax(np.abs(np.concatenate([M_ideal.ravel(), M_noisy.ravel()])))
fig, axs = plt.subplots(1, 2, figsize=(14, 0.42*N_src*N_det + 1.5), sharey=True)
plot_measurement_matrix(axs[0], M_ideal.reshape(N_src*N_det, N_gates),
                         'M ideale (senza rumore)', vlim=vmax)
plot_measurement_matrix(axs[1], M_noisy.reshape(N_src*N_det, N_gates),
                         'M con rumore poissoniano (shot noise)', vlim=vmax)
fig.colorbar(axs[1].images[0], ax=axs, shrink=0.8, label='C (-)')
fig.suptitle(f'Confronto M ideale vs M con shot noise '
             f'(N_tot = {N_tot:.0e} conteggi/coppia sorgente-rivelatore)')
plt.show()

# --- profilo di una singola coppia (sorgente, rivelatore), gate per gate ---
s_show2, d_show2 = 0, 0
plt.figure(figsize=(7, 4))
plt.plot(range(N_gates), M_ideal[s_show2, d_show2], 'o-', label='M ideale')
plt.plot(range(N_gates), M_noisy[s_show2, d_show2], 's--', label='M con rumore')
plt.xlabel('indice gate temporale')
plt.ylabel('C (-)')
plt.title(f'Sorgente S{s_show2+1}, rivelatore D{d_show2+1}: confronto ideale vs rumoroso')
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# STEP 6 - Decomposizione ai valori singolari di W
# =============================================================================
U, s_sv, Vt = np.linalg.svd(W, full_matrices=False)
k_max = s_sv.size

print("="*55)
print("SVD DI W")
print("="*55)
print(f"U: {U.shape}   s: {s_sv.shape}   Vt: {Vt.shape}")
print(f"Valore singolare massimo  s_0        = {s_sv[0]:.3e}")
print(f"Valore singolare minimo   s_{k_max-1:<3d} = {s_sv[-1]:.3e}")
print(f"Rapporto s_0/s_min (numero di condizionamento di W) = {s_sv[0]/s_sv[-1]:.2e}")
print("="*55 + "\n")

plt.figure()
plt.semilogy(np.arange(1, k_max + 1), s_sv, 'o-', ms=4)
plt.xlabel('ordine $i$'); plt.ylabel('$s_i$')
plt.title("Valori singolari di $W$ (scala logaritmica)\n"
          f"{N_src} sorgenti x {N_det} rivelatori x {N_gates} gate = {N_meas} misure")
plt.grid(); plt.show()


# =============================================================================
# STEP 7 - Alcuni modi (coppie u_i, v_i): dai piu' "forti" ai piu' "deboli"
# =============================================================================
orders_to_show = sorted(set(min(o, k_max - 1) for o in [0, 1, 5, 20, k_max//2, k_max - 1]))
iz_show = np.argmin(np.abs(z_coords - zp))

fig, axs = plt.subplots(len(orders_to_show), 2,
                         figsize=(9, 0.28*N_src*N_det*len(orders_to_show) + 2*len(orders_to_show)))
for row, i in enumerate(orders_to_show):
    U_img = U[:, i].reshape(N_src*N_det, N_gates)
    im = plot_measurement_matrix(axs[row, 0], U_img,
                                  f"$u_{{{i}}}$ (spazio misure) — $s_{{{i}}}$={s_sv[i]:.2e}")

    V_3d = Vt[i, :].reshape(x_coords.size, y_coords.size, z_coords.size)
    V_section = V_3d[:, :, iz_show].T
    vmax_v = np.abs(V_section).max()
    axs[row, 1].imshow(V_section, origin='lower', cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v,
                        extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                                y_coords[0]-step/2, y_coords[-1]+step/2))
    axs[row, 1].scatter(src_xy[:, 0], src_xy[:, 1], marker='*', c='k', s=35)
    axs[row, 1].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=10, alpha=0.5)
    axs[row, 1].set_title(f"$v_{{{i}}}$ (spazio voxel, $z\\approx${z_coords[iz_show]:.0f} mm)", fontsize=10)
fig.tight_layout()
plt.show()


# =============================================================================
# STEP 8 - Regolarizzazione di TIKHONOV, ideale vs rumore
#          (sostituisce la TSVD come metodo di inversione)
# =============================================================================
# Formulazione standard di Tikhonov (L = I, soluzione di default A_inf = 0 -
# come in "Relation_Project_Phase_2_Team_7.pdf", sez. REGULARIZATION):
#
#   min_A  || M - W A ||^2 + lambda^2 || A ||^2
#
# che porta alle equazioni normali
#
#   (lambda^2 I + W^T W) A = W^T M   =>   A_hat(lambda) = (W^T W + lambda^2 I)^-1 W^T M
#
# Costruire esplicitamente W^T W (N_vox x N_vox = 2048x2048) e' inutile e
# numericamente svantaggioso (il suo numero di condizionamento e' il
# QUADRATO di quello di W). Sfruttando la SVD gia' calcolata allo Step 6,
# la stessa soluzione si scrive come pseudo-inversa filtrata in modo
# CONTINUO (a differenza del cutoff 0/1 della TSVD):
#
#   A_hat(lambda) = V diag( s_i / (s_i^2 + lambda^2) ) U^T M
#                 = sum_i [ s_i / (s_i^2 + lambda^2) ] (u_i^T M) v_i
#
# Il fattore di filtro f_i = s_i^2/(s_i^2+lambda^2) vale ~1 per s_i >> lambda
# (modo forte, quasi non filtrato) e decresce con continuita' verso 0 per
# s_i << lambda (modo debole, smorzato gradualmente anziche' azzerato di
# colpo come nella TSVD): lambda e' il parametro di regolarizzazione,
# analogo al k della TSVD ma continuo.

M_ideal_flat = M_ideal.flatten()          # stesso ordine (s, d, g) delle righe di W
M_noisy_flat = np.nan_to_num(M_noisy, nan=0.0).flatten()

def tikhonov_solve(M_flat, lam):
    """A_hat(lambda) = V diag(s_i/(s_i^2+lambda^2)) U^T M
    (equivalente a (W^T W + lambda^2 I)^-1 W^T M, ma stabile numericamente)."""
    filt = s_sv / (s_sv**2 + lam**2)
    coeffs = filt * (U.T @ M_flat)
    return Vt.T @ coeffs

# lambda spazia in scala logaritmica da poco sopra il valore singolare
# massimo (regolarizzazione fortissima: quasi tutta l'informazione tagliata,
# A_hat -> 0) fino a molto sotto il valore singolare minimo (regolarizzazione
# trascurabile: A_hat(lambda->0) -> pseudo-inversa completa, come TSVD a
# k=k_max)
N_lambda = 80
lambda_values = np.logspace(np.log10(s_sv[0]*2), np.log10(s_sv[-1]*1e-2), N_lambda)

rel_err_ideal = np.zeros(N_lambda)
rel_err_noisy = np.zeros(N_lambda)

for j, lam in enumerate(lambda_values):
    A_hat_i = tikhonov_solve(M_ideal_flat, lam)
    A_hat_n = tikhonov_solve(M_noisy_flat, lam)
    rel_err_ideal[j] = np.linalg.norm(A_hat_i - A) / np.linalg.norm(A)
    rel_err_noisy[j] = np.linalg.norm(A_hat_n - A) / np.linalg.norm(A)

j_best = np.argmin(rel_err_noisy)
lambda_best = lambda_values[j_best]   # lambda "oracolo": nella pratica non si
                                       # conosce A, quindi non si potrebbe
                                       # calcolare - qui serve solo a mostrare
                                       # dove si trova il minimo vero

plt.figure()
plt.loglog(lambda_values, rel_err_ideal, 'o-', ms=3, label=r'errore su $A$ (dati IDEALI)')
plt.loglog(lambda_values, rel_err_noisy, 's-', ms=3, label=r'errore su $A$ (dati con RUMORE)')
plt.axvline(lambda_best, color='gray', linestyle='--', linewidth=1,
            label=f'$\\lambda$ ottimale (oracolo) = {lambda_best:.2e}')
plt.gca().invert_xaxis()   # lambda decrescente verso destra = regolarizzazione
                           # via via piu' debole (stessa direzione "meno
                           # regolarizzazione a destra" della TSVD in k)
plt.xlabel(r'parametro di regolarizzazione $\lambda$  (regolarizzazione piu debole $\to$)')
plt.ylabel(r'errore relativo $\|\hat A-A\|/\|A\|$')
plt.title(f'Regolarizzazione di Tikhonov: ideale vs con rumore\n'
          f'({N_src} sorgenti x {N_det} rivelatori x {N_gates} gate = {N_meas} misure)')
plt.legend(); plt.grid(which='both', alpha=0.4); plt.show()

print(f"Dati ideali: errore minimo = {rel_err_ideal.min():.3f} a "
      f"lambda={lambda_values[np.argmin(rel_err_ideal)]:.2e}")
print(f"Dati con rumore: errore minimo = {rel_err_noisy.min():.3f} a lambda={lambda_best:.2e}")
print(f"Dati con rumore, lambda minimo esplorato (quasi pseudo-inversa completa): "
      f"errore = {rel_err_noisy[-1]:.3e}")


# =============================================================================
# STEP 9 - Confronto visivo: A vero vs ricostruzioni (dati con rumore) a lambda diversi
# =============================================================================
lambda_show = sorted({lambda_best*10, lambda_best, lambda_best/10}, reverse=True)
lambda_show = [lam for lam in lambda_show if lambda_values[-1] <= lam <= lambda_values[0]]
if not lambda_show:
    lambda_show = [lambda_best]

iz_show2 = np.argmin(np.abs(z_coords - zp))

fig, axs = plt.subplots(1, len(lambda_show)+1, figsize=(4*(len(lambda_show)+1), 4.2))
if len(lambda_show) == 0:
    axs = [axs]

A_section_true = A_rep[:, :, iz_show2].T
axs[0].imshow(A_section_true, origin='lower', cmap='viridis', vmin=0, vmax=dmu_a,
              extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                      y_coords[0]-step/2, y_coords[-1]+step/2))
axs[0].scatter(src_xy[:, 0], src_xy[:, 1], marker='*', c='red', s=90)
axs[0].scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='cyan', s=25)
axs[0].set_title('$A$ vero'); axs[0].set_xlabel('x [mm]'); axs[0].set_ylabel('y [mm]')

for ax, lam in zip(axs[1:], lambda_show):
    A_hat = tikhonov_solve(M_noisy_flat, lam)
    A_hat_3d = A_hat.reshape(x_coords.size, y_coords.size, z_coords.size)
    section = A_hat_3d[:, :, iz_show2].T
    vmax_hat = np.abs(A_hat).max()
    err_lam = np.linalg.norm(A_hat - A) / np.linalg.norm(A)
    im = ax.imshow(section, origin='lower', cmap='RdBu_r', vmin=-vmax_hat, vmax=vmax_hat,
                    extent=(x_coords[0]-step/2, x_coords[-1]+step/2,
                            y_coords[0]-step/2, y_coords[-1]+step/2))
    ax.scatter(src_xy[:, 0], src_xy[:, 1], marker='*', c='k', s=90)
    ax.scatter(det_xy[:, 0], det_xy[:, 1], marker='o', c='k', s=18, alpha=0.5)
    ax.set_title(f'$\\hat A$ (rumore), $\\lambda$={lam:.1e}\nerr={err_lam:.2f}')
    ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"Ricostruzione Tikhonov da dati con rumore, $z\\approx${z_coords[iz_show2]:.0f} mm "
             f"(vero: $x_p$={xp:.0f}, $y_p$={yp:.0f}, $z_p$={zp:.0f} mm)\n"
             f"{N_src} sorgenti x {N_det} rivelatori (switch sequenziale)")
fig.tight_layout()
plt.show()

print("="*55)
print("LOCALIZZAZIONE (dati con rumore): voxel di picco vs posizione vera")
print("="*55)
print(f"Vero voxel perturbato:  ({xp:.1f}, {yp:.1f}, {zp:.1f}) mm")
for lam in lambda_show:
    A_hat = tikhonov_solve(M_noisy_flat, lam)
    peak = r_V[np.argmax(A_hat)]
    pos_mass = np.clip(A_hat, 0, None).sum()
    neg_mass = np.clip(-A_hat, 0, None).sum()
    w_abs = np.abs(A_hat)
    idx_sorted = np.argsort(-w_abs)
    cum = np.cumsum(w_abs[idx_sorted]); cum /= cum[-1]
    n50 = np.searchsorted(cum, 0.5) + 1
    print(f"lambda={lam:.2e}:  picco = ({peak[0]:5.2f}, {peak[1]:5.2f}, {peak[2]:5.2f}) mm   "
          f"| massa negativa/positiva = {neg_mass/max(pos_mass,1e-30):.2f}   "
          f"| voxel per il 50% della massa |A_hat| = {n50}")
print("="*55)

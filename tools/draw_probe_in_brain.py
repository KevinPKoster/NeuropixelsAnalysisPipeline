"""
Clean Allen-atlas drawing of the Neuropixels 1.0 insertion (CFA entry -> VAL): sagittal, dorsal and coronal sections.
Light theme; only the implicated regions are coloured, everything else is pale.

Run with an environment that has brainglobe-atlasapi + matplotlib (an isolated venv is used so analysis envs stay untouched):
    python draw_probe_in_brain.py
Output: D:\\4103_plots\\anatomy\\probe_in_brain_light.png
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from brainglobe_atlasapi import BrainGlobeAtlas

OUT = r"D:\4103_plots\anatomy\probe_in_brain_light.png"
a = BrainGlobeAtlas("allen_mouse_25um", brainglobe_dir=os.path.join(os.environ["TEMP"], "bg_atlas"))
ann = a.annotation; res = 25.0; AP0, DV0, ML0 = 5400.0, 332.0, 5739.0      # bregma in the CCF (um from front / top / right edge)
NI, NJ, NK = ann.shape
ids_of = lambda n: [a.structures[x]["id"] for x in a.get_structure_descendants(n)] + [a.structures[n]["id"]]
MO_ids, TH_ids, VAL_id = ids_of("MOp") + ids_of("MOs"), ids_of("TH"), a.structures["VAL"]["id"]
WM_ids = ids_of("fiber tracts") + ids_of("VS")

C = dict(bg="#ffffff", tissue="#f4f6f9", brain_edge="#cfd5de", fg="#1f2937", dim="#6b7280", cfa="#3b82f6", th="#14b8a6",
         val="#22c55e", wm="#9ca3af", probe="#1f2937", rec="#f59e0b")

def paint(brain, mo, cfa, th, val):
    img = np.zeros(brain.shape + (3,)); img[:] = to_rgb(C["bg"]); img[brain] = to_rgb(C["tissue"])
    def blend(m, col, al): img[m] = (1 - al) * img[m] + al * np.array(to_rgb(col))
    blend(mo, C["cfa"], 0.10); blend(cfa, C["cfa"], 0.55); blend(th & ~val, C["th"], 0.22); blend(val, C["val"], 0.85)
    return img

def brain_edge(brain):                                   # only the outline of the whole brain, very light
    b = np.zeros(brain.shape, bool); b[:-1] |= brain[:-1] != brain[1:]; b[:, :-1] |= brain[:, :-1] != brain[:, 1:]
    ov = np.zeros(brain.shape + (4,)); ov[b] = list(to_rgb(C["brain_edge"])) + [1.0]; return ov

# ---- the recommended insertion ----
AP_E, ML_E, PITCH, YAW = 0.5, 1.3, 25.0, 0.0
SURF_TO_TIP = 5380.0 + 460.0            # fitted brain-surface-to-tip distance + 0.46 mm, so the recorded window reaches VAL
k_e, i_e = int(round((ML0 - ML_E * 1000) / res)), int(round((AP0 - AP_E * 1000) / res))
dv_surf = float(np.argmax(ann[i_e, :, k_e] > 0)) * res
p, y = np.radians(PITCH), np.radians(YAW)
dvec = np.array([-np.sin(p), np.cos(p) * np.cos(y), -np.cos(p) * np.sin(y)])
pos = lambda L: (AP_E + L * dvec[0] / 1000, ML_E + L * dvec[2] / 1000, (dv_surf + L * dvec[1] - DV0) / 1000)   # AP, ML, depth below bregma (mm)
REC = (SURF_TO_TIP - 5340.0, SURF_TO_TIP - 1520.0)          # recorded sites: 1520..5340 um from the tip

def struct_along(L):
    ap, ml, dep = pos(L)
    i = np.round((AP0 - ap * 1000) / res).astype(int); j = np.round((dep * 1000 + DV0) / res).astype(int); k = np.round((ML0 - ml * 1000) / res).astype(int)
    ok = (i >= 0) & (i < NI) & (j >= 0) & (j < NJ) & (k >= 0) & (k < NK)
    v = np.zeros(len(L), int); v[ok] = ann[i[ok], j[ok], k[ok]]
    cls = np.full(len(L), "out", dtype=object)
    cls[np.isin(v, WM_ids)] = "wm"; cls[np.isin(v, TH_ids)] = "th"; cls[v == VAL_id] = "val"; cls[np.isin(v, MO_ids)] = "cfa"
    return cls
Lfine = np.arange(0, SURF_TO_TIP + 1, 10.0); cls_path = struct_along(Lfine)

fig = plt.figure(figsize=(17, 9.6), facecolor=C["bg"])
gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0], wspace=0.05, hspace=0.18, left=0.04, right=0.985, top=0.86, bottom=0.10)
axs = fig.add_subplot(gs[:, 0]); axd = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1])
for ax in (axs, axd, axc):
    ax.set_facecolor(C["bg"]); ax.tick_params(colors=C["dim"], labelsize=9, length=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(C["brain_edge"])

# ---- sagittal ----
sl = ann[:, :, k_e]
brain = (sl > 0).T; mo = np.isin(sl, MO_ids).T; th = np.isin(sl, TH_ids).T; val = (sl == VAL_id).T
apmm = (AP0 - np.arange(NI)[None, :] * res) / 1000; mlk = abs(ML0 - k_e * res) / 1000
cfa = mo & (apmm >= -0.25) & (apmm <= 1.25) & (mlk >= 0.8) & (mlk <= 2.0)
ext = (AP0 / 1000, (AP0 - NI * res) / 1000, (NJ * res - DV0) / 1000, -DV0 / 1000)
axs.imshow(paint(brain, mo, cfa, th, val), extent=ext, origin="upper", aspect="equal", interpolation="nearest")
axs.imshow(brain_edge(brain), extent=ext, origin="upper", aspect="equal", interpolation="nearest")
axs.set_xlim(3.0, -3.3); axs.set_ylim(6.6, -0.45)
axs.set_xlabel("AP from bregma (mm)", color=C["dim"], fontsize=10); axs.set_ylabel("depth below bregma level (mm)", color=C["dim"], fontsize=10)
axs.set_title(f"sagittal section, ML +{ML_E:.1f} mm", color=C["fg"], fontsize=12, pad=6)

t = np.array([dvec[0], dvec[1]]); t /= np.linalg.norm(t); nrm = np.array([t[1], -t[0]])
def shank_polygon(L0, L1, width_um=70.0, taper_um=250.0):
    L = np.linspace(L0, L1, 400); ap, ml, dep = pos(L); cen = np.column_stack([ap, dep])
    w = np.minimum(width_um / 2000.0, np.maximum((L1 - L) / taper_um, 0) * width_um / 2000.0)
    return np.vstack([cen + nrm * w[:, None], (cen - nrm * w[:, None])[::-1]])
axs.add_patch(Polygon(shank_polygon(-1500, SURF_TO_TIP), closed=True, facecolor=C["probe"], edgecolor="none", zorder=7))
colmap = {"cfa": C["cfa"], "wm": C["wm"], "th": C["th"], "val": C["val"]}
ap_s, _, dep_s = pos(Lfine); OFF = 0.10
for k in range(len(Lfine) - 1):
    if cls_path[k] == "out": continue
    q0 = np.array([ap_s[k], dep_s[k]]) + nrm * OFF; q1 = np.array([ap_s[k + 1], dep_s[k + 1]]) + nrm * OFF
    axs.plot([q0[0], q1[0]], [q0[1], q1[1]], color=colmap[cls_path[k]], lw=3.2, solid_capstyle="butt", zorder=8)
Lr = np.linspace(REC[0], REC[1], 100); ap_r, _, dep_r = pos(Lr)
axs.plot(ap_r - nrm[0] * OFF, dep_r - nrm[1] * OFF, color=C["rec"], lw=3.2, solid_capstyle="butt", zorder=9)
axs.plot([0], [0], marker="+", color=C["dim"], ms=9, mew=1.3, zorder=10)

def lab_(text, color, y, xy, size=12):
    axs.annotate(text, xy=xy, xytext=(2.85, y), color=color, fontsize=size, fontweight="bold", ha="left", va="center", zorder=12,
                 arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=3, shrinkB=1))
ap_e, _, dep_e = pos(0)
lab_("CFA", C["cfa"], 0.6, (ap_e + nrm[0] * OFF, dep_e + nrm[1] * OFF), size=14)
ti = np.where(cls_path == "th")[0]
if len(ti):
    ap_t, _, dep_t = pos(Lfine[ti[len(ti) // 4]]); lab_("thalamus", C["th"], 4.5, (ap_t + nrm[0] * OFF, dep_t + nrm[1] * OFF))
vi = np.where(cls_path == "val")[0]
if len(vi):
    ap_v, _, dep_v = pos(Lfine[vi[len(vi) // 2]]); lab_("VAL", C["val"], 5.4, (ap_v + nrm[0] * OFF, dep_v + nrm[1] * OFF), size=14)
mid = pos((REC[0] + REC[1]) / 2)
axs.annotate("384 recorded sites", xy=(mid[0] - nrm[0] * OFF, mid[2] - nrm[1] * OFF), xytext=(-1.75, 2.1), color=C["rec"], fontsize=12, fontweight="bold",
             ha="left", va="center", arrowprops=dict(arrowstyle="-", color=C["rec"], lw=0.8, shrinkA=3, shrinkB=1), zorder=12)
ap_tip, _, dep_tip = pos(SURF_TO_TIP)
axs.text(ap_tip - 0.3, dep_tip + 0.05, "tip", color=C["dim"], fontsize=10, ha="left", va="center")
axs.add_patch(Rectangle((-2.15, 0.75), 1.0, 0.05, color=C["fg"], zorder=12)); axs.text(-1.65, 0.68, "1 mm", color=C["fg"], fontsize=10, ha="center", va="bottom")

# ---- dorsal ----
brain_top = (ann > 0).any(axis=1); mo_full = np.isin(ann, MO_ids)
cfa_i = np.zeros((NI, NK), bool)
i0, i1 = int((AP0 - 1250) / res), int((AP0 + 250) / res)
for kA, kB in ((int((ML0 - 2000) / res), int((ML0 - 800) / res)), (int((ML0 + 800) / res), int((ML0 + 2000) / res))):
    cfa_i[i0:i1, kA:kB] = mo_full[i0:i1, :, kA:kB].any(axis=1)
th_i = np.isin(ann, TH_ids).any(axis=1); val_i = (ann == VAL_id).any(axis=1)
extd = ((ML0 - (NK - 1) * res) / 1000, ML0 / 1000, (AP0 - NI * res) / 1000, AP0 / 1000)
axd.imshow(paint(brain_top, mo_full.any(axis=1), cfa_i, th_i, val_i)[:, ::-1], extent=extd, origin="upper", aspect="equal", interpolation="nearest")
axd.imshow(brain_edge(brain_top)[:, ::-1], extent=extd, origin="upper", aspect="equal", interpolation="nearest")
axd.set_xlim(-4.6, 4.6); axd.set_ylim(-6.9, 4.2)
Lp = np.linspace(0, SURF_TO_TIP, 100); ap_p, ml_p, _ = pos(Lp)
axd.plot(ml_p, ap_p, color=C["probe"], lw=1.6)
Lr = np.linspace(REC[0], REC[1], 50); ap_r2, ml_r2, _ = pos(Lr); axd.plot(ml_r2, ap_r2, color=C["rec"], lw=3.0)
axd.plot([ML_E], [AP_E], "o", color=C["cfa"], ms=6, mec=C["fg"], mew=0.8); axd.plot([0], [0], marker="+", color=C["dim"], ms=8, mew=1.2)
axd.set_title("dorsal view", color=C["fg"], fontsize=11, pad=4); axd.set_ylabel("AP (mm)", color=C["dim"], fontsize=9)

# ---- coronal through VAL ----
L_val = Lfine[vi[len(vi) // 2]] if len(vi) else 3800.0
ap_c = pos(L_val)[0]; ic = int(round((AP0 - ap_c * 1000) / res)); slc = ann[ic]
brain_c = slc > 0; mo_c = np.isin(slc, MO_ids); th_c = np.isin(slc, TH_ids); val_c = slc == VAL_id
extc = ((ML0 - (NK - 1) * res) / 1000, ML0 / 1000, (NJ * res - DV0) / 1000, -DV0 / 1000)
axc.imshow(paint(brain_c, mo_c, np.zeros_like(mo_c), th_c, val_c)[:, ::-1], extent=extc, origin="upper", aspect="equal", interpolation="nearest")
axc.imshow(brain_edge(brain_c)[:, ::-1], extent=extc, origin="upper", aspect="equal", interpolation="nearest")
axc.set_xlim(-4.3, 4.3); axc.set_ylim(6.4, -0.5)
_, ml_c, dep_c = pos(np.linspace(0, SURF_TO_TIP, 100)); axc.plot(ml_c, dep_c, color=C["probe"], lw=1.0)
_, ml_r3, dep_r3 = pos(np.linspace(REC[0], REC[1], 50)); axc.plot(ml_r3, dep_r3, color=C["rec"], lw=2.6)
_, ml_x, dep_x = pos(L_val); axc.add_patch(Circle((ml_x, dep_x), 0.12, facecolor="none", edgecolor=C["probe"], lw=1.5, zorder=6))
axc.set_title(f"coronal section, AP {ap_c:+.1f} mm", color=C["fg"], fontsize=11, pad=4)
axc.set_xlabel("ML (mm)", color=C["dim"], fontsize=9); axc.set_ylabel("depth (mm)", color=C["dim"], fontsize=9)

handles = [Rectangle((0, 0), 1, 1, color=C["cfa"], alpha=0.55), Rectangle((0, 0), 1, 1, color=C["th"], alpha=0.35), Rectangle((0, 0), 1, 1, color=C["val"], alpha=0.85),
           Rectangle((0, 0), 1, 1, color=C["wm"]), Line2D([0], [0], color=C["rec"], lw=3)]
fig.legend(handles, ["CFA (approx.: motor cortex, AP -0.25 to +1.25 mm)", "thalamus", "VAL", "white matter / ventricle (strip beside the shank)", "recorded part of the shank"],
           loc="lower center", ncol=5, frameon=False, fontsize=10.5, labelcolor=C["fg"], bbox_to_anchor=(0.5, 0.012))
fig.text(0.04, 0.955, "Mouse 4103: CFA entry to VAL", fontsize=20, fontweight="bold", color=C["fg"], va="center")
fig.text(0.04, 0.915, f"Neuropixels 1.0, drawn to scale on Allen CCF sections.  Entry AP +{AP_E:.1f} / ML +{ML_E:.1f} mm, pitch {PITCH:.0f} deg posterior, no yaw, "
         f"brain surface to tip {SURF_TO_TIP/1000:.2f} mm.", fontsize=11, color=C["dim"], va="center")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, facecolor=C["bg"]); print("saved", OUT)

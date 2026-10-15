"""Revised manuscript figures (review revision, 2026-10-13).

Regenerates the main-text and supplementary figures at print width (34 pc,
5.65 in) with legible type, a common anisotropy colour scale, and Equal Earth
basemaps for global fields:

  fig_tissot_schematic.png    projection ellipse, forward indicatrix, reach ellipse (Fig. 1)
  geoclip_equal_earth.png     Equal Earth vs GeoCLIP fields with Tissot-style glyphs (Fig. 4)
  fig_indicatrix_v2.png       directional profiles (Fig. 2)
  fig_validation_v2.png       analytic input references (Fig. 3)
  dense_regional_v2.png       dense regional fields (Fig. 5)
  dense_regional_v3.png       world maps above the dense regional fields (not used; run 'dense_world')
  other_families_ee.png       fixed-grid world maps for CSP, Climplicit and SINR (supplement)
  neighbourhood_maps_v2.png   neighborhood maps and overlap counts (Fig. 6)
  satclip_backbones_ee.png    SatCLIP backbone fields (supplement)

Inputs are the cached profiles and measurements already in the package; only
numpy and matplotlib are required. Figures are written to outputs/figures/ and
copied to manuscript/figures/.

Usage: python scripts/make_revision_figures.py
"""
import shutil
import sys
import os
import subprocess
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon
import matplotlib.patheffects as pe

DATA = OUTPUT = OUT = COAST = None
MS = None
from etissot.geometry.fitting import fit_many, fit_ellipse, fitted_tensor  # noqa: E402
from etissot.statistics.spatial import GLAT, GLON  # noqa: E402
from etissot.geometry.references import equal_earth_metric  # noqa: E402


WIDTH = 5.65
B = np.arange(32) * 2 * np.pi / 32
A_NORM = LogNorm(vmin=1, vmax=8)
A_TICKS = [1, 1.5, 2, 3, 4, 6, 8]
A_CMAP = plt.get_cmap('viridis').copy()
A_CMAP.set_bad('0.85')
BEAR_CMAP = 'twilight'
FALLBACK = '#d6246e'

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'savefig.dpi': 400, 'figure.dpi': 100,
})

SHORT = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10',
         'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40',
         'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10',
         'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit',
         'SatCLIP L40 released': 'SatCLIP ViT-L40'}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    if MS is not None and MS.exists():
        shutil.copyfile(OUT / name, MS / name)
    print('wrote', name)


def latlon_label(lat, lon):
    la = '0°' if abs(lat) < 1e-9 else f"{abs(lat):g}°{'N' if lat > 0 else 'S'}"
    lo = '0°' if abs(lon) < 1e-9 else f"{abs(lon):g}°{'E' if lon > 0 else 'W'}"
    return f'{la}, {lo}'


def sci(x):
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return rf'{m:.1f}\times10^{{{e}}}'


# ---------------------------------------------------------------- Equal Earth
EE_A = (1.340264, -0.081106, 0.000893, 0.003796)


def ee(lat, lon):
    """Equal Earth projection (unit sphere), degrees in."""
    a1, a2, a3, a4 = EE_A
    phi, lam = np.radians(lat), np.radians(lon)
    t = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    t2 = t * t
    t6 = t2 ** 3
    x = 2 * np.sqrt(3) * lam * np.cos(t) / (3 * (a1 + 3 * a2 * t2 + t6 * (7 * a3 + 9 * a4 * t2)))
    y = t * (a1 + a2 * t2 + t6 * (a3 + a4 * t2))
    return x, y


def ee_frame(lat, lon, eps=1e-3):
    """Unit vectors of local north and east on the Equal Earth map (orthonormal, north along meridian)."""
    x1, y1 = ee(lat + eps, lon)
    x0, y0 = ee(lat - eps, lon)
    n = np.array([x1 - x0, y1 - y0])
    n /= np.linalg.norm(n)
    e = np.array([n[1], -n[0]])
    return n, e





def ee_basemap(ax):
    lat = np.linspace(-90, 90, 181)
    xl, yl = ee(lat, -180 + 0 * lat)
    xr, yr = ee(lat[::-1], 180 + 0 * lat)
    ax.add_patch(Polygon(np.c_[np.r_[xl, xr], np.r_[yl, yr]], closed=True, fill=False, lw=0.6, ec='0.3', zorder=6))
    for lo in range(-150, 181, 30):
        la = np.linspace(-90, 90, 91)
        x, y = ee(la, lo + 0 * la)
        ax.plot(x, y, color='0.55', lw=0.25, zorder=4)
    for la in range(-60, 61, 30):
        lo = np.linspace(-180, 180, 181)
        x, y = ee(la + 0 * lo, lo)
        ax.plot(x, y, color='0.55', lw=0.25, zorder=4)
    LO, LA = np.meshgrid(COAST['lon'], COAST['lat'])
    X, Y = ee(LA, LO)
    ax.contour(X, Y, COAST['land'], levels=[0.5], colors='0.15', linewidths=0.35, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.36, 1.36)
    ax.axis('off')


def ee_field(ax, values, norm, cmap):
    lat_e = np.r_[GLAT[:, 0] - 1.25, GLAT[-1, 0] + 1.25]
    lon_e = np.r_[GLON[0] - 2.5, GLON[0, -1] + 2.5]
    LO, LA = np.meshgrid(lon_e, lat_e)
    X, Y = ee(LA, LO)
    return ax.pcolormesh(X, Y, np.ma.masked_invalid(values.reshape(GLAT.shape)), norm=norm, cmap=cmap,
                         shading='flat', rasterized=True, zorder=1)


def glyphs(ax, A, theta, fallback=None, lat_step=20, lon_step=30, size=0.075, color='white', max_lat=60):
    """Equal-area stretch ellipses in each site's local north--east frame."""
    A = A.reshape(GLAT.shape)
    theta = theta.reshape(GLAT.shape)
    fb = np.zeros(GLAT.shape, bool) if fallback is None else fallback.reshape(GLAT.shape)
    rows = [i for i, la in enumerate(GLAT[:, 0]) if abs(la % lat_step) < 1e-9 and abs(la) <= max_lat]
    cols = [j for j, lo in enumerate(GLON[0]) if abs((lo + 2.5) % lon_step) < 1e-9]
    t = np.linspace(0, 2 * np.pi, 60)
    for i in rows:
        for j in cols:
            if fb[i, j] or not np.isfinite(A[i, j]):
                continue
            a = size * np.sqrt(min(A[i, j], 12))
            b = size / np.sqrt(min(A[i, j], 12))
            u = np.array([np.cos(theta[i, j]), np.sin(theta[i, j])])  # (north, east)
            v = np.array([-u[1], u[0]])
            ne = np.outer(a * np.cos(t), u) + np.outer(b * np.sin(t), v)
            n, e = ee_frame(GLAT[i, j], GLON[i, j])
            cx, cy = ee(GLAT[i, j], GLON[i, j])
            pts = np.c_[cx + ne[:, 0] * n[0] + ne[:, 1] * e[0], cy + ne[:, 0] * n[1] + ne[:, 1] * e[1]]
            ax.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=0.55, zorder=7,
                                 path_effects=[pe.withStroke(linewidth=1.3, foreground='0.1', alpha=0.6)]))


def a_colorbar(fig, mappable, ax, label='Anisotropy $A$ (log scale)', **kw):
    cb = fig.colorbar(mappable, ax=ax, ticks=A_TICKS, extend='max', **kw)
    cb.ax.set_yticklabels([f'{t:g}' for t in A_TICKS]) if kw.get('orientation', 'vertical') == 'vertical' else cb.ax.set_xticklabels([f'{t:g}' for t in A_TICKS])
    cb.ax.minorticks_off()
    cb.set_label(label)
    cb.outline.set_linewidth(0.5)
    return cb


# ------------------------------------------------------------ figures
def schematic():
    A, th = 2.0, np.radians(30)
    u = np.array([np.sin(th), np.cos(th)])  # (x=east, y=north) of greatest stretch
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 2 * np.pi, 200)
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.15))
    L, Rc = np.array([-1.55, 0.0]), np.array([1.55, 0.0])

    def arrow(ax, text):
        ax.add_patch(FancyArrowPatch((-0.8, 0), (0.55, 0), arrowstyle='-|>', mutation_scale=8, lw=0.7, color='0.3'))
        ax.text(-0.12, 0.12, text, ha='center', va='bottom', fontsize=7.1, color='0.3')

    def ne_cross(ax, c, r=0.28):
        for d, lab, ha, va in [((0, r), 'N', 'center', 'bottom'), ((r, 0), 'E', 'left', 'center')]:
            ax.annotate('', xy=(c[0] + d[0], c[1] + d[1]), xytext=c,
                        arrowprops=dict(arrowstyle='-|>', lw=0.5, color='0.45', mutation_scale=5, shrinkA=0, shrinkB=0))
            ax.text(c[0] + 1.2 * d[0], c[1] + 1.2 * d[1], lab, ha=ha, va=va, fontsize=6.9, color='0.45')

    # (a) map projection
    ax = axs[0]
    ax.add_patch(Circle(L, 0.55, fill=False, lw=1.0, color='#2166ac'))
    rot = 0.35
    e1, e2 = np.array([np.cos(rot), np.sin(rot)]), np.array([-np.sin(rot), np.cos(rot)])
    ell = np.outer(0.55 * np.sqrt(A) * np.cos(t), e1) + np.outer(0.55 / np.sqrt(A) * np.sin(t), e2)
    ax.plot(Rc[0] + ell[:, 0], Rc[1] + ell[:, 1], color='#c23b32', lw=1.0)
    arrow(ax, 'projection')
    ax.text(L[0], -0.8, 'small circle\non the Earth', ha='center', va='top', fontsize=7.1)
    ax.text(Rc[0], -0.8, 'ellipse on\nthe map', ha='center', va='top', fontsize=7.1)
    ax.set_title('(a) Tissot indicatrix', loc='left', fontsize=7.5)

    # (b) location encoder
    ax = axs[1]
    ring = np.c_[np.sin(B), np.cos(B)] * 0.55
    ax.plot(L[0] + ring[:, 0], L[1] + ring[:, 1], 'o', ms=1.4, color='#2166ac')
    ax.plot(*L, 'k*', ms=3.5)
    fwd = np.outer(0.55 * np.sqrt(A) * np.cos(t), np.array([1, 0])) + np.outer(0.55 / np.sqrt(A) * np.sin(t), np.array([0, 1]))
    ax.plot(Rc[0] + fwd[:, 0], Rc[1] + fwd[:, 1], color='#c23b32', lw=1.0)
    ax.plot([Rc[0] - 0.55 * np.sqrt(A), Rc[0] + 0.55 * np.sqrt(A)], [0, 0], color='k', lw=0.6)
    ax.text(Rc[0], 0.44, 'greatest stretch', ha='center', va='bottom', fontsize=6.9)
    arrow(ax, 'encoder $f$')
    ax.text(L[0], -0.8, '32 probes at\nradius $r$', ha='center', va='top', fontsize=7.1)
    ax.text(Rc[0], -0.8, 'plane in\nfeature space', ha='center', va='top', fontsize=7.1)
    ax.set_title('(b) Forward indicatrix', loc='left', fontsize=7.5)

    # (c) reach ellipse on the ground
    ax = axs[2]
    R0 = 0.62
    reach = np.outer(R0 / np.sqrt(A) * np.cos(t), u) + np.outer(R0 * np.sqrt(A) * np.sin(t), v)
    ax.add_patch(Circle((0, 0), R0, fill=False, lw=0.8, ls='--', color='#2166ac'))
    ax.plot(reach[:, 0], reach[:, 1], color='#e68a28', lw=1.1)
    ax.plot(0, 0, 'k*', ms=3.5)
    g = R0 / np.sqrt(A) * u
    ax.annotate('', xy=g, xytext=-g, arrowprops=dict(arrowstyle='<->', lw=0.6, color='k', mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(*(g + np.array([0.32, 0.12])), 'greatest\nstretch', fontsize=6.9, ha='left', va='bottom')
    h = R0 * np.sqrt(A) * v
    ax.text(*(h + np.array([-0.08, 0.0])), 'longest\nreach', fontsize=6.9, ha='right', va='center')
    ne_cross(ax, np.array([1.35, -0.95]))
    ax.text(0, -1.0, 'equal predicted feature\ndistance (orange) vs equal\ngeographic distance (blue)',
            ha='center', va='top', fontsize=6.9)
    ax.set_title('(c) Reach ellipse', loc='left', fontsize=7.5)
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-2.55, 2.55)
        ax.set_ylim(-1.75, 1.1)
        ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.08)
    save(fig, 'fig_tissot_schematic.png')


def geoclip_equal_earth():
    D = np.load(DATA / 'profiles/global.npz')
    f = fit_many(D['prof__GeoCLIP'], B)
    a_ee, th_ee = equal_earth_metric(GLAT.ravel(), GLON.ravel())
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 3.45), gridspec_kw=dict(hspace=0.12, wspace=0.03))
    for col, (name, A, th, fb) in enumerate([('Equal Earth', a_ee, th_ee, None),
                                             ('GeoCLIP, 1 km', f['A_fit'], f['theta_fit'], f['fallback'])]):
        ax = axs[0, col]
        m = ee_field(ax, A, A_NORM, A_CMAP)
        ee_basemap(ax)
        glyphs(ax, A, th, fb)
        ax.set_title(f'{name}: anisotropy')
        ax = axs[1, col]
        bt = np.degrees(th).copy()
        if fb is not None:
            bt[fb] = np.nan
        mb = ee_field(ax, bt, Normalize(0, 180), BEAR_CMAP)
        ee_basemap(ax)
        if fb is not None:
            x, y = ee(GLAT.ravel()[fb], GLON.ravel()[fb])
            ax.scatter(x, y, s=5, marker='x', color=FALLBACK, linewidths=0.6, zorder=8)
        ax.set_title(f'{name}: bearing')
    a_colorbar(fig, m, axs[0, :].tolist(), fraction=0.025, pad=0.01, aspect=14)
    cb = fig.colorbar(mb, ax=axs[1, :].tolist(), fraction=0.025, pad=0.01, aspect=14, ticks=[0, 45, 90, 135, 180])
    cb.set_label('Bearing (° from north)')
    cb.outline.set_linewidth(0.5)
    save(fig, 'geoclip_equal_earth.png')


def indicatrix():
    D = np.load(DATA / 'profiles/global.npz')
    th = np.linspace(0, 2 * np.pi, 361)
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 6.6), subplot_kw={'projection': 'polar'})
    for i, (name, radius) in enumerate([('GeoCLIP', 1), ('SatCLIP vit16l40', 1), ('SINR cap1000', 0.3)]):
        for j, (lat, lon) in enumerate([(40, -105), (0, 20), (75, 20)]):
            idx = np.argmin((GLAT.ravel() - lat) ** 2 + (GLON.ravel() - lon) ** 2)
            la, lo = GLAT.ravel()[idx], GLON.ravel()[idx]
            p = D['prof__' + name][idx]
            p = p / np.exp(np.mean(np.log(np.maximum(p, 1e-30))))
            f = fit_ellipse(p, B)
            curve = np.sqrt(f['lambda1'] * np.cos(th - f['theta_fit']) ** 2 + f['lambda2'] * np.sin(th - f['theta_fit']) ** 2)
            ax = axes[i, j]
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.plot(np.r_[B, B[0]], np.r_[p, p[0]], 'o-', ms=1.8, lw=0.6, color='#2166ac')
            ax.plot(th, curve, c='#c23b32', lw=0.9)
            ang = f['theta_fit']
            ax.plot([ang, ang + np.pi], [curve.max(), curve.max()], c='k', lw=0.7)
            ax.set_title(f"{SHORT[name]}, {radius:g} km\n{latlon_label(la, lo)}\n"
                         f"$A_r$ = {f['A_fit']:.2f}; {np.degrees(ang):.0f}°; $\\nu$ = ${sci(f['ne'])}$",
                         fontsize=6.6, pad=8, linespacing=1.15)
            ax.set_yticklabels([])
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
            ax.tick_params(labelsize=6, pad=-1)
            ax.grid(alpha=0.3, lw=0.4)
    handles = [Line2D([], [], color='#2166ac', marker='o', ms=2.5, lw=0.6, label='Measured stretch (32 bearings)'),
               Line2D([], [], color='#c23b32', lw=0.9, label='Fitted tensor curve'),
               Line2D([], [], color='k', lw=0.7, label='Principal axis (greatest stretch)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.8)
    save(fig, 'fig_indicatrix_v2.png')


def legendre(l, x):
    c = np.zeros(l + 1)
    c[l] = 1
    return np.polynomial.legendre.legval(x, c)


def validation():
    G = np.load(DATA / 'profiles/global.npz')
    I = np.load(DATA / 'profiles/input_references.npz')
    lat = GLAT[:, 0]
    phi = np.radians(lat)

    def sh_reference(degree):
        t = phi + np.pi / 2
        x, sn = np.cos(t), np.sin(t)
        z = np.zeros_like(x)
        s0 = sum((2 * l + 1) * l * (l + 1) / (8 * np.pi) for l in range(degree))
        for l in range(1, degree):
            dp = -l * (x * legendre(l, x) - legendre(l - 1, x)) / sn
            z += (2 * l + 1) / (4 * np.pi) * dp ** 2
        return np.sqrt(1 + (np.pi ** 2 - 1) * z / s0)

    def zonal(profile):
        return np.median(fit_many(profile, B)['A_fit'].reshape(GLAT.shape), axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.25), sharey=True)
    axs[0].plot(lat, 1 / np.cos(phi), 'k-', lw=0.9, label=r'Analytic $\sec\varphi$')
    axs[0].plot(lat, np.ones_like(lat), 'k:', lw=0.9, label='Isotropic')
    for key, label, marker in [('grid', 'Planar sinusoids', 'o'), ('cart3d', '3-D Cartesian', '^')]:
        axs[0].plot(lat, zonal(G['prof__ref ' + key]), marker, ms=2.2, mfc='none', mew=0.6,
                    label=f"{label}, {float(G['r__ref ' + key]):g} km")
    axs[0].set_title('Planar and isotropic inputs')
    axs[1].plot(lat, np.maximum(2 * np.cos(phi), 1 / (2 * np.cos(phi))), 'k-', lw=0.9, label='Analytic 2:1')
    axs[1].plot(lat, zonal(G['prof__ref SINR-input']), 'o', ms=2.2, mfc='none', mew=0.6,
                label=f"Wrap input, {float(G['r__ref SINR-input']):g} km")
    axs[1].set_title('SINR wrap input')
    for degree, marker, style in [(10, 'o', '-'), (40, 's', '--')]:
        axs[2].plot(lat, sh_reference(degree), 'k' + style, lw=0.9, label=rf'Analytic, $L={degree}$')
        axs[2].plot(lat, zonal(I[f'in__SatCLIP vit16l{degree}']), marker, ms=2.2, mfc='none', mew=0.6,
                    label=rf'Implemented, $L={degree}$, 1 km')
    axs[2].set_title('SatCLIP harmonic input')
    for ax in axs:
        ax.set_yscale('log')
        ax.set_ylim(0.9, 12)
        ax.set_yticks([1, 1.5, 2, 3, 5, 10])
        ax.set_yticklabels(['1', '1.5', '2', '3', '5', '10'])
        ax.minorticks_off()
        ax.set_xlabel('Latitude (°)')
        ax.set_xticks([-80, -40, 0, 40, 80])
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=5.8, frameon=False, loc='upper center', handlelength=1.6)
    axs[0].set_ylabel('Anisotropy $A$ (log scale)')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_validation_v2.png')


def bearing_ticks(ax, lon, lat, a, b, fb, skip, length):
    yy, xx = np.meshgrid(lat[::skip], lon[::skip], indexing='ij')
    th = b[::skip, ::skip]
    mask = (a[::skip, ::skip] > 1 + 1e-6) & ~fb[::skip, ::skip]
    dx = length * np.sin(th) / np.cos(np.radians(yy))
    dy = length * np.cos(th)
    segs = np.stack([np.c_[xx[mask] - dx[mask] / 2, yy[mask] - dy[mask] / 2],
                     np.c_[xx[mask] + dx[mask] / 2, yy[mask] + dy[mask] / 2]], axis=1)
    ax.add_collection(LineCollection(segs, colors='white', linewidths=0.45, zorder=4,
                                     path_effects=[pe.withStroke(linewidth=1.1, foreground='0.05', alpha=0.7)]))


def dense():
    D = dict(np.load(DATA / 'profiles/dense_regional.npz'))
    D.update(dict(np.load(OUTPUT / 'profiles/dense_fitted.npz')))
    panels = [('CSP iNat2018__NA', 'CSP-iNat, North America', 0.1), ('CSP iNat2018__AF', 'CSP-iNat, Central Africa', 0.1),
              ('SINR cap1000__NA', 'SINR cap-1000', 0.3), ('Climplicit__NA', 'Climplicit', 0.1),
              ('SatCLIP r18l40__NA', 'SatCLIP R18-L40', 1.0)]
    fig = plt.figure(figsize=(WIDTH, 4.2))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=1.6, top=0.95, bottom=0.17)
    slots = [gs[0, :2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    LO, LA = np.meshgrid(COAST['lon'], COAST['lat'])
    for slot, (key, label, r) in zip(slots, panels):
        lat, lon = D['lat__' + key], D['lon__' + key]
        a, b, fb = D['A__' + key], D['B__' + key], D['fallback__' + key].astype(bool)
        ax = fig.add_subplot(slot)
        im = ax.pcolormesh(lon, lat, a, shading='auto', cmap=A_CMAP, norm=A_NORM, rasterized=True)
        ax.contour(LO, LA, COAST['land'], levels=[0.5], colors='0.95', linewidths=0.4)
        skip = max(1, len(lon) // 16)
        bearing_ticks(ax, lon, lat, a, b, fb, skip, (lat[-1] - lat[0]) / 22)
        fy, fx = np.nonzero(fb)
        ax.scatter(lon[fx], lat[fy], s=4, marker='x', color=FALLBACK, linewidths=0.5, zorder=5)
        ax.set_xlim(lon[0], lon[-1])
        ax.set_ylim(lat[0], lat[-1])
        ax.set_aspect(1 / np.cos(np.radians(0.5 * (lat[0] + lat[-1]))))
        ax.set_title(f'{label}\n{r:g} km; median $A$ = {np.median(a):.2f}', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Longitude (°)', fontsize=6.5)
        if slot in (slots[0], slots[3]):
            ax.set_ylabel('Latitude (°)', fontsize=6.5)
    cax = fig.add_axes([0.25, 0.085, 0.5, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=A_TICKS, extend='max')
    cb.ax.set_xticklabels([f'{t:g}' for t in A_TICKS])
    cb.ax.minorticks_off()
    cb.set_label('Anisotropy $A$ (log scale)')
    save(fig, 'dense_regional_v2.png')


DENSE_PANELS = [('CSP iNat2018__NA', 'CSP-iNat, North America', 0.1), ('CSP iNat2018__AF', 'CSP-iNat, Central Africa', 0.1),
                ('SINR cap1000__NA', 'SINR cap-1000', 0.3), ('Climplicit__NA', 'Climplicit', 0.1),
                ('SatCLIP r18l40__NA', 'SatCLIP R18-L40', 1.0)]


def ee_box(ax, lat0, lat1, lon0, lon1, label=None):
    la = np.linspace(lat0, lat1, 40)
    lo = np.linspace(lon0, lon1, 40)
    path = np.r_[np.c_[la, lon0 + 0 * la], np.c_[lat1 + 0 * lo, lo], np.c_[la[::-1], lon1 + 0 * la],
                 np.c_[lat0 + 0 * lo, lo[::-1]]]
    x, y = ee(path[:, 0], path[:, 1])
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=False, ec='white', lw=0.9, zorder=9,
                         path_effects=[pe.withStroke(linewidth=1.8, foreground='0.05')]))
    if label:
        tx, ty = ee(lat1, lon0)
        ax.text(tx, ty + 0.03, label, color='white', fontsize=6.5, fontweight='bold', ha='left', va='bottom', zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='0.05')])


def dense_world():
    """Fixed-grid world maps (as in the GeoCLIP figure) above the dense regional maps."""
    G = np.load(DATA / 'profiles/global.npz')
    D = dict(np.load(DATA / 'profiles/dense_regional.npz'))
    D.update(dict(np.load(OUTPUT / 'profiles/dense_fitted.npz')))
    world = [('CSP iNat2018', 'CSP-iNat, 0.1 km', ['e', 'f']), ('SINR cap1000', 'SINR cap-1000, 0.3 km', ['g']),
             ('Climplicit', 'Climplicit, 0.1 km', ['h']), ('SatCLIP r18l40', 'SatCLIP R18-L40, 1 km', ['i'])]
    letters = dict(zip([k for k, _, _ in DENSE_PANELS], 'efghi'))
    fig = plt.figure(figsize=(WIDTH, 6.45))
    gs = fig.add_gridspec(4, 6, height_ratios=[0.78, 0.78, 1, 1], hspace=0.36, wspace=1.6,
                          top=0.975, bottom=0.09, left=0.06, right=0.985)
    wslots = [gs[0, :3], gs[0, 3:], gs[1, :3], gs[1, 3:]]
    for slot, (name, label, regs), tag in zip(wslots, world, 'abcd'):
        ax = fig.add_subplot(slot)
        f = fit_many(G['prof__' + name], B)
        m = ee_field(ax, f['A_fit'], A_NORM, A_CMAP)
        ee_basemap(ax)
        glyphs(ax, f['A_fit'], f['theta_fit'], f['fallback'], size=0.07)
        fb = f['fallback']
        x, y = ee(GLAT.ravel()[fb], GLON.ravel()[fb])
        ax.scatter(x, y, s=4, marker='x', color=FALLBACK, linewidths=0.5, zorder=8)
        for key in [k for k, _, _ in DENSE_PANELS if k.split('__')[0] == name]:
            lat, lon = D['lat__' + key], D['lon__' + key]
            ee_box(ax, lat[0], lat[-1], lon[0], lon[-1], f'({letters[key]})')
        ax.set_title(f'({tag}) {label}', fontsize=7)
    LO, LA = np.meshgrid(COAST['lon'], COAST['lat'])
    slots = [gs[2, :2], gs[2, 2:4], gs[2, 4:6], gs[3, 1:3], gs[3, 3:5]]
    for slot, (key, label, r) in zip(slots, DENSE_PANELS):
        lat, lon = D['lat__' + key], D['lon__' + key]
        a, b, fb = D['A__' + key], D['B__' + key], D['fallback__' + key].astype(bool)
        ax = fig.add_subplot(slot)
        im = ax.pcolormesh(lon, lat, a, shading='auto', cmap=A_CMAP, norm=A_NORM, rasterized=True)
        ax.contour(LO, LA, COAST['land'], levels=[0.5], colors='0.95', linewidths=0.4)
        skip = max(1, len(lon) // 16)
        bearing_ticks(ax, lon, lat, a, b, fb, skip, (lat[-1] - lat[0]) / 22)
        fy, fx = np.nonzero(fb)
        ax.scatter(lon[fx], lat[fy], s=4, marker='x', color=FALLBACK, linewidths=0.5, zorder=5)
        ax.set_xlim(lon[0], lon[-1])
        ax.set_ylim(lat[0], lat[-1])
        ax.set_aspect(1 / np.cos(np.radians(0.5 * (lat[0] + lat[-1]))))
        ax.set_title(f'({letters[key]}) {label}\n{r:g} km; median $A$ = {np.median(a):.2f}', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Longitude (°)', fontsize=6.5)
        if slot in (slots[0], slots[3]):
            ax.set_ylabel('Latitude (°)', fontsize=6.5)
    cax = fig.add_axes([0.25, 0.032, 0.5, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=A_TICKS, extend='max')
    cb.ax.set_xticklabels([f'{t:g}' for t in A_TICKS])
    cb.ax.minorticks_off()
    cb.set_label('Anisotropy $A$ (log scale)')
    save(fig, 'dense_regional_v3.png')


def satclip_backbones():
    G = np.load(DATA / 'profiles/global.npz')
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 2.75), gridspec_kw=dict(hspace=0.12, wspace=0.03))
    for ax, (name, label) in zip(axs.flat, [(f'SatCLIP {tag}l{d}', f'{lab}-L{d}') for d in [10, 40]
                                            for tag, lab in [('vit16', 'ViT'), ('r18', 'R18'), ('r50', 'R50')]]):
        f = fit_many(G['prof__' + name], B)
        m = ee_field(ax, f['A_fit'], A_NORM, A_CMAP)
        ee_basemap(ax)
        glyphs(ax, f['A_fit'], f['theta_fit'], f['fallback'], size=0.07)
        fb = f['fallback']
        x, y = ee(GLAT.ravel()[fb], GLON.ravel()[fb])
        ax.scatter(x, y, s=4, marker='x', color=FALLBACK, linewidths=0.5, zorder=8)
        ax.set_title(f'SatCLIP {label}', fontsize=7.5)
    a_colorbar(fig, m, axs.ravel().tolist(), fraction=0.02, pad=0.01, aspect=18)
    save(fig, 'satclip_backbones_ee.png')


def other_families():
    """Fixed-grid world maps for CSP, Climplicit and SINR, in the style of the SatCLIP figure."""
    G = np.load(DATA / 'profiles/global.npz')
    items = [('CSP iNat2018', 'CSP-iNat, 0.1 km'), ('CSP fMoW', 'CSP-fMoW, 0.1 km'), ('Climplicit', 'Climplicit, 0.1 km'),
             ('SINR cap1000', 'SINR cap-1000, 0.3 km'), ('SINR cap10', 'SINR cap-10, 0.3 km'),
             ('SINR distilled', 'SINR distilled, 0.3 km')]
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 2.75), gridspec_kw=dict(hspace=0.12, wspace=0.03))
    for ax, (name, label) in zip(axs.flat, items):
        f = fit_many(G['prof__' + name], B)
        m = ee_field(ax, f['A_fit'], A_NORM, A_CMAP)
        ee_basemap(ax)
        glyphs(ax, f['A_fit'], f['theta_fit'], f['fallback'], size=0.07)
        fb = f['fallback']
        x, y = ee(GLAT.ravel()[fb], GLON.ravel()[fb])
        ax.scatter(x, y, s=4, marker='x', color=FALLBACK, linewidths=0.5, zorder=8)
        ax.set_title(label, fontsize=7.5)
    a_colorbar(fig, m, axs.ravel().tolist(), fraction=0.02, pad=0.01, aspect=18)
    save(fig, 'other_families_ee.png')


def neighbourhood_maps():
    import json
    d = np.load(DATA / 'measurements/neighbourhoods.npz')
    meta = json.loads(str(d['metadata']))
    names = [m['name'] for m in meta]
    queries = np.degrees(d['queries'])

    def select(mi, qi, ri=1, k=40):
        delta = d['displacements'][ri, qi]
        fit = fit_ellipse(d[f'profiles_{mi}'][1, qi], B)
        geo = np.linalg.norm(delta, axis=1)
        act = d[f'distances_{mi}'][ri, qi]
        order = lambda x: np.argsort(x, kind='stable')[:k]
        pred = None if fit['fallback'] else np.einsum('ni,ij,nj->n', delta, fitted_tensor(fit), delta)
        return delta, fit, order(geo), order(act), None if pred is None else order(pred)

    chosen = [('GeoCLIP', 0, 0), ('SatCLIP L40 released', 70, 0), ('SINR distilled', 60, 0),
              ('CSP iNat2018', 0, 170), ('Climplicit', 60, 60)]
    fig = plt.figure(figsize=(WIDTH, 6.9))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28)
    for slot, (name, lat, lon) in zip([gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[2, 0]], chosen):
        ax = fig.add_subplot(slot)
        mi = names.index(name)
        qi = int(np.flatnonzero(np.all(np.isclose(queries, [lat, lon]), axis=1))[0])
        delta, fit, g, e, t = select(mi, qi)
        east, north = delta[:, 1], delta[:, 0]
        ax.scatter(east, north, s=2.5, c='0.82', zorder=1, linewidths=0)
        ax.add_patch(Circle((0, 0), float(np.linalg.norm(delta[g], axis=1).max()), fill=False, ec='#2166ac', ls='--', lw=0.8, zorder=2))
        thr = float(np.sort(d[f'distances_{mi}'][1, qi], kind='stable')[39])
        G = fitted_tensor(fit)
        tt = np.linspace(0, 2 * np.pi, 361)
        dirs = np.c_[np.cos(tt), np.sin(tt)]
        bnd = dirs * (thr / np.sqrt(np.einsum('ni,ij,nj->n', dirs, G, dirs)))[:, None]
        ax.plot(bnd[:, 1], bnd[:, 0], color='#e68a28', lw=1.0, zorder=2)
        ax.scatter(east[g], north[g], s=20, facecolors='none', edgecolors='#2166ac', lw=0.55, zorder=2)
        ax.scatter(east[e], north[e], s=7, c='#e68a28', zorder=3, linewidths=0)
        ax.scatter(east[t], north[t], s=14, c='#16835c', marker='x', linewidths=0.55, zorder=4)
        ax.scatter([0], [0], marker='*', s=40, c='k', zorder=5)
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect='equal')
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.grid(alpha=0.15, lw=0.4)
        eg = len(np.intersect1d(g, e))
        et = len(np.intersect1d(e, t))
        ax.text(0.5, 0.03, f'E/G {eg}/40   E/T {et}/40', transform=ax.transAxes, ha='center', fontsize=6.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))
        ax.set_title(f'{SHORT[name]}; {latlon_label(lat, lon)}', fontsize=7.5)
        ax.set_xlabel('East displacement (km)', fontsize=6.5)
        ax.set_ylabel('North displacement (km)', fontsize=6.5)
    # Overlap counts across all valid comparisons
    pairs = {'E/G': [], 'E/T': []}
    for mi in range(len(meta)):
        for qi in range(len(queries)):
            _, _, g, e, t = select(mi, qi)
            if t is None:
                continue
            pairs['E/G'].append(len(np.intersect1d(e, g)))
            pairs['E/T'].append(len(np.intersect1d(e, t)))
    sub = gs[2, 1].subgridspec(2, 1, hspace=1.05)
    lo = min(min(v) for v in pairs.values())
    edges = np.arange(lo - 0.5, 41.5, 1)
    for row, (key, title) in enumerate([('E/G', 'Embedding vs geographic'), ('E/T', 'Embedding vs tensor')]):
        ax = fig.add_subplot(sub[row])
        ax.hist(pairs[key], bins=edges, color='0.35', rwidth=0.8)
        ax.set_title(f'{title} ({len(pairs[key])} comparisons)', fontsize=7, loc='left')
        ax.set_xlim(lo - 0.8, 40.8)
        ax.set_ylabel('Count', fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(axis='y', alpha=0.2, lw=0.4)
        ax.set_axisbelow(True)
        if row == 1:
            ax.set_xlabel('Shared neighbors (of 40)', fontsize=6.5)
    handles = [Line2D([], [], marker='o', ls='', mfc='none', mec='#2166ac', ms=5, label='Geographic neighbors'),
               Line2D([], [], marker='o', ls='', color='#e68a28', ms=3.5, label='Embedding neighbors'),
               Line2D([], [], marker='x', ls='', color='#16835c', ms=4.5, label='Tensor-predicted neighbors'),
               Line2D([], [], color='#e68a28', lw=1.0, label='Reach ellipse'),
               Line2D([], [], color='#2166ac', ls='--', lw=0.8, label='Fortieth geographic neighbor'),
               Line2D([], [], marker='*', ls='', color='k', ms=6, label='Query')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035), fontsize=6.8)
    save(fig, 'neighbourhood_maps_v2.png')


def run(workspace):
    """Regenerate the journal figures from declared inputs in a Workspace."""
    renderer = os.environ.get('ETISSOT_FIGURE_LIBRARIES')
    if renderer and os.environ.get('ETISSOT_FIGURE_CHILD') != '1':
        libraries = Path(renderer).expanduser().resolve()
        if not (libraries/'matplotlib').is_dir():
            raise ValueError('ETISSOT_FIGURE_LIBRARIES must contain the selected matplotlib installation.')
        env = dict(os.environ)
        env['ETISSOT_FIGURE_CHILD'] = '1'
        paths = [str(libraries), str(Path(__file__).resolve().parents[2])]
        if env.get('PYTHONPATH'):
            paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(paths)
        subprocess.run([sys.executable, '-m', 'etissot.reporting.manuscript_figures',
                        '--data', str(workspace.data), '--output', str(workspace.output)],
                       env=env, stdout=sys.stdout, stderr=sys.stderr, check=True)
        return
    global DATA, OUTPUT, OUT, COAST
    DATA, OUTPUT = workspace.data, workspace.output
    OUT = OUTPUT / 'figures'
    COAST = np.load(DATA / 'reference/maps/coastlines.npz', allow_pickle=False)
    try:
        for function in [schematic, geoclip_equal_earth, indicatrix, validation,
                         dense, other_families, satclip_backbones, neighbourhood_maps]:
            function()
    finally:
        COAST.close()
    (OUTPUT/'summaries/figure_environment.json').write_text(json.dumps({
        'matplotlib':matplotlib.__version__, 'numpy':np.__version__,
        'python':sys.version.split()[0], 'manuscript_modified':False,
        'pixel_identity':'Checked separately by manuscript validation; not assumed from version match.'
    },indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    from etissot.workspace import Workspace
    parser=argparse.ArgumentParser(description='Regenerate manuscript figures without editing manuscript files.')
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(Workspace(args.data,args.output))

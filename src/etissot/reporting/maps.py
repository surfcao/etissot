"""Plot dense regional, backbone and input-reference maps."""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, GLON
from scipy.special import eval_legendre

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    ROOT = workspace.data
    P = workspace.output
    matplotlib.use('Agg')
    D = dict(np.load(ROOT / 'profiles/dense_regional.npz'))
    if not (P / 'profiles/dense_fitted.npz').exists():
        from etissot.diagnostics.dense_cached import run as refit_dense
        refit_dense(workspace)
    D.update(dict(np.load(P / 'profiles/dense_fitted.npz')))
    G = np.load(ROOT / 'profiles/global.npz')
    C = np.load(ROOT / 'reference/maps/coastlines.npz')
    B = np.arange(32) * 2 * np.pi / 32
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8})

    def panel(ax, lat, lon, a, b, title, vmax, skip, length, fallback):
        im = ax.pcolormesh(lon, lat, a, shading='auto', cmap='viridis', norm=LogNorm(vmin=1, vmax=vmax))
        ax.contour(C['lon'], C['lat'], C['land'], levels=[0.5], colors='.9', linewidths=0.4, alpha=0.8)
        yy, xx = np.meshgrid(lat[::skip], lon[::skip], indexing='ij')
        th = b[::skip, ::skip]
        dx = length * np.sin(th) / np.cos(np.radians(yy))
        dy = length * np.cos(th)
        mask = (a[::skip, ::skip] > 1 + 1e-06) & ~fallback[::skip, ::skip]
        ax.quiver(xx[mask], yy[mask], dx[mask], dy[mask], angles='xy', scale_units='xy', scale=1, color='white', alpha=0.55, width=0.002, pivot='mid', headwidth=0, headlength=0, headaxislength=0)
        fx, fy = np.meshgrid(lon, lat)
        ax.scatter(fx[fallback], fy[fallback], s=12, marker='x', color='#ff3282', linewidths=0.65, zorder=5)
        ax.set_xlim(lon[0], lon[-1])
        ax.set_ylim(lat[0], lat[-1])
        ax.set_title(title)
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        return im
    panels = [('CSP iNat2018__NA', 'CSP-iNat: North America', 0.1), ('CSP iNat2018__AF', 'CSP-iNat: Central Africa', 0.1), ('SINR cap1000__NA', 'SINR cap-1000', 0.3), ('Climplicit__NA', 'Climplicit', 0.1), ('SatCLIP r18l40__NA', 'SatCLIP R18-L40', 1.0)]
    fig = plt.figure(figsize=(12, 7.1))
    gs = fig.add_gridspec(2, 6)
    slots = [gs[0, :2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for slot, (key, label, r) in zip(slots, panels):
        lat = D['lat__' + key]
        lon = D['lon__' + key]
        a = D['A__' + key]
        b = D['B__' + key]
        ax = fig.add_subplot(slot)
        skip = max(1, len(lon) // 22)
        im = panel(ax, lat, lon, a, b, label + f'\n{r:g} km; median A = {np.median(a):.2f}', 8, skip, (lat[-1] - lat[0]) / 35, D['fallback__' + key])
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=[1, 2, 4, 8])
        cb.ax.set_yticklabels(['1', '2', '4', '8'])
        cb.set_label('Reported A')
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.text(0.5, 0.006, 'Pink crosses: sampled-minimum fallback; bearing ticks omitted at these sites.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(P / 'figures/dense_regional_maps.png', dpi=210, bbox_inches='tight')
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    for ax, (name, label) in zip(axs.flat, [(f'SatCLIP {tag}l{degree}', f'{label}-L{degree}') for degree in [10, 40] for tag, label in [('vit16', 'ViT'), ('r18', 'R18'), ('r50', 'R50')]]):
        f = fit_many(G['prof__' + name], B)
        a = f['A_fit'].reshape(GLAT.shape)
        b = f['theta_fit'].reshape(GLAT.shape)
        im = panel(ax, GLAT[:, 0], GLON[0], a, b, label, 4, 5, 7, f['fallback'].reshape(GLAT.shape))
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=[1, 2, 4])
        cb.ax.set_yticklabels(['1', '2', '4'])
        cb.set_label('Reported A')
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
    fig.text(0.5, 0.006, 'Pink crosses: sampled-minimum fallback; bearing ticks omitted at these sites.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(P / 'figures/satclip_backbones_maps.png', dpi=210, bbox_inches='tight')
    plt.close(fig)
    S = json.loads((P / 'summaries/dense_summary.json').read_text())
    rows = []
    for key, label, r in panels:
        q = S[key]
        rows.append(f"{label}&{q['step_degrees']:g}&{q['sites']:,}&{q['first_acf_below_half_degrees']:g}&{q['fallback_count']}" + '\\\\')
    text = "\\paragraph{Dense regional fields.} Five regional panels were re-evaluated with the operative checkpoint paths in journal supplement S7. All 32-bearing profiles are retained and fitted with the primary harmonic estimator. The two CSP grids use $0.1^\\circ$ spacing; the other grids use $0.5^\\circ$. Line segments are transformed from north--east bearings to longitude--latitude chart directions, accounting for latitude and axis scaling. Pink crosses identify the sampled-minimum fallback, and bearing ticks are omitted at those sites. We subtract each latitude row's mean log-anisotropy and average the Pearson correlations between row segments separated by each longitude lag. The first sampled lag with mean correlation below 0.5 is reported below. These are regional, longitude-direction summaries, resolved only at the stated grid spacing; they do not estimate an isotropic global range.\n\\begin{table}[htbp]\\centering\\small\n\\caption{Sampling and longitude-direction correlation scales of the current fitted regional fields. The half-correlation lag is in degrees of longitude.}\n\\label{tab:dense}\n\\begin{tabular}{@{}lrrrr@{}}\\toprule\nRegion / checkpoint&Grid ($^\\circ$)&Sites&Half-corr. lag&Fallbacks\\\\\\midrule\n" + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n\\end{table}\n'
    (P / 'tables/dense_protocol.tex').write_text(text)
    print(json.dumps({k: {q: v for q, v in row.items() if q != 'row_mean_longitude_acf'} for k, row in S.items()}, indent=2))
    I = np.load(ROOT / 'profiles/input_references.npz')
    lat = GLAT[:, 0]
    phi = np.radians(lat)

    def sh_reference(degree):
        t = phi + np.pi / 2
        x = np.cos(t)
        sn = np.sin(t)
        z = np.zeros_like(x)
        s0 = sum(((2 * l + 1) * l * (l + 1) / (8 * np.pi) for l in range(degree)))
        for l in range(1, degree):
            dp = -l * (x * eval_legendre(l, x) - eval_legendre(l - 1, x)) / sn
            z += (2 * l + 1) / (4 * np.pi) * dp ** 2
        return np.sqrt(1 + (np.pi ** 2 - 1) * z / s0)

    def zonal(profile):
        return np.median(fit_many(profile, B)['A_fit'].reshape(GLAT.shape), axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    axs[0].plot(lat, 1 / np.cos(phi), 'k-', label='Analytic $\\sec\\varphi$')
    axs[0].plot(lat, np.ones_like(lat), 'k:', label='Isotropic reference')
    for key, label, marker in [('grid', 'Planar grid', 'o'), ('cart3d', 'Cartesian', '^')]:
        axs[0].plot(lat, zonal(G['prof__ref ' + key]), marker, ms=3, label=label + f" ({float(G['r__ref ' + key]):g} km)")
    axs[0].set_yscale('log')
    axs[0].set_title('Planar and isotropic references')
    axs[0].set_ylabel('Fitted anisotropy A')
    axs[1].plot(lat, np.maximum(2 * np.cos(phi), 1 / (2 * np.cos(phi))), 'k-', label='Analytic 2:1 reference')
    axs[1].plot(lat, zonal(G['prof__ref SINR-input']), 'o', ms=3, label=f"Wrap input ({float(G['r__ref SINR-input']):g} km)")
    axs[1].set_title('Latitude-scaled wrap input')
    for degree, marker, style in [(10, 'o', '-'), (40, 's', '--')]:
        axs[2].plot(lat, sh_reference(degree), 'k' + style, label=f'Analytic $m=0$ ×π, L{degree}')
        axs[2].plot(lat, zonal(I[f'in__SatCLIP vit16l{degree}']), marker, ms=3, fillstyle='none', label=f'Inspected SH L{degree} (1 km)')
    axs[2].set_title('SatCLIP harmonic input')
    axs[2].set_ylim(0.9, 3.1)
    for ax in axs:
        ax.set_xlabel('Latitude (degrees)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.8)
    fig.tight_layout()
    fig.savefig(P / 'figures/fig_validation.png', dpi=210, bbox_inches='tight')
    plt.close(fig)

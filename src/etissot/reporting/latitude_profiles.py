"""Latitude profiles of output anisotropy against latitude-only input references.

For the twelve checkpoints whose input reference depends on latitude alone, the
area-weighted correlation r_A between log output and log reference anisotropy
splits exactly into two factors, because the reference is constant along each
parallel:

    r_A = r_zonal * sqrt(latitude share),

where r_zonal correlates the area-weighted row means of log output anisotropy
with the reference profile, and the latitude share is the fraction of the
variance of log output anisotropy explained by the row means (between-row over
total variance). GeoCLIP's Equal Earth reference also varies with longitude, so
the split is approximate there and its factors are reported for comparison only.

Uses the fixed grid at the primary radii, |latitude| <= 60 degrees, the cosine
weights and seam exclusions of diagnostics.geographic; r_A must reproduce the
fixed-grid values in geometry.json. Writes summaries/latitude_profiles.json and
figures/latitude_profiles.png.
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from etissot.geometry.fitting import fit_many
from etissot.geometry.references import equal_earth_metric
from etissot.statistics.spatial import LATS, LONS, wcorr

B = np.arange(32) * 2 * np.pi / 32
NAMES = ['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP r18l10', 'SatCLIP r50l10', 'SatCLIP vit16l40', 'SatCLIP r18l40',
         'SatCLIP r50l40', 'SINR cap1000', 'SINR cap10', 'SINR distilled', 'CSP iNat2018', 'CSP fMoW', 'Climplicit']
SHORT = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10',
         'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40',
         'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10',
         'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}


def reference(name, lat, lon, inputs):
    """Input-reference anisotropy as in diagnostics.geographic (fixed grid)."""
    if name == 'GeoCLIP':
        return equal_earth_metric(lat, lon)[0]
    if name.startswith('SatCLIP'):
        key = 'in__SatCLIP vit16l40' if name.endswith('l40') else 'in__SatCLIP vit16l10'
        return fit_many(inputs[key].reshape(-1, 32), B)['A_fit']
    sec = 1 / np.cos(np.radians(lat))
    return sec if name.startswith('CSP') else np.maximum(2 / sec, sec / 2)


def domain_mask(name, lat, lon):
    """Probe discs that cross the antimeridian are excluded for chart-dependent encoders."""
    if name == 'GeoCLIP' or name.startswith('CSP') or name == 'Climplicit':
        radius = 1.0 if name == 'GeoCLIP' else 0.1
        clearance = 6371 * np.arcsin(np.clip(abs(np.cos(np.radians(lat)) * np.sin(np.radians(180 - abs(lon)))), 0, 1))
        return ~((abs(lon) > 90) & (clearance <= radius))
    return np.ones(len(lat), bool)


def decompose(x, y, w, lat, rows):
    """r_A, the correlation of row means and the latitude share of the variance of x."""
    keep = w > 0
    r_a = float(wcorr(x, y, w))
    zx = np.array([np.average(x[(lat == la) & keep], weights=w[(lat == la) & keep]) for la in rows])
    zy = np.array([np.average(y[(lat == la) & keep], weights=w[(lat == la) & keep]) for la in rows])
    zw = np.array([w[(lat == la) & keep].sum() for la in rows])
    mean = np.average(x[keep], weights=w[keep])
    total = np.average((x[keep] - mean) ** 2, weights=w[keep])
    between = np.average((zx - mean) ** 2, weights=zw)
    return r_a, float(wcorr(zx, zy, zw)), float(between / total)


def run(workspace):
    lat, lon = np.meshgrid(LATS, LONS, indexing='ij')
    lat, lon = lat.ravel(), lon.ravel()
    geometry = json.loads((workspace.output / 'summaries/geometry.json').read_text())
    with np.load(workspace.data / 'profiles/global.npz') as g, np.load(workspace.data / 'profiles/input_references.npz') as inputs:
        fields = {n: (fit_many(g['prof__' + n].reshape(-1, 32), B)['A_fit'], reference(n, lat, lon, inputs)) for n in NAMES}
    rows60 = [la for la in LATS if abs(la) <= 60]
    rows85 = [la for la in LATS if abs(la) <= 85]
    summary = dict(sample='fixed grid, primary radii, |latitude| <= 60 degrees, cosine weights', checkpoints={})
    profiles = {}
    for n in NAMES:
        a, ref = fields[n]
        domain = domain_mask(n, lat, lon)
        w = np.cos(np.radians(lat)) * ((abs(lat) <= 60) & domain)
        r_a, r_zonal, share = decompose(np.log(a), np.log(ref), w, lat, rows60)
        if abs(r_a - geometry['fixed_grid'][n]['corr60']) > 1e-9:
            raise ValueError(f'Latitude-profile r_A differs from geometry.json for {n}')
        latitude_only = n != 'GeoCLIP'
        if latitude_only and abs(r_a - r_zonal * np.sqrt(share)) > 1e-6:
            raise ValueError(f'r_A does not factor into row-mean correlation and latitude share for {n}')
        summary['checkpoints'][n] = dict(r_A=r_a, r_zonal=r_zonal, latitude_share=share, latitude_only_reference=latitude_only)
        profiles[n] = [(la, np.percentile(a[(lat == la) & domain], [25, 50, 75]), np.percentile(ref[(lat == la) & domain], [0, 50, 100]))
                       for la in rows85]
    others = {n: v for n, v in summary['checkpoints'].items() if v['latitude_only_reference']}
    summary['latitude_only'] = dict(
        r_zonal_range=[min(v['r_zonal'] for v in others.values()), max(v['r_zonal'] for v in others.values())],
        latitude_share_range=[min(v['latitude_share'] for v in others.values()), max(v['latitude_share'] for v in others.values())])
    (workspace.output / 'summaries').mkdir(parents=True, exist_ok=True)
    (workspace.output / 'summaries/latitude_profiles.json').write_text(json.dumps(summary, indent=1) + '\n')
    plot(profiles, summary, workspace.output / 'figures/latitude_profiles.png')
    return summary


def plot(profiles, summary, path):
    with matplotlib.rc_context():
        plt.rcdefaults()
        plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                             'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6,
                             'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.dpi': 400})
        fig, axes = plt.subplots(4, 4, figsize=(5.65, 5.9), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, n in zip(axes, NAMES):
            y = np.array([r[0] for r in profiles[n]])
            out = np.array([r[1] for r in profiles[n]])
            ref = np.array([r[2] for r in profiles[n]])
            for side in (-1, 1):
                ax.axvspan(60 * side, 85 * side, color='0.93', lw=0, zorder=0)
            ax.fill_between(y, out[:, 0], out[:, 2], color='#1b6ca8', alpha=0.25, lw=0)
            ax.plot(y, out[:, 1], color='#1b6ca8', lw=0.9, label='Output (median, interquartile range)')
            if n == 'GeoCLIP':
                ax.fill_between(y, ref[:, 0], ref[:, 2], color='#d95f02', alpha=0.18, lw=0)
            ax.plot(y, ref[:, 1], color='#d95f02', lw=0.9, ls='--', label='Input reference')
            s = summary['checkpoints'][n]
            ax.text(0.5, 0.96, f"$r_A$ {s['r_A']:.2f}  profile {s['r_zonal']:.2f}\nshare {s['latitude_share']:.2f}",
                    transform=ax.transAxes, ha='center', va='top', fontsize=5.5, linespacing=1.1,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.6))
            ax.set_yscale('log')
            ax.set_ylim(1, 20)
            ax.set_xlim(-85, 85)
            ax.set_xticks([-60, 0, 60])
            ax.set_title(SHORT[n], fontsize=7.5, pad=2)
            ax.grid(alpha=0.25, lw=0.4)
        axes[0].set_yticks([1, 1.5, 2, 3, 5, 10])
        axes[0].set_yticklabels(['1', '1.5', '2', '3', '5', '10'])
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for ax in axes[len(NAMES):]:
            ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[-2].legend(handles, labels, loc='center left', frameon=False, bbox_to_anchor=(-0.05, 0.5))
        for i in (9, 10, 11, 12):
            axes[i].tick_params(labelbottom=True)
            axes[i].set_xlabel('Latitude ($^\\circ$)')
        for i in (0, 4, 8, 12):
            axes[i].set_ylabel('Anisotropy $A$')
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.07, top=0.97, wspace=0.08, hspace=0.32)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0.02, metadata={'Software': None})
        plt.close(fig)

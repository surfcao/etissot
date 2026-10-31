"""Radius-screening figure (supplementary Figure S1).

Reads the declared increment-curve measurements in
data/measurements/review/radius_screening.json (collected with
scripts/radius_screening.py from the pinned checkpoints), writes
outputs/summaries/radius_screening.json with derived ratios, and plots
outputs/figures/radius_screening.png.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

LABELS = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10',
          'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40',
          'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10',
          'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}
COLORS = ['#1b6ca8', '#d95f02', '#1b9e77', '#7570b3']


def site_label(lat, lon):
    ns = 'N' if lat > 0 else ('S' if lat < 0 else '')
    ew = 'E' if lon > 0 else 'W'
    return f'{abs(lat):g}°{ns}, {abs(lon):g}°{ew}'


def derived(d):
    """Ratios used in the manuscript text, per checkpoint and site."""
    R = np.array(d['radii_km'])
    out = {}
    for name, v in d['checkpoints'].items():
        i0 = int(np.argmin(abs(R - v['primary_radius_km'])))
        i10 = int(np.argmin(abs(R - 10.0)))  # nearest screened radius to 10 km
        # A second small-radius reference (the screened radius nearest 10 m; see
        # reference_radius_km) guards against rounding at the smallest radius.
        i_ref = int(np.argmin(abs(R - 0.01)))
        rows = []
        for s in v['sites']:
            rms = np.array(s['rms_stretch'])
            rows.append(dict(lat=s['lat'], lon=s['lon'],
                             primary_over_smallest=float(rms[i0] / rms[0]),
                             primary_over_reference=float(rms[i0] / rms[i_ref]),
                             at_comparison_radius_over_primary=float(rms[i10] / rms[i0]),
                             cached_profile_max_relative_difference=s.get('cached_profile_max_relative_difference')))
        drops = [1 - r['primary_over_smallest'] for r in rows]
        drops_reference = [1 - r['primary_over_reference'] for r in rows]
        diffs = [r['cached_profile_max_relative_difference'] for r in rows if r['cached_profile_max_relative_difference'] is not None]
        out[name] = dict(sites=rows, min_drop_at_primary=float(min(drops)), max_drop_at_primary=float(max(drops)),
                         min_drop_vs_reference=float(min(drops_reference)), max_drop_vs_reference=float(max(drops_reference)),
                         smallest_radius_km=float(R[0]), reference_radius_km=float(R[i_ref]),
                         comparison_radius_km=float(R[i10]),
                         max_cached_profile_difference=float(max(diffs)) if diffs else None)
    return out


def plot(d, path):
    R = np.array(d['radii_km'])
    with matplotlib.rc_context():
        plt.rcdefaults()
        fig, axes = plt.subplots(4, 4, figsize=(11, 10), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, name in zip(axes, LABELS):
            v = d['checkpoints'][name]
            r0 = v['primary_radius_km']; i0 = int(np.argmin(abs(R - r0)))
            ax.axhspan(0.95, 1.05, color='0.92', zorder=0)
            ax.axvline(r0, color='0.3', ls='--', lw=1, zorder=1)
            for c, s in zip(COLORS, v['sites']):
                rms = np.array(s['rms_stretch'])
                ax.plot(R, rms / rms[i0], color=c, lw=1.4, label=site_label(s['lat'], s['lon']))
            ax.set_title(f'{LABELS[name]} ({r0:g} km)', fontsize=10)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_ylim(0.2, 2.0); ax.set_xlim(R.min(), R.max())
            ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(['0.25', '0.5', '1', '2'])
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which='major', color='0.85', lw=0.5)
        for ax in axes[len(LABELS):]:
            ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[-1].legend(handles, labels, loc='center', title='Site', frameon=False, fontsize=9)
        for i in (9, 10, 11, 12):
            axes[i].tick_params(labelbottom=True)
            axes[i].set_xlabel('Probe radius (km)')
        for i in (0, 4, 8, 12):
            axes[i].set_ylabel('RMS stretch relative to\nprimary radius')
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def run(workspace):
    d = json.loads((workspace.data / 'measurements/review/radius_screening.json').read_text())
    if set(d['checkpoints']) != set(LABELS):
        raise ValueError('radius_screening.json must contain all thirteen checkpoints')
    summary = dict(d, derived=derived(d))
    (workspace.output / 'summaries').mkdir(parents=True, exist_ok=True)
    (workspace.output / 'summaries/radius_screening.json').write_text(json.dumps(summary, indent=1) + '\n')
    plot(d, workspace.output / 'figures/radius_screening.png')


if __name__ == '__main__':
    import argparse
    from etissot.workspace import Workspace
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    run(Workspace(a.data, a.output))

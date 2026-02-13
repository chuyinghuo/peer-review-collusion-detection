"""
Plot success heatmaps (Figure 6 / reference style): fraction of target papers with
a colluder assigned, and fraction of colluders with a colluder assigned (mean +/- sem).

Usage (from repo root):
  python scripts/plot_success_figure.py [--results-dir results] [--out figure6.png]
  Saves figure6a.png (frac_papers AAMAS), figure6b.png (frac_papers S2ORC),
  figure6c.png (frac_revs AAMAS), figure6d.png (frac_revs S2ORC).
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'libs'))
sys.path.insert(0, SCRIPT_DIR)

DATASET_LABELS = {'aamas_sub3': 'AAMAS', 'wu': 'S2ORC'}

# Metric -> title fragment (paper: "Fraction of target papers..." / "Fraction of colluders...")
SUCCESS_TITLES = {
    'frac_papers': 'Fraction of target papers with a colluder assigned',
    'frac_revs': 'Fraction of colluders with a colluder assigned',
}


def find_latest_success_csv(results_dir, dataset, bipartite=False):
    """Return path to most recent success_results CSV for dataset, or None."""
    suffix = '_bipartite' if bipartite else ''
    pattern = os.path.join(results_dir, f'success_results_{dataset}{suffix}_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_success_heatmap(results_dir, dataset, metric, bipartite=False, k_max=35):
    """
    Load success_results CSV and compute per (gamma, k) mean and sem for the given metric.
    Returns (pivot_mean, pivot_sem) or (None, None).
    """
    path = find_latest_success_csv(results_dir, dataset, bipartite=bipartite)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    for col in ['planted_k', 'planted_gamma', 'metric', 'trial', 'value']:
        if col not in df.columns:
            return None, None
    df = df[df['metric'] == metric].copy()
    if len(df) == 0:
        return None, None
    df = df[(df['planted_k'] >= 2) & (df['planted_k'] <= k_max)]
    df['planted_gamma'] = df['planted_gamma'].round(decimals=2)
    agg = df.groupby(['planted_gamma', 'planted_k'])['value'].agg(['mean', 'std', 'count'])
    agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
    pivot_mean = agg['mean'].unstack(level='planted_k')
    pivot_sem = agg['sem'].unstack(level='planted_k')
    return pivot_mean, pivot_sem


def plot_success_heatmap(ax, pivot_mean, pivot_sem, title, y_label='edge density (γ)', font_size=12, k_max=35):
    """Heatmap of success metric (mean +/- sem), style like paper Figure 6."""
    if pivot_mean is None or pivot_mean.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        return
    k_vals = sorted([c for c in pivot_mean.columns if 2 <= c <= k_max])
    if not k_vals:
        ax.set_title(title, fontsize=font_size + 2)
        return
    pivot_mean = pivot_mean.reindex(columns=k_vals)
    pivot_sem = pivot_sem.reindex(columns=k_vals) if pivot_sem is not None else None
    pivot_mean = pivot_mean.dropna(how='all').dropna(axis=1, how='all')
    if pivot_mean.size == 0 or len(pivot_mean) == 0 or len(pivot_mean.columns) == 0:
        ax.set_title(title, fontsize=font_size + 2)
        return
    if pivot_sem is not None:
        pivot_sem = pivot_sem.loc[pivot_mean.index, pivot_mean.columns]
    n_rows, n_cols = len(pivot_mean), len(pivot_mean.columns)
    aspect = 'equal' if 2 <= n_rows <= 8 else 'auto'
    im = ax.imshow(pivot_mean.values, aspect=aspect, cmap='YlOrRd', vmin=0, vmax=1,
                   extent=[pivot_mean.columns.min() - 0.5, pivot_mean.columns.max() + 0.5,
                           n_rows - 0.5, -0.5])
    # X-ticks: show every k; if too many, show every 2nd or 3rd for readability
    step = 1 if n_cols <= 15 else (2 if n_cols <= 25 else 3)
    tick_j = list(range(0, n_cols, step))
    ax.set_xticks([pivot_mean.columns[j] for j in tick_j])
    ax.set_xticklabels([int(pivot_mean.columns[j]) for j in tick_j], fontsize=font_size)
    ax.set_yticks(np.arange(n_rows))
    y_labels = [f'{float(pivot_mean.index[i]):.2g}' for i in range(n_rows)]
    ax.set_yticklabels(y_labels, fontsize=font_size)
    ax.set_xlabel('number of reviewers (k)', fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2)
    # Cell text: only annotate a subset of cells when many columns to avoid crowding
    text_step = 1 if n_cols <= 10 else (2 if n_cols <= 18 else 3)
    cell_font = max(6, min(9, font_size - 2 if n_cols <= 12 else font_size - 4))
    show_sem = n_cols <= 12  # omit +/- sem when many columns to shorten label
    for i in range(n_rows):
        for j in range(0, n_cols, text_step):
            k = pivot_mean.columns[j]
            m = pivot_mean.iloc[i, j]
            if pd.isna(m):
                continue
            s = pivot_sem.iloc[i, j] if pivot_sem is not None and not pd.isna(pivot_sem.iloc[i, j]) else 0
            if show_sem and not (pd.isna(s) or (isinstance(s, (int, float)) and s <= 0)):
                text = f'{m:.2f}+/-{s:.2f}'
            else:
                text = f'{m:.2f}'
            ax.text(float(k), i, text, ha='center', va='center', fontsize=cell_font)
    cbar = plt.colorbar(im, ax=ax, label='fraction')
    cbar.ax.tick_params(labelsize=font_size)


def main():
    parser = argparse.ArgumentParser(description='Plot success heatmaps (Fig 6 / reference style).')
    parser.add_argument('--results-dir', default='results', help='Directory with success_results_*.csv')
    parser.add_argument('--out', default='figure6.png', help='Base path (figure6a.png ... figure6d.png)')
    parser.add_argument('--bipartite', action='store_true', help='Use bipartite success results')
    parser.add_argument('--datasets', nargs='+', default=['aamas_sub3', 'wu'])
    parser.add_argument('-kx', '--k_max', type=int, default=35)
    parser.add_argument('--font-size', type=int, default=12)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    base, ext = os.path.splitext(args.out)
    y_label = 'bid density (η)' if args.bipartite else 'edge density (γ)'
    heatmap_figsize = (14, 8)  # wider so many k values don't crowd
    saved = []

    # (a) frac_papers AAMAS, (b) frac_papers S2ORC, (c) frac_revs AAMAS, (d) frac_revs S2ORC
    panels = [
        ('frac_papers', args.datasets[0]),
        ('frac_papers', args.datasets[1] if len(args.datasets) > 1 else args.datasets[0]),
        ('frac_revs', args.datasets[0]),
        ('frac_revs', args.datasets[1] if len(args.datasets) > 1 else args.datasets[0]),
    ]
    for idx, (metric, dataset) in enumerate(panels):
        label = DATASET_LABELS.get(dataset, dataset)
        title = f'{SUCCESS_TITLES[metric]}, {label}'
        mean_df, sem_df = load_success_heatmap(
            results_dir, dataset, metric, bipartite=args.bipartite, k_max=args.k_max
        )
        fig, ax = plt.subplots(1, 1, figsize=heatmap_figsize)
        plot_success_heatmap(
            ax, mean_df, sem_df, title,
            y_label=y_label, font_size=args.font_size, k_max=args.k_max
        )
        fig.subplots_adjust(left=0.14)
        path = f'{base}{chr(ord("a") + idx)}{ext}'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(path)

    print('Saved:', saved)


if __name__ == '__main__':
    main()

"""
Plot detection-probability heatmaps (Figure 6 style): at each (k, gamma) show the
fraction of trials where the detection algorithm found at least one colluder.

Usage (from repo root):
  python scripts/plot_detection_figure.py [--results-dir results] [--out figure6.png] [--method densest_subgraph]
  Saves figure6a.png (AAMAS), figure6b.png (S2ORC) by default.
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


def find_latest_detection_csv(results_dir, dataset, method, bipartite=False, graph_type=None):
    """Return path to most recent detection_results CSV for dataset and method, or None."""
    suffix = '_bipartite' if bipartite else ''
    g = f'_{graph_type}' if graph_type else ''
    pattern = os.path.join(results_dir, f'detection_results_{dataset}_{method}{g}{suffix}_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_detection_probability(results_dir, dataset, method, bipartite=False, graph_type=None, k_max=35):
    """
    Load detection results and compute per (k, gamma) detection probability and sem.
    Detection = at least one planted colluder found (num_true_positive >= 1).
    Returns (pivot_mean, pivot_sem) or (None, None).
    """
    path = find_latest_detection_csv(results_dir, dataset, method, bipartite=bipartite, graph_type=graph_type)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    for col in ['planted_k', 'planted_gamma', 'trial', 'num_true_positive']:
        if col not in df.columns:
            return None, None
    df = df[(df['planted_k'] >= 2) & (df['planted_k'] <= k_max)].copy()
    # Round gamma so float precision doesn't split groups (e.g. 0.9000001 vs 0.9)
    df['planted_gamma'] = df['planted_gamma'].round(decimals=2)
    df['detected'] = (df['num_true_positive'] >= 1).astype(float)
    agg = df.groupby(['planted_gamma', 'planted_k'])['detected'].agg(['mean', 'std', 'count'])
    agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
    pivot_mean = agg['mean'].unstack(level='planted_k')
    pivot_sem = agg['sem'].unstack(level='planted_k')
    return pivot_mean, pivot_sem


def plot_detection_heatmap(ax, pivot_mean, pivot_sem, title, y_label='edge density (γ)', font_size=12, k_max=35):
    """Heatmap of detection probability (mean ± sem) over k and gamma."""
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
    aspect = 'equal' if 2 <= n_rows <= 4 else 'auto'
    im = ax.imshow(pivot_mean.values, aspect=aspect, cmap='viridis', vmin=0, vmax=1,
                   extent=[pivot_mean.columns.min() - 0.5, pivot_mean.columns.max() + 0.5,
                           n_rows - 0.5, -0.5])
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
    text_step = 1 if n_cols <= 10 else (2 if n_cols <= 18 else 3)
    cell_font = max(6, min(9, font_size - 2 if n_cols <= 12 else font_size - 4))
    show_sem = n_cols <= 12
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
    cbar = plt.colorbar(im, ax=ax, label='P(detect \u2265 1 colluder)')
    cbar.ax.tick_params(labelsize=font_size)


def main():
    parser = argparse.ArgumentParser(description='Plot detection probability heatmaps (Fig 6 style).')
    parser.add_argument('--results-dir', default='results', help='Directory with detection_results_*.csv')
    parser.add_argument('--out', default='figure6.png', help='Base path for outputs (figure6a.png, figure6b.png)')
    parser.add_argument('--method', default='densest_subgraph', help='Detection method (e.g. densest_subgraph, fraudar)')
    parser.add_argument('--graph-type', default=None, help='Graph type for method (e.g. B for fraudar)')
    parser.add_argument('--bipartite', action='store_true', help='Use bipartite detection results')
    parser.add_argument('--datasets', nargs='+', default=['aamas_sub3', 'wu'])
    parser.add_argument('-kx', '--k_max', type=int, default=35)
    parser.add_argument('--font-size', type=int, default=12)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    base, ext = os.path.splitext(args.out)
    y_label = 'bid density (η)' if args.bipartite else 'edge density (γ)'
    heatmap_figsize = (12, 8)
    saved = []

    for idx, dataset in enumerate(args.datasets[:2]):
        label = DATASET_LABELS.get(dataset, dataset)
        mean_df, sem_df = load_detection_probability(
            results_dir, dataset, args.method,
            bipartite=args.bipartite, graph_type=args.graph_type, k_max=args.k_max
        )
        fig, ax = plt.subplots(1, 1, figsize=heatmap_figsize)
        plot_detection_heatmap(
            ax, mean_df, sem_df,
            f'Detection probability, {label}',
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

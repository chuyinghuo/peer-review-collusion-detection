"""
Generate Figure 1-style plots from clique and peeling results
(paper: On the Detection of Reviewer-Author Collusion Rings From Paper Bidding, arxiv.org/abs/2402.07860).

Usage (from repo root):
  python scripts/plot_figure1.py [--results-dir results] [--out figure1.png]
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

# Add repo root, libs, and scripts for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'libs'))
sys.path.insert(0, SCRIPT_DIR)

# Dataset display names (paper uses AAMAS / S2ORC)
DATASET_LABELS = {'aamas_sub3': 'AAMAS', 'wu': 'S2ORC'}


def find_latest_clique_csv(results_dir, dataset, bipartite=False):
    """Return path to most recent clique_results CSV for dataset, or None."""
    suffix = '_bipartite' if bipartite else ''
    pattern = os.path.join(results_dir, f'clique_results_{dataset}{suffix}_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_clique_counts(results_dir, dataset, bipartite=False):
    """Load clique counts as pivot table: index=gamma, columns=k, values=num_cliques.
    Returns (pivot_df, timeout_mask_df) for lower-bound cells."""
    path = find_latest_clique_csv(results_dir, dataset, bipartite)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    if 'gamma' not in df.columns or 'k' not in df.columns or 'num_cliques' not in df.columns:
        return None, None
    df = df[df['k'] >= 2].copy()
    if len(df) == 0:
        return None, None
    pivot = df.pivot_table(index='gamma', columns='k', values='num_cliques', aggfunc='first')
    timeout = df.pivot_table(index='gamma', columns='k', values='timeout_flag', aggfunc='first') if 'timeout_flag' in df.columns else None
    return pivot, timeout


def _format_count(v):
    if v >= 1e9:
        return f'{v/1e9:.1f}E9'
    if v >= 1e6:
        return f'{v/1e6:.1f}E6'
    if v >= 1e3:
        return f'{v/1e3:.1f}E3'
    return f'{int(v)}'


def plot_exact_counts_heatmap(ax, pivot, timeout_df, title, k_max=13, gamma_ticks=None, font_size=12):
    """Heatmap for exact counts (Fig 1a/1b style). Cells with timeout show '≥' prefix."""
    if pivot is None or pivot.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        return
    if gamma_ticks is None:
        gamma_ticks = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    k_vals = sorted([c for c in pivot.columns if 2 <= c <= k_max])
    if not k_vals:
        ax.set_title(title, fontsize=font_size + 2)
        return
    pivot = pivot.reindex(index=gamma_ticks, columns=k_vals)
    pivot = pivot.dropna(how='all').dropna(axis=1, how='all')
    if pivot.size == 0 or len(pivot) == 0 or len(pivot.columns) == 0:
        ax.set_title(title, fontsize=font_size + 2)
        return
    # Color by log10(count+1). Use aspect='equal' so cells are square and axis labels stay correct.
    vals = pivot.fillna(0).values.astype(float)
    vals_log = np.where(vals <= 0, 0, np.log10(np.maximum(vals, 1)))
    n_rows, n_cols = len(pivot), len(pivot.columns)
    im = ax.imshow(vals_log, aspect='equal', cmap='Blues', vmin=0,
                   extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                           n_rows - 0.5, -0.5])
    # X-axis: ticks at actual k values (same coords as image extent)
    ax.set_xticks(pivot.columns.values)
    ax.set_xticklabels([int(k) for k in pivot.columns], fontsize=font_size)
    # Y-axis: one tick per row, labels = edge density (γ) as 1.0, 0.9, ...
    ax.set_yticks(np.arange(n_rows))
    y_labels = [f'{float(pivot.index[i]):.1f}' for i in range(n_rows)]
    ax.set_yticklabels(y_labels, fontsize=font_size)
    ax.set_xlabel('number of reviewers (k)', fontsize=font_size)
    ax.set_ylabel('edge density (γ)', fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2)
    # Cell text in data coords (k, row_i) so first column doesn't overlap y-axis labels
    cell_font = max(7, min(10, font_size - 2))
    for i in range(n_rows):
        for j in range(n_cols):
            gamma = pivot.index[i]
            k = pivot.columns[j]
            v = pivot.iloc[i, j]
            if pd.isna(v):
                continue
            is_lb = False
            if timeout_df is not None and gamma in timeout_df.index and k in timeout_df.columns:
                try:
                    is_lb = bool(timeout_df.loc[gamma, k])
                except Exception:
                    pass
            pre = '≥' if is_lb else ''
            text = pre + _format_count(v)
            if v > 0 or is_lb:
                ax.text(float(k), i, text, ha='center', va='center', fontsize=cell_font)
    cbar = plt.colorbar(im, ax=ax, label='log10(count)')
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label('log10(count)', fontsize=font_size)


def get_peeling_frontier(dataset, bipartite=False):
    """Compute greedy peeling frontier (k, edge_density) for unipartite or bipartite."""
    try:
        import utils  # noqa: F401
        import oqc
    except ImportError as e:
        sys.stderr.write(f'Peeling import failed: {e}. Run: python scripts/detection_eval.py <dataset> -f\n')
        return None
    try:
        if bipartite:
            df = oqc.greedy_peeling_frontier_bipartite(dataset)
            # column is 'bid_density' for bipartite
            if 'bid_density' not in df.columns:
                return None
            df = df.rename(columns={'bid_density': 'edge_density'})
        else:
            df = oqc.greedy_peeling_frontier(dataset)
        return df[df['k'] >= 2].sort_values(by='k')
    except Exception:
        return None


def load_peeling_csv(results_dir, dataset, bipartite=False):
    """Load peeling_results_{dataset}.csv if present."""
    suffix = '_bipartite' if bipartite else ''
    path = os.path.join(results_dir, f'peeling_results_{dataset}{suffix}.csv')
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if 'bid_density' in df.columns and 'edge_density' not in df.columns:
        df = df.rename(columns={'bid_density': 'edge_density'})
    return df[df['k'] >= 2].sort_values(by='k') if 'k' in df.columns else None


def plot_peeling_frontier(ax, df, title, k_max=34, y_label='edge density (γ)', font_size=12):
    """Shaded region plot (Fig 1c/1d): for each k, max γ where an honest group exists."""
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        return
    col = 'edge_density' if 'edge_density' in df.columns else 'bid_density'
    if col not in df.columns:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        return
    df = df[df['k'] <= k_max].sort_values(by='k')
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
        ax.set_title(title, fontsize=font_size + 2)
        return
    k = df['k'].values
    gamma = np.asarray(df[col].values, dtype=float)
    ax.fill_between(k, 0.5, gamma, alpha=0.35, color='green')
    ax.plot(k, gamma, color='darkgreen', linewidth=2)
    ax.set_xlim(2, k_max)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('number of reviewers (k)', fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2)
    ax.tick_params(axis='both', labelsize=font_size)


def main():
    parser = argparse.ArgumentParser(description='Plot Figure 1-style from results')
    parser.add_argument('--results-dir', default='results', help='Directory containing clique_*.csv and peeling_*.csv')
    parser.add_argument('--out', default='figure1.png', help='Output base path (saves figure1a.png, figure1b.png, ...)')
    parser.add_argument('--datasets', nargs='+', default=['aamas_sub3', 'wu'], help='Datasets (e.g. aamas_sub3 wu)')
    parser.add_argument('--font-size', type=int, default=14, help='Base font size for labels and text')
    parser.add_argument('--unipartite-only', action='store_true', help='Only unipartite (no bipartite peeling)')
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    base, ext = os.path.splitext(args.out)
    font_size = args.font_size
    heatmap_figsize = (12, 8)   # wider for heatmap grid (k columns, gamma rows)
    line_figsize = (9, 7)
    saved = []

    dataset_a = args.datasets[0]
    dataset_b = args.datasets[1] if len(args.datasets) > 1 else dataset_a

    # (a) Exact counts, first dataset (AAMAS)
    fig, ax = plt.subplots(1, 1, figsize=heatmap_figsize)
    pivot_a, timeout_a = load_clique_counts(results_dir, dataset_a, bipartite=False)
    plot_exact_counts_heatmap(
        ax, pivot_a, timeout_a,
        f'Exact counts, {DATASET_LABELS.get(dataset_a, dataset_a)}',
        k_max=13, font_size=font_size
    )
    fig.subplots_adjust(left=0.14)  # room for y-axis labels so cell text doesn't overlap
    path_a = f'{base}a{ext}'
    plt.savefig(path_a, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(path_a)

    # (b) Exact counts, second dataset (S2ORC)
    fig, ax = plt.subplots(1, 1, figsize=heatmap_figsize)
    pivot_b, timeout_b = load_clique_counts(results_dir, dataset_b, bipartite=False)
    plot_exact_counts_heatmap(
        ax, pivot_b, timeout_b,
        f'Exact counts, {DATASET_LABELS.get(dataset_b, dataset_b)}',
        k_max=13, font_size=font_size
    )
    fig.subplots_adjust(left=0.14)  # room for y-axis labels so cell text doesn't overlap
    path_b = f'{base}b{ext}'
    plt.savefig(path_b, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(path_b)

    # (c) Greedy peeling frontier, first dataset
    fig, ax = plt.subplots(1, 1, figsize=line_figsize)
    peel_c = load_peeling_csv(results_dir, dataset_a, bipartite=False)
    if peel_c is None:
        peel_c = get_peeling_frontier(dataset_a, bipartite=False)
    plot_peeling_frontier(
        ax, peel_c,
        f'Groups found by greedy peeling, {DATASET_LABELS.get(dataset_a, dataset_a)}',
        k_max=34, font_size=font_size
    )
    plt.tight_layout()
    path_c = f'{base}c{ext}'
    plt.savefig(path_c, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(path_c)

    # (d) Greedy peeling frontier, second dataset
    fig, ax = plt.subplots(1, 1, figsize=line_figsize)
    peel_d = load_peeling_csv(results_dir, dataset_b, bipartite=False)
    if peel_d is None:
        peel_d = get_peeling_frontier(dataset_b, bipartite=False)
    plot_peeling_frontier(
        ax, peel_d,
        f'Groups found by greedy peeling, {DATASET_LABELS.get(dataset_b, dataset_b)}',
        k_max=34, font_size=font_size
    )
    plt.tight_layout()
    path_d = f'{base}d{ext}'
    plt.savefig(path_d, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(path_d)

    print('Saved:', ', '.join(saved))


if __name__ == '__main__':
    main()

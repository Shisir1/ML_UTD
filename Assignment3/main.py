"""
main.py

Runs K-means (Jaccard distance) clustering for several values of K on a
given tweet dataset, and prints/saves a results table matching the
format requested in the assignment:

    Value of K | SSE | Size of each cluster

Usage:
    python main.py <path_to_raw_tweet_file> [k1 k2 k3 ...]

If no k values are given, defaults to [2, 3, 5, 8, 10] (5 values, as
required: "at least 5 different values of K").
"""

import os
import sys
import datetime
from preprocess import load_and_preprocess
from kmeans_jaccard import KmeansJaccard


# Folder where result tables get appended across runs.
# Using a path relative to this script (portable across machines) rather
# than an absolute '/results' at the filesystem root, which would require
# root permissions and wouldn't work the same way on every OS/user account.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'kmeans_results.txt')


def format_results_table(results, filepath: str) -> str:
    """
    Build a clean, fixed-width, aligned table (as a string) from the
    (k, sse, cluster_sizes) tuples, with a header identifying which
    run/dataset/timestamp it came from.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    col_k, col_sse, col_sizes = 6, 10, 60
    lines = []
    lines.append('=' * (col_k + col_sse + col_sizes + 4))
    lines.append(f'Run: {timestamp}  |  Dataset: {filepath}')
    lines.append('-' * (col_k + col_sse + col_sizes + 4))
    lines.append(f'{"K":<{col_k}}{"SSE":<{col_sse}}{"Cluster sizes":<{col_sizes}}')
    lines.append('-' * (col_k + col_sse + col_sizes + 4))

    for k, sse, sizes in results:
        size_str = ', '.join(f'{i+1}: {s} tweets' for i, s in enumerate(sizes))
        lines.append(f'{k:<{col_k}}{sse:<{col_sse}.4f}{size_str}')

    lines.append('=' * (col_k + col_sse + col_sizes + 4))
    lines.append('')  # blank line separating runs
    return '\n'.join(lines)


def save_results_table(table_str: str):
    """Append the table to the results file, creating the folder/file if
    they don't exist yet. Never overwrites previous runs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(table_str + '\n')


def run(filepath: str, k_values, max_iters: int = 50, seed: int = 42):
    tweets = load_and_preprocess(filepath)
    print(f'Loaded and preprocessed {len(tweets)} tweets from {filepath}\n')

    results = []
    for k in k_values:
        model = KmeansJaccard(k=k, max_iters=max_iters, seed=seed)
        model.fit(tweets)
        results.append((k, model.sse_, model.cluster_sizes_))
        print(f'K={k:<3} done. SSE={model.sse_:.4f}')

    # --- print results table ---
    table_str = format_results_table(results, filepath)
    print('\n' + table_str)

    # --- append to results file (never overwrites previous runs) ---
    save_results_table(table_str)
    print(f'Results appended to: {RESULTS_FILE}')

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <path_to_raw_tweet_file> [k1 k2 k3 ...]')
        sys.exit(1)

    filepath = sys.argv[1]
    if len(sys.argv) > 2:
        k_values = [int(x) for x in sys.argv[2:]]
    else:
        k_values = [2, 3, 5, 8, 10]

    run(filepath, k_values)
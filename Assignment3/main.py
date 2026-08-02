import sys
from preprocess import load_and_preprocess
from kmeans_jaccard import KmeansJaccard

def run (filepath: str, k_values, max_iters: int = 50, seed: int = 42):
    tweets = load_and_preprocess(filepath)
    print(f'Loaded and preprocessed {len(tweets)} tweets from {filepath}\n')

    results = []
    for k in k_values: 
        model = KmeansJaccard(k=k, max_iters=max_iters, seed=seed)
        model.fit(tweets)
        results.append((k, model.sse_, model.cluster_sizes_))
        print(f'K={k}: SSE={model.sse_:.4f}, Cluster Sizes={model.cluster_sizes_}')

        #Print result table
        print('\nValue of K\tSSE\t\tSize of each cluster')
        for k, sse, sizes in results:
            size_str = ', '.join(f'{i+1}: {s} tweets' for i, s in enumerate(sizes))
            print(f'{k}\t\t{sse:.4f}\t{size_str}')

        return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <filepath> [k_values] [max_iters] [seed]")
        sys.exit(1)

    filepath = sys.argv[1]
    if len(sys.argv) > 2:
        k_values = [int(x) for x in sys.argv[2:]]
    else:
        k_values = [2,3,5,8,10]

    run(filepath, k_values)
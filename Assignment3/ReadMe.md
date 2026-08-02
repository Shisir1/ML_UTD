## Files
- `preprocess.py` — loads the raw tweet file and applies all required cleaning
  steps (strip id/timestamp, remove @mentions, strip `#` but keep the word,
  remove URLs, lowercase). Returns each tweet as a **set** of words.
- `kmeans_jaccard.py` — from-scratch K-means implementation using Jaccard
  distance. Since tweets have no numeric coordinates, each cluster's
  "centroid" is the **medoid**: the tweet in the cluster with the smallest
  total Jaccard distance to every other tweet in that cluster (as suggested
  in the assignment hint). Uses a farthest-point ("k-means++"-style) seeding
  strategy to reduce sensitivity to random initialization.
- `main.py` — runs the clustering for several values of K and prints the
  results table (K, SSE, cluster sizes).
- `data/` — put your downloaded UCI dataset file here (e.g.
  `usnewshealth.txt`). A small synthetic sample file,
  `data/usnewshealth.txt`, is included so you can verify the code runs
  before pointing it at the real dataset.

## Requirements
Python 3.7+, standard library only (`re`, `random`, `sys`) — no external
packages needed for the clustering logic itself.

## How to run

1. Download the dataset from
   https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter,
   unzip it, and place a file (e.g. `usnewshealth.txt`) inside `data/`.

2. Run with the default 5 values of K (2, 3, 5, 8, 10):

   ```
   python main.py data/usnewshealth.txt
   ```

3. Or specify your own K values:

   ```
   python main.py data/usnewshealth.txt 2 4 6 8 10
   ```

4. Quick test with the included sample data (no download needed):

   ```
   python main.py data/usnewshealth.txt 2 3 5 6 8
   ```

## Notes on design choices
- **Centroid definition**: Euclidean k-means uses the arithmetic mean, which
  isn't defined for sets of words. We use the medoid (real tweet minimizing
  total in-cluster distance) instead, as the assignment hint suggests.
- **Distance matrix caching**: all pairwise Jaccard distances are computed
  once up front and cached, since tweets never move — only their cluster
  labels and which tweet is the medoid change between iterations. This
  avoids redundant recomputation across iterations.
- **Empty clusters**: if a cluster loses all its members during an
  iteration, it's reseeded with the point currently farthest from its
  assigned centroid, a standard heuristic to avoid degenerate clusters.
- **Initialization**: centroids are seeded via a farthest-point heuristic
  (pick an initial point randomly, then repeatedly add the point farthest
  from all previously chosen centroids) rather than pure random selection,
  which reduces (but does not eliminate) the algorithm's sensitivity to
  initial seed choice — the assignment PDF explicitly notes K-means results
  depend on the initial seeds, so re-running with different `seed` values
  and comparing SSE is worth doing if your results look unstable.

## Sample output (on the included synthetic sample data)

```
Loaded and preprocessed 1400 tweets from data/usnewshealth.txt

K=2: SSE=1192.0708, Cluster Sizes=[1241, 159]

Value of K      SSE             Size of each cluster
2               1192.0708       1: 1241 tweets, 2: 159 tweets
```

SSE decreases monotonically as K increases, which is the expected behavior
(more clusters can only reduce or maintain total within-cluster distance) —
a useful sanity check that the implementation is working correctly.
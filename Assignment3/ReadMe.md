# Part II — Tweet Clustering with K-means (Jaccard Distance)

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
- `main.py` — runs the clustering for several values of K, prints a
  formatted results table (K, SSE, cluster sizes), and **appends** that
  table to `results/kmeans_results.txt` (never overwrites previous runs —
  see "Results output" below).
- `data/` — put your downloaded UCI dataset file here (e.g.
  `usnewshealth.txt`).
- `results/` — created automatically on first run. Contains
  `kmeans_results.txt`, a running log of every experiment you've run.

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

## Results output

Every time `main.py` is run, the results table is:
1. printed to the console, and
2. **appended** to `results/kmeans_results.txt` — the folder and file are
   created automatically if they don't already exist. Previous runs are
   never overwritten; each run is written as its own timestamped block, so
   the file accumulates a full history of every experiment.

Each block in the file looks like this:

```
================================================================================
Run: 2026-08-02 03:26:26  |  Dataset: data/sample_tweets.txt
--------------------------------------------------------------------------------
K     SSE       Cluster sizes
--------------------------------------------------------------------------------
2     10.7870   1: 12 tweets, 2: 4 tweets
3     9.0845    1: 10 tweets, 2: 4 tweets, 3: 2 tweets
5     6.1136    1: 6 tweets, 2: 3 tweets, 3: 2 tweets, 4: 2 tweets, 5: 3 tweets
================================================================================
```

Running the script again (even with different K values or a different
dataset) adds a new block below the previous ones rather than replacing the
file's contents — verified by running the script twice in a row and
confirming both runs' tables are present in the final file.

**Note on the results path**: the results folder is created at
`<script_directory>/results/`, i.e. relative to wherever `main.py` lives, not
at an absolute `/results` at the filesystem root. An absolute root-level path
would need elevated permissions and wouldn't be portable across machines/OSes
— if you specifically need the absolute path, change the `RESULTS_DIR`
constant near the top of `main.py`.

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
Value of K	SSE		Size of each cluster
2		10.7870	1: 12 tweets, 2: 4 tweets
3		9.0845	1: 10 tweets, 2: 4 tweets, 3: 2 tweets
5		6.1136	1: 6 tweets, 2: 3 tweets, 3: 2 tweets, 4: 2 tweets, 5: 3 tweets
6		4.6689	1: 4 tweets, 2: 3 tweets, 3: 2 tweets, 4: 2 tweets, 5: 3 tweets, 6: 2 tweets
8		1.9146	1: 2 tweets, 2: 2 tweets, 3: 2 tweets, 4: 2 tweets, 5: 2 tweets, 6: 2 tweets, 7: 2 tweets, 8: 2 tweets
```

SSE decreases monotonically as K increases, which is the expected behavior
(more clusters can only reduce or maintain total within-cluster distance) —
a useful sanity check that the implementation is working correctly.
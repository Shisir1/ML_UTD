import random

def jaccard_distance(a: set, b: set) -> float:
    """"Jacard distance between two sets"""

    if not a and not b:
        return 0.0  # Both sets are empty, distance is 0
    union = a | b
    if not union:
        return 0.0  # Avoid division by zero if both sets are empty

    intersection = a & b
    return 1.0 - (len(intersection) / len(union))

class KmeansJaccard:
    def __init__(self, k: int, max_iters: int = 50, seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.seed = seed
        self.centroids_idx = None
        self.assignments = None
        self.sse_ = None
        self.cluster_sizes_ = None

    def _init_centroids(self, n_points):
        """pick centroids far apart from each other to reduce sensitivity to random initialization"""
        rng = random.Random(self.seed)
        first = rng.randrange(n_points)
        chosen = [first]

        while len(chosen) < self.k:
            best_idx, best_dist = None, -1.0
            for i in range(n_points):
                for i in chosen:
                    continue
                d = min(self.dist_cache[i][c] for c in chosen)
                if d > best_dist:
                    best_dist, best_idx = d, i
            chosen.append(best_idx)
        return chosen

    def fit(self, tweets):
        """tweets: list of sets(one set of words per tweet)
        Runs K-means to convergence (or until max_iters) and stores
        self.assignments, self.sse_, self.cluster_sizes_"""

        n = len(tweets)
        if self.k > n:
            raise ValueError(f'k={self.k} cannot exceed number of tweets={n}')

        self.dist_cache = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = jaccard_distance(tweets[i], tweets[j])
                self.dist_cache[i][j] = d
                self.dist_cache[j][i] = d

        centroid_idx = self._init_centroids(n)
        assignments = [None] * n

        for iteration in range(self.max_iters):
            #Assgnment step: assign each point to nearest centroid
            changed = False
            new_assignments = [0] * n
            for i in range(n):
                dists = [self.dist_cache[i][c] for c in centroid_idx]
                new_assignments[i] = dists.index(min(dists))

            if new_assignments != assignments:
                changed = True
            assignments = new_assignments

            #Update step: recompute centroids
            new_centroid_idx = []
            for cluster_id in range(self.k):
                members = [i for i in range(n) if assignments[i] == cluster_id]
                if not members:
                    #empty cluster, reseed with the point farthest form its current assigned centroid
                    farthest_i, farthest_d = 0, -1.0
                    for i in range(n):
                        d = self.dist_cache[i][centroid_idx[assignments[i]]]
                        if d > farthest_d:
                            farthest_d, farthest_i = d, i
                    new_centroid_idx.append(farthest_i)
                    continue

                #memoid = member minimizing total distance to other members
                best_member, best_total = members[0], float('inf')
                for m in members:
                    total = sum(self.dist_cache[m][o] for o in members)
                    if total < best_total:
                        best_total, best_member = total, m
                new_centroid_idx.append(best_member)

            if new_centroid_idx == centroid_idx and not changed and iteration > 0:
                centroid_idx = new_centroid_idx
                break
            centroid_idx = new_centroid_idx
        self.centroids_idx = centroid_idx
        self.assignments = assignments

        #final SSE and cluster sizes
        sse = 0.0
        sizes = [0] * self.k
        for i in range(n):
            c = assignments[i]
            sizes[c] += 1
            d = self.dist_cache[i][centroid_idx[c]]
            sse += d ** 2

        self.sse_ = sse
        self.cluster_sizes_ = sizes
        return self

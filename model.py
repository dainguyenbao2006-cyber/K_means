import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = [[] for _ in range(self.k)]

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_indices]

        for _ in range(self.max_iters):

            self.clusters = self._create_clusters(self.centroids)
            
        
            centroids_old = self.centroids

            
            self.centroids = self._get_centroids(self.clusters)

            
            if self._is_converged(centroids_old, self.centroids):
                break

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
    
        distances = [np.sqrt(np.sum((sample - point)**2)) for point in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [np.sqrt(np.sum((centroids_old[i] - centroids_new[i])**2)) for i in range(self.k)]
        return sum(distances) == 0

    def predict(self, X):
    
        labels = []
        for sample in X:
            labels.append(self._closest_centroid(sample, self.centroids))
        return np.array(labels)
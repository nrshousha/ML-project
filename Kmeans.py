from data_loading import read_csv

# ============================
# K-Means Clustering Class (No Libraries)
# ============================
def preprocess_data(header, data):
    preprocessed_data = []
    for row in data:
        try:
            pclass = float(row[2])  # Pclass
            sex = 0.0 if row[4] == 'male' else 1.0  # Sex
            age = float(row[5]) if row[5] != '' else 30.0  # Age (استبدال بالقيمة الافتراضية)
            fare = float(row[9]) if row[9] != '' else 10.0  # Fare (استبدال بالقيمة الافتراضية)
            survived = float(row[1])  # Survived (target)

            cleaned_row = [pclass, sex, age, fare, survived]
            preprocessed_data.append(cleaned_row)
        except:
            continue
    return preprocessed_data

class KMeans:
    def __init__(self, n_clusters=2, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = []

    def initialize_centroids(self, X):
        # اختيار أول n_samples كمراكز أولية بدلاً من العشوائية
        self.centroids = X[:self.n_clusters]

    def assign_clusters(self, X):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, row in enumerate(X):
            distances = [self.euclidean_distance(row, centroid) for centroid in self.centroids]
            cluster_idx = self.argmin(distances)
            clusters[cluster_idx].append(idx)
        return clusters

    def update_centroids(self, X, clusters):
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                # إذا كانت المجموعة فارغة، لا تغيير على المركز
                new_centroids.append([0.0] * len(X[0])) 
                continue
            mean = []
            for i in range(len(X[0])):
                sum_feature = sum(float(X[idx][i]) for idx in cluster)
                mean.append(sum_feature / len(cluster))
            new_centroids.append(mean)
        return new_centroids

    def euclidean_distance(self, a, b):
        return sum((float(a[i]) - float(b[i])) ** 2 for i in range(len(a))) ** 0.5

    def argmin(self, lst):
        min_value = lst[0]
        min_index = 0
        for i in range(1, len(lst)):
            if lst[i] < min_value:
                min_value = lst[i]
                min_index = i
        return min_index

    def fit(self, X):
        self.initialize_centroids(X)
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, clusters)
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
        self.clusters = clusters

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        preds = []
        for row in X:
            distances = [self.euclidean_distance(row, centroid) for centroid in self.centroids]
            preds.append(self.argmin(distances))
        return preds

    def calculate_sse(self, X):
        sse = 0.0
        for i, cluster in enumerate(self.clusters):
            for idx in cluster:
                sse += self.euclidean_distance(X[idx], self.centroids[i]) ** 2
        return sse

    def get_clusters(self):
        return self.clusters

    def get_centroids(self):
        return self.centroids


# --- Testing K-Means ---
if __name__ == "__main__":
    header, data = read_csv("train.csv")
    preprocessed_data = preprocess_data(header, data)

    print("\n=== K-Means Clustering ===")
    kmeans = KMeans(n_clusters=2, max_iters=100)
    kmeans.fit(preprocessed_data)

    clusters = kmeans.get_clusters()
    centroids = kmeans.get_centroids()

    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: {len(cluster)} passengers")

    for idx, centroid in enumerate(centroids):
        print(f"Cluster {idx} centroid: {['{:.2f}'.format(c) for c in centroid]}")

    # حساب الـ SSE لتقييم الأداء
    sse = kmeans.calculate_sse(preprocessed_data)
    print(f"Sum of Squared Errors (SSE): {sse:.2f}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import base64

K_RANGE = range(1, 21)  # v2
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")


def load_data():
    """Loads the CSV and returns it as a base64 string."""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """Drops nulls, scales the features and saves the scaler."""
    df = pickle.loads(base64.b64decode(data_b64))

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # v2: save scaler for test data
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(min_max_scaler, f)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """Finds the optimal k with the elbow method and saves that model."""
    df = pickle.loads(base64.b64decode(data_b64))

    kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # v2: train final model with optimal k
    kl = KneeLocator(list(K_RANGE), sse, curve="convex", direction="decreasing")
    optimal_k = int(kl.elbow)
    print(f"Optimal no. of clusters: {optimal_k}")

    final_model = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
    final_model.fit(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(final_model, f)

    return {"sse": sse, "optimal_k": optimal_k}


def evaluate_model(data_b64: str, filename: str):
    """v2: prints silhouette score and cluster sizes."""
    df = pickle.loads(base64.b64decode(data_b64))

    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        model = pickle.load(f)

    labels = model.predict(df)
    score = silhouette_score(df, labels, sample_size=2000, random_state=42)
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    print(f"Silhouette score: {score:.3f}")
    print(f"Customers per cluster: {sizes}")
    return {"silhouette_score": round(float(score), 3),
            "cluster_sizes": {int(k): int(v) for k, v in sizes.items()}}


def load_model_elbow(filename: str, results: dict):
    """Loads the model and predicts clusters for test.csv."""
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        loaded_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print(f"Model was trained with k = {results['optimal_k']}")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    df_scaled = scaler.transform(df)  # v2: scale test data
    preds = loaded_model.predict(df_scaled)

    print(f"Predicted clusters: {preds.tolist()}")
    return [int(p) for p in preds]

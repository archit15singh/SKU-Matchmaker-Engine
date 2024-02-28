import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


@timeit
def clean_columns(df, columns):
    special_chars_pattern = r'[!@#$%^&*(),.?":{}|<>]'
    for column in columns:
        df[column] = df[column].str.replace("null", "", case=False, regex=False)
        df[column] = df[column].str.replace(special_chars_pattern, "", regex=True)
        df[column] = df[column].str.strip()
        df[column] = df[column].str.lower()

    return df


@timeit
def sample_balanced_rows(df, column_name, n_samples_per_group=None, random_state=42):
    if n_samples_per_group is None:
        n_samples_per_group = df[column_name].value_counts().min()
    return df.groupby(column_name).sample(
        n=n_samples_per_group, random_state=random_state
    )


@timeit
def read_data(zip_path):
    return pd.read_json(zip_path, compression="gzip", lines=True)


@timeit
def generate_embeddings(product_titles, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(product_titles, batch_size=36)
    return embeddings


@timeit
def cluster_embeddings(
    embeddings, min_cluster_size, min_samples=None, cluster_selection_epsilon=None
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else None,
        cluster_selection_epsilon=(
            cluster_selection_epsilon if cluster_selection_epsilon is not None else 0
        ),
        gen_min_span_tree=True,
        core_dist_n_jobs=36,
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels


@timeit
def tune_hdbscan_parameters(embeddings):
    min_cluster_sizes = list(range(5, 50, 5))
    min_samples_list = [None] + list(range(5, 50, 5))
    cluster_selection_epsilons = [0.0, 0.1, 0.5, 1.0]

    best_score = -1
    best_params = {}

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            for cluster_selection_epsilon in cluster_selection_epsilons:
                cluster_labels = cluster_embeddings(
                    embeddings,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                )
                if len(set(cluster_labels)) > 1 and not (
                    list(cluster_labels).count(-1) == len(cluster_labels)
                ):
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "cluster_selection_epsilon": cluster_selection_epsilon,
                        }
                print(
                    f"Params: {min_cluster_size}, {min_samples}, {cluster_selection_epsilon} - Score: {score}"
                )

    return best_params, best_score


@timeit
def calculate_silhouette_score(embeddings, cluster_labels):
    filtered_embeddings = embeddings[cluster_labels != -1]
    filtered_labels = cluster_labels[cluster_labels != -1]

    if len(set(filtered_labels)) > 1:
        score = silhouette_score(filtered_embeddings, filtered_labels)
        return score
    else:
        return None


@timeit
def main():
    zip_path = "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/all_gs.json.gz"
    df = read_data(zip_path)
    print(f"Original DataFrame shape: {df.shape}")

    columns_to_clean = ["title_left", "title_right"]
    df = clean_columns(df, columns_to_clean)
    print(f"Cleaned DataFrame shape: {df.shape}")

    df = sample_balanced_rows(df, "label")
    print(f"Sampled balanced DataFrame shape: {df.shape}")
    df.to_csv(
        "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/all_gs_cleaned.csv",
        index=False,
    )

    all_titles = pd.concat([df["title_left"], df["title_right"]])
    unique_titles = all_titles.unique()
    print(f"Total unique product titles: {len(unique_titles)}")

    embeddings = generate_embeddings(unique_titles, "all-mpnet-base-v2")

    # Tune HDBSCAN parameters
    best_params, best_score = tune_hdbscan_parameters(embeddings)
    print(f"Best Parameters: {best_params}, Best Silhouette Score: {best_score}")

    # Cluster with the best parameters
    cluster_labels = cluster_embeddings(
        embeddings,
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params.get("min_samples"),
        cluster_selection_epsilon=best_params.get("cluster_selection_epsilon"),
    )

    silhouette_score = calculate_silhouette_score(embeddings, cluster_labels)
    if silhouette_score is not None:
        print(f"Silhouette Score: {silhouette_score}")
    else:
        print("Silhouette Score could not be calculated due to insufficient clusters.")

    clusters_df = pd.DataFrame(
        {"title": unique_titles, "cluster_label": cluster_labels}
    )

    cluster_counts = clusters_df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_label", "count"]
    cluster_counts = cluster_counts.sort_values(by="cluster_label")

    for label in cluster_counts["cluster_label"]:
        print(f"\nCluster {label}:")
        titles_in_cluster = clusters_df[clusters_df["cluster_label"] == label][
            "title"
        ].tolist()
        print(f"Number of items: {len(titles_in_cluster)}")
        print("Items:", titles_in_cluster[:5])

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x="cluster_label", y="count", data=cluster_counts, palette="viridis"
    )
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Items")
    plt.title("Number of Items in Each Cluster")
    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.show(block=False)

    print()


if __name__ == "__main__":
    main()

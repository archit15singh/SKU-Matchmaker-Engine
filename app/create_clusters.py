import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns


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
    embeddings = model.encode(product_titles, batch_size=64)
    return embeddings


@timeit
def cluster_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels


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

    all_titles = pd.concat([df["title_left"], df["title_right"]])
    unique_titles = all_titles.unique()
    print(f"Total unique product titles: {len(unique_titles)}")

    embeddings = generate_embeddings(unique_titles, "all-mpnet-base-v1")
    cluster_labels = cluster_embeddings(embeddings)

    # Creating a DataFrame for cluster labels and product titles
    clusters_df = pd.DataFrame(
        {"title": unique_titles, "cluster_label": cluster_labels}
    )

    # Getting the count of items in each cluster
    cluster_counts = clusters_df["cluster_label"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_label", "count"]
    cluster_counts = cluster_counts.sort_values(by="cluster_label")

    # Plotting the clusters with their sizes
    plt.figure(figsize=(12, 8))
    sns.barplot(x="cluster_label", y="count", data=cluster_counts, palette="viridis")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Items")
    plt.title("Number of Items in Each Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print out the cluster details for review
    for label in cluster_counts["cluster_label"]:
        if label != -1:  # We skip the noise points
            print(f"\nCluster {label}:")
            titles_in_cluster = clusters_df[clusters_df["cluster_label"] == label][
                "title"
            ].tolist()
            print(f"Number of items: {len(titles_in_cluster)}")
            print("Items:", titles_in_cluster)


if __name__ == "__main__":
    main()
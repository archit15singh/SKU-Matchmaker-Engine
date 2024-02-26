from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd


def timeit(func):
    """
    Decorator to time functions
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


@timeit
def load_data(file_path):
    """
    Load data from a JSON file into a pandas DataFrame
    """
    return pd.read_json(file_path, lines=True)


@timeit
def sample_balanced_rows(df, column_name, n_samples_per_group=None):
    if n_samples_per_group is None:
        n_samples_per_group = df[column_name].value_counts().min()
    return df.groupby(column_name).sample(n=n_samples_per_group)


@timeit
def preprocess_data(df):
    """
    Preprocess data by selecting specific columns and dropping nulls
    """
    df = df[
        ["title_left", "title_right", "label", "cluster_id_left", "cluster_id_right"]
    ]
    initial_row_count = len(df)
    df = df.dropna(subset=["title_left", "title_right"])
    final_row_count = len(df)

    print(f"\nRow count before dropping nulls: {initial_row_count}")
    print(f"Row count after dropping nulls: {final_row_count}")

    return df


@timeit
def print_label_percentages(df):
    """
    Print percentages of values in 'label'
    """
    label_percentages = df["label"].value_counts(normalize=True) * 100
    print("\nPercentages of values in 'label':")
    print(label_percentages)


@timeit
def calculate_summary_statistics(df):
    """
    Calculate and print summary statistics for title lengths
    """
    df["title_left_length"] = df["title_left"].apply(len)
    df["title_right_length"] = df["title_right"].apply(len)
    print("\nSummary statistics on lengths of 'title_left' and 'title_right':")
    summary_stats = df[["title_left_length", "title_right_length"]].describe()
    print(summary_stats)
    return summary_stats


@timeit
def detect_outliers(df, summary_stats):
    """
    Detect and print outliers based on IQR for title lengths
    """
    Q1_left, Q3_left = (
        summary_stats.loc["25%", "title_left_length"],
        summary_stats.loc["75%", "title_left_length"],
    )
    IQR_left = Q3_left - Q1_left
    lower_bound_left = Q1_left - 1.5 * IQR_left
    upper_bound_left = Q3_left + 1.5 * IQR_left

    Q1_right, Q3_right = (
        summary_stats.loc["25%", "title_right_length"],
        summary_stats.loc["75%", "title_right_length"],
    )
    IQR_right = Q3_right - Q1_right
    lower_bound_right = Q1_right - 1.5 * IQR_right
    upper_bound_right = Q3_right + 1.5 * IQR_right

    outliers_left = df[
        (df["title_left_length"] < lower_bound_left)
        | (df["title_left_length"] > upper_bound_left)
    ]
    outliers_right = df[
        (df["title_right_length"] < lower_bound_right)
        | (df["title_right_length"] > upper_bound_right)
    ]

    print(
        f"\nOutliers in 'title_left_length':\n{outliers_left[['title_left', 'title_left_length']]}"
    )
    print(
        f"\nOutliers in 'title_right_length':\n{outliers_right[['title_right', 'title_right_length']]}"
    )


@timeit
def print_null_in_titles(df):
    """
    Print rows where 'title_left' or 'title_right' contain the word "null"
    """
    null_in_titles = df[
        df["title_left"].str.contains("null", case=False)
        | df["title_right"].str.contains("null", case=False)
    ]
    print(f"\nRows with 'null' in 'title_left' or 'title_right': {len(null_in_titles)}")


@timeit
def cleaning_insights(column):
    """
    Generate and print cleaning insights for a given DataFrame column
    """
    insights = {
        "unnecessary_whitespace": (
            column.str.startswith(" ") | column.str.endswith(" ")
        ).mean()
        * 100,
        "commas": column.str.contains(",", regex=False).mean() * 100,
        "contains_null_word": column.str.contains("null", case=False).mean() * 100,
        "only_whitespace": column.str.isspace().mean() * 100,
        "numeric_characters": column.str.contains("[0-9]", regex=True).mean() * 100,
        "special_characters": column.str.contains(
            '[!@#$%^&*(),.?":{}|<>]', regex=True
        ).mean()
        * 100,
        "uppercase": column.str.isupper().mean() * 100,
        "specific_keyword": column.str.contains("example_keyword", case=False).mean()
        * 100,
    }

    print(f"\nCleaning insights for '{column.name}':")
    for insight, value in insights.items():
        print(f"Percentage of rows with {insight.replace('_', ' ')}: {value:.2f}%")


@timeit
def clean_columns(df, columns):
    special_chars_pattern = r'[!@#$%^&*(),.?":{}|<>]'
    for column in columns:
        df[column] = df[column].str.replace("null", "", case=False, regex=False)
        df[column] = df[column].str.replace(special_chars_pattern, "", regex=True)
        df[column] = df[column].str.strip()
    return df


@timeit
def encode_titles_to_vectors(
    df,
    title_left_col="title_left",
    title_right_col="title_right",
    batch_size=32,
    model_name="all-MiniLM-L6-v2",
):
    model = SentenceTransformer(model_name)

    all_titles = pd.concat([df[title_left_col], df[title_right_col]]).unique()

    embeddings = model.encode(
        all_titles,
        batch_size=batch_size,
        show_progress_bar=True,
        output_value="sentence_embedding",
    )

    title_to_embedding = {
        title: embedding for title, embedding in zip(all_titles, embeddings)
    }

    df["title_left_vector"] = df[title_left_col].map(title_to_embedding)
    df["title_right_vector"] = df[title_right_col].map(title_to_embedding)

    return df


@timeit
def main():
    file_path = "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/computers_train/computers_train_xlarge.json"
    df = load_data(file_path)
    df = preprocess_data(df)
    df = clean_columns(df, columns=["title_left", "title_right"])
    print_label_percentages(df)
    summary_stats = calculate_summary_statistics(df)
    detect_outliers(df, summary_stats)
    print_null_in_titles(df)
    print_null_in_titles(df)
    cleaning_insights(df["title_left"])
    cleaning_insights(df["title_right"])
    file_path = "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/computers_train/computers_train_xlarge_cleaned.csv"
    # df.to_csv(file_path, index=False)
    df = sample_balanced_rows(df, "label")
    df = encode_titles_to_vectors(df)
    df["cosine_similarity"] = df.apply(
        lambda row: util.cos_sim(row["title_left_vector"], row["title_right_vector"])[
            0
        ][0].item(),
        axis=1,
    )
    filtered_df = df[
        (df["label"] == 0)
        & (df["cosine_similarity"] >= 0.0)
        & (df["cosine_similarity"] <= 0.1)
    ]

    # Counting the number of rows that match the conditions
    num_rows = filtered_df.shape[0]
    print()


if __name__ == "__main__":
    main()

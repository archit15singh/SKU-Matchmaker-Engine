import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


@timeit
def load_data(file_path):
    return pd.read_json(file_path, lines=True)


@timeit
def basic_overview(df):
    print(f"Shape of the Dataset: {df.shape}")
    print("Data Types:")
    print(df.dtypes)
    missing_values = df.isnull().sum()
    print("Missing Values:")
    print(missing_values)
    missing_values.plot(kind="bar", title="Missing Values in Each Column")
    plt.show()  # Show the plot for missing values
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        print(f"Unique values in {column}: {df[column].nunique()}")
    print("Summary Statistics for Numerical Fields:")
    print(df[["price_left", "price_right"]].describe())


@timeit
def data_quality_checks(df):
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    print("Consistency in `category_left` vs. `category_right`:")
    print((df["category_left"] == df["category_right"]).value_counts())
    print("Price Anomalies:")
    print(df[(df["price_left"] < 0) | (df["price_right"] < 0)])
    df["identifiers_left_count"] = df["identifiers_left"].str.count(";") + 1
    df["identifiers_right_count"] = df["identifiers_right"].str.count(";") + 1
    print("Identifiers Consistency Check:")
    print(df[["identifiers_left_count", "identifiers_right_count"]].describe())


@timeit
def text_data_quality(df):
    print("Empty Descriptions Left:")
    print(df[df["description_left"].str.strip() == ""].shape[0])
    print("Empty Descriptions Right:")
    print(df[df["description_right"].str.strip() == ""].shape[0])


@timeit
def univariate_analysis(df):
    df["price_left"].hist(bins=50, alpha=0.5, label="Price Left")
    df["price_right"].hist(bins=50, alpha=0.5, label="Price Right")
    plt.legend()
    plt.title("Price Distribution")
    plt.show()
    df["category_left"].value_counts().plot(
        kind="bar", title="Frequency of Categories in Left Products"
    )
    plt.show()


@timeit
def bivariate_analysis(df):
    sns.scatterplot(data=df, x="price_left", y="price_right")
    plt.title("Price Left vs. Price Right")
    plt.show()


@timeit
def text_analysis(df):
    text = " ".join(df["description_left"].dropna())
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


@timeit
def categorical_data_analysis(df):
    pd.crosstab(df["brand_left"], df["category_left"]).plot(kind="bar", figsize=(10, 8))
    plt.show()


def main():
    file_path = "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/computers_train/computers_train_xlarge.json"
    df = load_data(file_path)

    basic_overview(df)
    data_quality_checks(df)
    text_data_quality(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    text_analysis(df)
    categorical_data_analysis(df)


if __name__ == "__main__":
    main()

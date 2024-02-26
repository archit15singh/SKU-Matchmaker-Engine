import json
import time
import pandas as pd


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


def main():
    file_path = "/Users/architsingh/Documents/projects/SKU-Matchmaker-Engine/data/computers_train/computers_train_xlarge.json"
    df = load_data(file_path)

    df = df[["description_right", "description_left"]]


if __name__ == "__main__":
    main()

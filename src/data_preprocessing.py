import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df = df[["Cuisines", "Average Cost for two", "Aggregate rating", "Votes"]]

    df = df.dropna(subset=["Cuisines"]).copy()

    df["Average Cost for two"] = df["Average Cost for two"].fillna(
        df["Average Cost for two"].median()
    )
    df["Aggregate rating"] = df["Aggregate rating"].fillna(
        df["Aggregate rating"].median()
    )
    df["Votes"] = df["Votes"].fillna(0)

    df["Primary Cuisine"] = df["Cuisines"].apply(
        lambda x: x.split(",")[0].strip()
    )
    
    cuisine_counts = df["Primary Cuisine"].value_counts()
    df = df[df["Primary Cuisine"].isin(cuisine_counts[cuisine_counts >= 5].index)]

    return df

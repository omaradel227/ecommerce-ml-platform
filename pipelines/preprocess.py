import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.preprocessing import LabelEncoder

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

RATINGS_FILE  = os.path.join(RAW_DIR, "ratings_Beauty.csv")
PRODUCTS_FILE = os.path.join(RAW_DIR, "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv")

def preprocess_ratings():
    df = pd.read_csv(RATINGS_FILE)
    print(f"  Raw shape: {df.shape}")

    df = df.drop_duplicates()
    df = df.dropna()
    user_counts   = df["UserId"].value_counts()
    active_users  = user_counts[user_counts >= 5].index
    df            = df[df["UserId"].isin(active_users)]
    product_counts   = df["ProductId"].value_counts()
    active_products  = product_counts[product_counts >= 5].index
    df               = df[df["ProductId"].isin(active_products)]

    print(f"  After filtering: {df.shape}")
    print(f"  Unique users   : {df['UserId'].nunique()}")
    print(f"  Unique products: {df['ProductId'].nunique()}")

    df.to_csv(os.path.join(PROCESSED_DIR, "ratings_clean.csv"), index=False)
    print("  Saved ratings_clean.csv")
    return df

def preprocess_purchase_predictor(ratings_df):

    user_features = ratings_df.groupby("UserId").agg(
        user_avg_rating    = ("Rating", "mean"),
        user_rating_count  = ("Rating", "count"),
        user_rating_std    = ("Rating", "std"),
        user_max_rating    = ("Rating", "max"),
        user_min_rating    = ("Rating", "min"),
    ).reset_index()

    product_features = ratings_df.groupby("ProductId").agg(
        product_avg_rating    = ("Rating", "mean"),
        product_rating_count  = ("Rating", "count"),
        product_rating_std    = ("Rating", "std"),
    ).reset_index()

    df = ratings_df.merge(user_features,    on="UserId")
    df = df.merge(product_features, on="ProductId")

    df["target"] = (df["Rating"] >= 4).astype(int)

    le_user    = LabelEncoder()
    le_product = LabelEncoder()
    df["user_encoded"]    = le_user.fit_transform(df["UserId"])
    df["product_encoded"] = le_product.fit_transform(df["ProductId"])

    df["user_rating_std"]    = df["user_rating_std"].fillna(0)
    df["product_rating_std"] = df["product_rating_std"].fillna(0)

    feature_cols = [
        "user_encoded",
        "product_encoded",
        "user_avg_rating",
        "user_rating_count",
        "user_rating_std",
        "user_max_rating",
        "user_min_rating",
        "product_avg_rating",
        "product_rating_count",
        "product_rating_std",
    ]

    df[feature_cols + ["target"]].to_csv(
        os.path.join(PROCESSED_DIR, "purchase_features.csv"), index=False
    )
    print(f"  Shape          : {df.shape}")
    print(f"  Target balance : {df['target'].value_counts(normalize=True).to_dict()}")
    print("  Saved purchase_features.csv")
    return df

def preprocess_products():
    print("\nProcessing products")
    df = pd.read_csv(PRODUCTS_FILE)
    print(f"  Raw shape: {df.shape}")

    df = df[["Product Name", "Category", "Image", "Selling Price", "About Product"]].copy()

    df = df.dropna(subset=["Category", "Image"])

    df["main_category"] = df["Category"].apply(
        lambda x: x.split("|")[0].strip() if isinstance(x, str) else "Unknown"
    )

    cat_counts     = df["main_category"].value_counts()
    valid_cats     = cat_counts[cat_counts >= 10].index
    df             = df[df["main_category"].isin(valid_cats)]

    df["image_url"] = df["Image"].apply(
        lambda x: x.split("|")[0].strip() if isinstance(x, str) else ""
    )

    df["price"] = df["Selling Price"].str.replace("$", "").str.replace(",", "")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

    print(f"  After filtering : {df.shape}")
    print(f"  Categories      : {df['main_category'].nunique()}")
    print(f"  Top categories  :\n{df['main_category'].value_counts().head()}")

    df.to_csv(os.path.join(PROCESSED_DIR, "products_clean.csv"), index=False)
    print("  Saved products_clean.csv")
    return df

def main():
    mlflow.set_experiment("preprocessing")

    with mlflow.start_run(run_name="preprocess_all"):
        ratings_df  = preprocess_ratings()
        purchase_df = preprocess_purchase_predictor(ratings_df)
        products_df = preprocess_products()

        mlflow.log_metric("ratings_rows",         len(ratings_df))
        mlflow.log_metric("ratings_unique_users",  ratings_df["UserId"].nunique())
        mlflow.log_metric("ratings_unique_products", ratings_df["ProductId"].nunique())
        mlflow.log_metric("purchase_rows",         len(purchase_df))
        mlflow.log_metric("products_rows",         len(products_df))
        mlflow.log_metric("product_categories",    products_df["main_category"].nunique())

        print("\nPreprocessing complete.")
        print(f"  Ratings    : {len(ratings_df):,} rows")
        print(f"  Purchase   : {len(purchase_df):,} rows")
        print(f"  Products   : {len(products_df):,} rows")

if __name__ == "__main__":
    main()
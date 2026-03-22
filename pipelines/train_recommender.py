import pandas as pd
import numpy as np
import mlflow
import os
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import onnxruntime as rt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models/recommender"
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIGS = [
    {"n_factors": 50,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"n_factors": 100, "n_epochs": 30, "lr_all": 0.01,  "reg_all": 0.02},
]

def load_data():
    print("Loading ratings...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings_clean.csv"))

    df = df.sample(n=100000, random_state=42)
    print(f"  Using {len(df):,} ratings")
    print(f"  Users   : {df['UserId'].nunique():,}")
    print(f"  Products: {df['ProductId'].nunique():,}")
    print(f"  Rating range: {df['Rating'].min()} - {df['Rating'].max()}")

    reader  = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[["UserId", "ProductId", "Rating"]], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    return trainset, testset, df

def plot_rating_distribution(df, run_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Rating Distribution - {run_name}")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"rating_dist_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_predictions_vs_actual(predictions, run_name):
    actual    = [pred.r_ui for pred in predictions]
    predicted = [pred.est for pred in predictions]
    fig, ax   = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.1, s=5, color="steelblue")
    ax.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Predicted vs Actual - {run_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"pred_vs_actual_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def get_top_n_recommendations(model, user_id, df, n=10):
    all_products    = df["ProductId"].unique()
    rated_products  = df[df["UserId"] == user_id]["ProductId"].tolist()
    unrated         = [p for p in all_products if p not in rated_products]

    predictions = [model.predict(user_id, product) for product in unrated[:500]]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [(pred.iid, round(pred.est, 2)) for pred in predictions[:n]]

def save_model(model, run_name):
    path = os.path.join(MODEL_DIR, f"svd_{run_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path

def train_and_log(config, trainset, testset, df, run_index):
    run_name = f"svd_run_{run_index:02d}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_param("train_size", trainset.n_ratings)
        mlflow.log_param("test_size",  len(testset))
        mlflow.log_param("n_users",    trainset.n_users)
        mlflow.log_param("n_items",    trainset.n_items)

        model = SVD(**config, random_state=42)
        model.fit(trainset)

        predictions = model.test(testset)
        rmse        = accuracy.rmse(predictions, verbose=False)
        mae         = accuracy.mae(predictions,  verbose=False)

        coverage = len(predictions) / len(testset)

        mlflow.log_metric("rmse",     rmse)
        mlflow.log_metric("mae",      mae)
        mlflow.log_metric("coverage", coverage)

        print(f"  {run_name} | rmse={rmse:.4f} | mae={mae:.4f} | coverage={coverage:.4f}")

        dist_path = plot_rating_distribution(df, run_name)
        pred_path = plot_predictions_vs_actual(predictions, run_name)
        mlflow.log_artifact(dist_path)
        mlflow.log_artifact(pred_path)

        sample_user = df["UserId"].value_counts().index[0]
        recs        = get_top_n_recommendations(model, sample_user, df, n=10)
        print(f"    Sample recs for user {sample_user[:15]}...:")
        for product, score in recs[:3]:
            print(f"      {product} → predicted rating: {score}")

        mlflow.log_param("sample_user", sample_user)
        mlflow.log_text(
            str(recs),
            artifact_file="sample_recommendations.txt"
        )

        model_path = save_model(model, run_name)
        mlflow.log_artifact(model_path)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=mlflow.pyfunc.PythonModel(),
            artifacts={"model_path": model_path},
            registered_model_name="recommender",
        )

        return {
            "run_name":   run_name,
            "config":     config,
            "rmse":       rmse,
            "mae":        mae,
            "model_path": model_path,
        }

def main():
    mlflow.set_experiment("recommender")
    trainset, testset, df = load_data()

    dist_path = plot_rating_distribution(df, "overall")
    mlflow.set_experiment("recommender")

    results = []
    for i, config in enumerate(CONFIGS):
        print(f"\nTraining config {i+1}/{len(CONFIGS)}: {config}")
        result = train_and_log(config, trainset, testset, df, i + 1)
        results.append(result)

    best = min(results, key=lambda x: x["rmse"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        marker = " <- best" if r["run_name"] == best["run_name"] else ""
        print(f"  {r['run_name']} | rmse={r['rmse']:.4f} | mae={r['mae']:.4f}{marker}")

    print(f"\nBest model : {best['run_name']}")
    print(f"Best RMSE  : {best['rmse']:.4f}")
    print(f"Model path : {best['model_path']}")
    print("\nCheck MLflow UI at http://localhost:5001")

if __name__ == "__main__":
    main()
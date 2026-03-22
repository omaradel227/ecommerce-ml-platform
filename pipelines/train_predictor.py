import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import onnxruntime as rt
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models/predictor"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
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

CONFIGS = [
    {"n_estimators": 100, "max_depth": 6,  "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 8,  "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 6,  "learning_rate": 0.1,  "subsample": 0.9},
]

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "purchase_features.csv"))
    X  = df[FEATURE_COLS].astype(float)
    y  = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def plot_confusion_matrix(y_true, y_pred, run_name):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Buy", "Buy"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {run_name}")
    path = os.path.join(MODEL_DIR, f"confusion_matrix_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def plot_feature_importance(model, run_name):
    importance = model.feature_importances_
    fig, ax    = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURE_COLS, importance, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {run_name}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"feature_importance_{run_name}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def export_to_onnx(model, run_name):
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.get_booster().feature_names = [f"f{i}" for i in range(len(FEATURE_COLS))]
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model   = convert_xgboost(model_copy, initial_types=initial_type)
    path         = os.path.join(MODEL_DIR, f"predictor_{run_name}.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path

def verify_onnx(onnx_path, X_test, y_test):
    sess       = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    sample     = X_test.values[:1000].astype(np.float32)
    preds      = sess.run([label_name], {input_name: sample})[0]
    return accuracy_score(y_test.values[:1000], preds)

def train_and_log(config, X_train, X_test, y_train, y_test, run_index):
    run_name = f"xgb_smote_run_{run_index:02d}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("features",   len(FEATURE_COLS))
        mlflow.log_param("sampling",   "SMOTE")

        print("    Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"    Before SMOTE: {y_train.value_counts().to_dict()}")
        print(f"    After SMOTE : {pd.Series(y_train_resampled).value_counts().to_dict()}")
        mlflow.log_param("train_size_after_smote", len(X_train_resampled))

        model = XGBClassifier(
            **config,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("auc",       auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        print(f"  {run_name} | acc={accuracy:.4f} | f1={f1:.4f} | auc={auc:.4f}")

        cm_path = plot_confusion_matrix(y_test, y_pred, run_name)
        fi_path = plot_feature_importance(model, run_name)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        onnx_path = export_to_onnx(model, run_name)
        onnx_acc  = verify_onnx(onnx_path, X_test, y_test)
        mlflow.log_metric("onnx_accuracy", onnx_acc)
        mlflow.log_artifact(onnx_path)
        print(f"    ONNX verified | acc={onnx_acc:.4f}")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="purchase_predictor",
        )

        return {
            "run_name":  run_name,
            "config":    config,
            "accuracy":  accuracy,
            "f1_score":  f1,
            "auc":       auc,
            "onnx_path": onnx_path,
        }

def main():
    mlflow.set_experiment("purchase_predictor")
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Class balance: {y_train.value_counts().to_dict()}")

    results = []
    for i, config in enumerate(CONFIGS):
        print(f"\nTraining config {i+1}/{len(CONFIGS)}: {config}")
        result = train_and_log(config, X_train, X_test, y_train, y_test, i + 1)
        results.append(result)

    best = max(results, key=lambda x: x["auc"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        marker = " <- best" if r["run_name"] == best["run_name"] else ""
        print(f"  {r['run_name']} | acc={r['accuracy']:.4f} | f1={r['f1_score']:.4f} | auc={r['auc']:.4f}{marker}")

    print(f"\nBest model : {best['run_name']}")
    print(f"Best AUC   : {best['auc']:.4f}")
    print(f"ONNX model : {best['onnx_path']}")
    print("\nCheck MLflow UI at http://localhost:5001")

if __name__ == "__main__":
    main()
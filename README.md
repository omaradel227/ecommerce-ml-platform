# E-Commerce ML Platform

A production-grade machine learning platform serving three models for an e-commerce use case: purchase prediction, product recommendations, and product image classification. Built with a full MLOps stack.

---

## Architecture
```
Raw Data (CSV + Images)
        ↓
Preprocessing Pipeline (DVC versioned)
        ↓
Three ML Models (MLflow tracked)
        ↓
ONNX Export (framework-independent inference)
        ↓
FastAPI REST API
        ↓
Evidently Drift Monitoring
```

---

## Models

### Purchase Predictor
- **Algorithm:** XGBoost + SMOTE oversampling
- **Task:** Binary classification — predict whether a user will buy a product based on their rating behavior
- **Features:** User average rating, rating count, rating std, product average rating, product rating count, and encoded user/product IDs
- **Class imbalance:** 78% buy vs 22% no-buy — solved with SMOTE which synthetically oversampled the minority class, improving F1 from 0.83 to 0.89
- **ONNX export:** Yes — framework-independent inference at ~14ms latency

| Metric | Value |
|---|---|
| AUC | 0.8804 |
| F1 Score | 0.8876 |
| Accuracy | 0.8249 |
| Inference latency | ~14ms |

---

### Recommendation Engine
- **Algorithm:** SVD (Singular Value Decomposition) — Matrix Factorization via the Surprise library
- **Task:** Generate Top-N personalized product recommendations for a given user
- **Data:** 313,823 ratings from 51,369 users across 19,369 products (filtered from 2M raw ratings)
- **Approach:** Learns latent user and product embeddings from the interaction matrix. Users with similar taste get similar recommendations even for products they haven't rated
- **ONNX export:** Not applicable — SVD is a collaborative filtering model saved as pickle and served directly

| Metric | Value |
|---|---|
| RMSE | 1.1491 |
| MAE | 0.9013 |
| Coverage | 100% |
| Inference latency | ~113ms |

---

### Image Classifier
- **Algorithm:** ResNet50 with Transfer Learning (pretrained on ImageNet)
- **Task:** Classify product images into 5 categories: Toys & Games, Clothing & Shoes, Home & Kitchen, Sports & Outdoors, Baby Products
- **Approach:** Froze early ResNet50 layers, replaced final layer with Dropout(0.4) + Linear, fine-tuned layer3 and layer4 on product images
- **ONNX export:** Yes — exported via torch.onnx for framework-independent inference
- **Why 65% accuracy:** The dataset has only ~2,200 images across 5 categories with significant visual overlap. Toys & Games and Baby Products share very similar visual patterns (colorful plastic objects). Sports & Outdoors and Clothing overlap in athletic wear. A production system would require 10,000+ images per category with cleaner category boundaries to reach 85%+. The pipeline architecture (transfer learning, ONNX export, MLflow tracking) is production-ready — the accuracy limitation is purely a data quantity and quality issue.

| Metric | Value |
|---|---|
| Accuracy | 0.65 |
| Macro F1 | 0.58 |
| Inference latency | ~856ms |
| Categories | 5 |
---

## MLOps Stack

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking, model registry |
| DVC | Data and model versioning |
| Evidently | Data drift monitoring |
| ONNX | Framework-independent model export |
| ONNX Runtime | Optimized inference |
| GitHub Actions | CI/CD automated retraining |
| FastAPI | REST API serving |

---

## Project Structure
```
ecommerce-ml-platform/
├── pipelines/
│   ├── preprocess.py          # Data preprocessing + feature engineering
│   ├── train_predictor.py     # XGBoost purchase predictor training
│   ├── train_recommender.py   # SVD recommender training
│   └── train_classifier.py    # ResNet50 image classifier training
├── api/
│   └── app.py                 # FastAPI serving all 3 models
├── monitoring/
│   └── drift.py               # Evidently drift detection
├── .github/
│   └── workflows/
│       └── retrain.yml        # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

---

## Setup
```bash
# Clone the repo
git clone https://github.com/omaradel227/ecommerce-ml-platform
cd ecommerce-ml-platform

# Install dependencies
pip install -r requirements.txt

# Download datasets from Kaggle and place in data/raw/
# - https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings
# - https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020

# Run preprocessing
python3 pipelines/preprocess.py

# Train all models
python3 pipelines/train_predictor.py
python3 pipelines/train_recommender.py
python3 pipelines/train_classifier.py

# Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### Health Check
```
GET /health
```

### Purchase Prediction
```
POST /predict/purchase
```
```json
{
    "user_encoded": 1,
    "product_encoded": 1,
    "user_avg_rating": 4.2,
    "user_rating_count": 10,
    "user_rating_std": 0.8,
    "user_max_rating": 5.0,
    "user_min_rating": 3.0,
    "product_avg_rating": 4.1,
    "product_rating_count": 50,
    "product_rating_std": 0.9
}
```
Response:
```json
{
    "prediction": 1,
    "label": "buy",
    "confidence": 0.602,
    "buy_probability": 0.602,
    "latency_ms": 14
}
```

### Recommendations
```
POST /predict/recommend
```
```json
{
    "user_id": "A2V5R832QCSOMX",
    "n": 5
}
```
Response:
```json
{
    "user_id": "A2V5R832QCSOMX",
    "recommendations": [
        {"product_id": "B000142FVW", "predicted_rating": 4.96},
        {"product_id": "B0000536P3", "predicted_rating": 4.87}
    ],
    "rated_count": 223,
    "latency_ms": 113
}
```

### Image Classification
```
POST /predict/classify
```
```json
{
    "image_url": "https://images-na.ssl-images-amazon.com/images/I/51j3fPQTQkL.jpg"
}
```
Response:
```json
{
    "predicted_category": "Sports & Outdoors",
    "confidence": 0.9893,
    "all_scores": {
        "Baby Products": 0.0002,
        "Clothing, Shoes & Jewelry": 0.0002,
        "Home & Kitchen": 0.0021,
        "Sports & Outdoors": 0.9893,
        "Toys & Games": 0.0082
    },
    "latency_ms": 856
}
```

### Unified Endpoint
```
POST /predict
```
```json
{
    "model": "recommend",
    "recommend": {
        "user_id": "A2V5R832QCSOMX",
        "n": 5
    }
}
```

---

## Drift Monitoring

Evidently monitors the purchase predictor's input features for data drift using the Kolmogorov-Smirnov test.
```bash
python3 monitoring/drift.py
```

Results:
```
Normal data  — drift detected: False  (share: 0.0)
Drifted data — drift detected: True   (share: 0.625)
```

HTML reports are saved to `monitoring/reports/` with full visual distribution comparisons per feature.

---

## Experiment Tracking

All training runs are tracked in MLflow.
```bash
mlflow ui --port 5001
```

Open `http://localhost:5001` to see all experiments, compare runs, and view the model registry.

---

## CI/CD

GitHub Actions automatically retrains models when:
- Code in `pipelines/` changes
- Data in `data/` changes  
- Every Sunday at midnight (scheduled)
- Manually triggered from the GitHub Actions tab

---

## Performance

| Model | Metric | Value |
|---|---|---|
| Purchase Predictor | AUC | 0.8804 |
| Purchase Predictor | F1 Score | 0.8876 |
| Purchase Predictor | Accuracy | 0.8249 |
| Purchase Predictor | Inference latency | ~14ms |
| Recommender | RMSE | 1.1491 |
| Recommender | Inference latency | ~113ms |
| Image Classifier | Accuracy | 0.65 |
| Image Classifier | Inference latency | ~856ms |

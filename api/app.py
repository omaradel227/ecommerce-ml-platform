import os
import time
import pickle
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PREDICTOR_ONNX   = os.path.join(BASE_DIR, "models/predictor/predictor_xgb_smote_run_02.onnx")
RECOMMENDER_PKL  = os.path.join(BASE_DIR, "models/recommender/svd_svd_run_01.pkl")
CLASSIFIER_ONNX  = os.path.join(BASE_DIR, "models/classifier/classifier_resnet50_run_01.onnx")
LABEL_ENCODER    = os.path.join(BASE_DIR, "models/classifier/label_encoder.pkl")
RATINGS_CSV      = os.path.join(BASE_DIR, "data/processed/ratings_clean.csv")
IMG_SIZE         = 224

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    print("  Loading purchase predictor...")
    state["predictor"] = rt.InferenceSession(PREDICTOR_ONNX)

    print("  Loading recommender...")
    with open(RECOMMENDER_PKL, "rb") as f:
        state["recommender"] = pickle.load(f)

    print("  Loading image classifier...")
    state["classifier"]    = rt.InferenceSession(CLASSIFIER_ONNX)
    with open(LABEL_ENCODER, "rb") as f:
        state["label_encoder"] = pickle.load(f)

    print("  Loading ratings data...")
    import pandas as pd
    state["ratings"] = pd.read_csv(RATINGS_CSV)

    state["start_time"] = time.time()
    print("All models loaded. API ready.")
    yield
    state.clear()

app = FastAPI(
    title="E-Commerce ML Platform",
    version="1.0.0",
    description="Serves purchase predictor, recommender, and image classifier",
    lifespan=lifespan,
)

class PurchaseRequest(BaseModel):
    user_encoded:          float
    product_encoded:       float
    user_avg_rating:       float
    user_rating_count:     float
    user_rating_std:       float
    user_max_rating:       float
    user_min_rating:       float
    product_avg_rating:    float
    product_rating_count:  float
    product_rating_std:    float

class RecommendRequest(BaseModel):
    user_id: str
    n:       Optional[int] = 10

class ClassifyRequest(BaseModel):
    image_url: str

class PredictRequest(BaseModel):
    model:    str
    purchase: Optional[PurchaseRequest]  = None
    recommend: Optional[RecommendRequest] = None
    classify:  Optional[ClassifyRequest]  = None

def load_image_from_url(url: str) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    response = requests.get(url, timeout=5)
    img      = Image.open(BytesIO(response.content)).convert("RGB")
    tensor   = transform(img).unsqueeze(0)
    return tensor.numpy()

@app.get("/health")
def health():
    uptime = round(time.time() - state.get("start_time", time.time()))
    return {
        "status":   "healthy",
        "uptime_s": uptime,
        "models": {
            "purchase_predictor": "loaded" if "predictor"   in state else "not loaded",
            "recommender":        "loaded" if "recommender" in state else "not loaded",
            "image_classifier":   "loaded" if "classifier"  in state else "not loaded",
        }
    }

@app.post("/predict/purchase")
def predict_purchase(request: PurchaseRequest):
    start = time.time()

    features = np.array([[
        request.user_encoded,
        request.product_encoded,
        request.user_avg_rating,
        request.user_rating_count,
        request.user_rating_std,
        request.user_max_rating,
        request.user_min_rating,
        request.product_avg_rating,
        request.product_rating_count,
        request.product_rating_std,
    ]], dtype=np.float32)

    sess       = state["predictor"]
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name  = sess.get_outputs()[1].name

    prediction  = sess.run([label_name], {input_name: features})[0][0]
    probabilities = sess.run([prob_name], {input_name: features})[0][0]

    return {
        "prediction":      int(prediction),
        "label":           "buy" if prediction == 1 else "no_buy",
        "confidence":      round(float(max(probabilities)), 4),
        "buy_probability": round(float(probabilities[1]), 4),
        "latency_ms":      round((time.time() - start) * 1000),
    }

@app.post("/predict/recommend")
def predict_recommend(request: RecommendRequest):
    start    = time.time()
    df       = state["ratings"]
    model    = state["recommender"]

    if request.user_id not in df["UserId"].values:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found.")

    rated_products = df[df["UserId"] == request.user_id]["ProductId"].tolist()
    all_products   = df["ProductId"].unique()
    unrated        = [p for p in all_products if p not in rated_products]

    predictions = [model.predict(request.user_id, p) for p in unrated[:500]]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:request.n]

    return {
        "user_id":         request.user_id,
        "recommendations": [
            {"product_id": p.iid, "predicted_rating": round(p.est, 2)}
            for p in top_n
        ],
        "rated_count":  len(rated_products),
        "latency_ms":   round((time.time() - start) * 1000),
    }

@app.post("/predict/classify")
def predict_classify(request: ClassifyRequest):
    start = time.time()

    try:
        image = load_image_from_url(request.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    sess       = state["classifier"]
    input_name = sess.get_inputs()[0].name
    logits     = sess.run(None, {input_name: image})[0][0]

    probs      = np.exp(logits) / np.exp(logits).sum()
    pred_idx   = int(np.argmax(probs))
    le         = state["label_encoder"]

    return {
        "predicted_category": le.classes_[pred_idx],
        "confidence":         round(float(probs[pred_idx]), 4),
        "all_scores": {
            le.classes_[i]: round(float(probs[i]), 4)
            for i in range(len(le.classes_))
        },
        "latency_ms": round((time.time() - start) * 1000),
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if request.model == "purchase":
        if not request.purchase:
            raise HTTPException(status_code=400, detail="purchase field required for model=purchase")
        return predict_purchase(request.purchase)

    elif request.model == "recommend":
        if not request.recommend:
            raise HTTPException(status_code=400, detail="recommend field required for model=recommend")
        return predict_recommend(request.recommend)

    elif request.model == "classify":
        if not request.classify:
            raise HTTPException(status_code=400, detail="classify field required for model=classify")
        return predict_classify(request.classify)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: purchase, recommend, classify"
        )
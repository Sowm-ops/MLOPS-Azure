import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Global holders (initialized on startup)
imdb_model = None
heart_model = None
vectorizer = None
scaler = None


# ---------------------------
# Load models on startup
# ---------------------------
@app.on_event("startup")
def load_models():
    global imdb_model, heart_model, vectorizer, scaler

    model_dir = os.path.join(os.path.dirname(__file__), "models")

    try:
        imdb_model = joblib.load(os.path.join(model_dir, "imdb_best.pkl"))
        heart_model = joblib.load(os.path.join(model_dir, "heart_best.pkl"))
        vectorizer = joblib.load(os.path.join(model_dir, "imdb_vectorizer.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "heart_scaler.pkl"))

        print("✅ All models loaded successfully")

    except Exception as e:
        print("❌ Model loading failed:", e)
        raise RuntimeError(e)


# ---------------------------
# Request models
# ---------------------------
class IMDBRequest(BaseModel):
    text: str

class HeartRequest(BaseModel):
    features: list[float]


# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    if imdb_model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok"}


# ---------------------------
# Prediction endpoints
# ---------------------------
@app.post("/predict/imdb")
def predict_imdb(request: IMDBRequest):
    x = vectorizer.transform([request.text])
    pred = imdb_model.predict(x)
    return {"prediction": int(pred[0])}


@app.post("/predict/heart")
def predict_heart(request: HeartRequest):
    x = scaler.transform([request.features])
    pred = heart_model.predict(x)
    return {"prediction": int(pred[0])}

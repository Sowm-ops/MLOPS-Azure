import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load models
imdb_model = joblib.load("models/imdb_best.pkl")
heart_model = joblib.load("models/heart_best.pkl")
vectorizer = joblib.load("models/imdb_vectorizer.pkl")
scaler = joblib.load("models/heart_scaler.pkl")


# Request models
class IMDBRequest(BaseModel):
    text: str

class HeartRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


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

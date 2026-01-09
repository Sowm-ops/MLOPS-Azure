import joblib
from fastapi import FastAPI

app = FastAPI()

imdb_model = joblib.load("models/imdb_best.pkl")
heart_model = joblib.load("models/heart_best.pkl")
vectorizer = joblib.load("models/imdb_vectorizer.pkl")
scaler = joblib.load("models/heart_scaler.pkl")


@app.post("/predict/imdb")
def predict_imdb(text: str):
    x = vectorizer.transform([text])
    pred = imdb_model.predict(x)
    return {"prediction": int(pred[0])}


@app.post("/predict/heart")
def predict_heart(features: list[float]):
    x = scaler.transform([features])
    pred = heart_model.predict(x)
    return {"prediction": int(pred[0])}

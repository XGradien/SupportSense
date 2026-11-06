import os, joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = os.path.dirname(__file__)
model = joblib.load(os.path.join(HERE, "model.joblib"))
vec = model.named_steps["tfidf"]; clf = model.named_steps["clf"]
CLASSES = list(clf.classes_)

def top_terms(cls, n=6):
    idx = list(clf.classes_).index(cls)
    coef = clf.coef_[idx]; feat = vec.get_feature_names_out()
    keep = [i for i in coef.argsort()[::-1] if coef[i] > 0][:n]
    return [feat[i] for i in keep][:n]

TERMS = {c: top_terms(c) for c in CLASSES}
app = FastAPI(title="SupportSense", version="1.0.0")

class In(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded", "intents": len(CLASSES)}

@app.post("/predict")
def predict(d: In):
    t = d.text.strip()
    if not t:
        return {"error": "empty text"}
    p = model.predict_proba([t.lower().strip()])[0]
    order = list(p.argsort()[::-1])
    top = CLASSES[order[0]]; conf = float(p[order[0]])
    return {
        "text": t, "intent": top, "confidence": round(conf, 4), "confidence_pct": round(conf * 100, 1),
        "key_terms": TERMS.get(top, []),
        "alternatives": [{"intent": CLASSES[i], "confidence": round(float(p[i]), 4)} for i in order[1:4]],
    }

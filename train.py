import json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from data.intents import SENTENCES, INTENTS

texts  = [t.lower().strip() for t, _ in SENTENCES]
labels = [b for _, b in SENTENCES]
Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", min_df=1, norm="l2")),
    ("clf",   LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", solver="lbfgs")),
])
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
rep = classification_report(yte, pred, zero_division=0, output_dict=True)
acc = round(float(accuracy_score(yte, pred)), 4)
cv = cross_val_score(pipe, Xtr, ytr, cv=3)
metrics = {
    "n_total": len(SENTENCES), "n_train": len(Xtr), "n_test": len(Xte),
    "test_accuracy": acc,
    "cv_mean": round(float(cv.mean()), 4), "cv_std": round(float(cv.std()), 4),
    "per_class": {k: {c: round(rep[k][c], 3) for c in ("precision", "recall", "f1-score")} for k in INTENTS},
    "params": {"vectorizer": "TfidfVectorizer unigram+bigram, sublinear_tf, L2",
               "classifier": "LogisticRegression C=1.0, class_weight=balanced, lbfgs"},
}
joblib.dump(pipe, "model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2)
print("samples:", len(SENTENCES), "| train", len(Xtr), "| test", len(Xte))
print("TEST ACCURACY:", acc)
print(classification_report(yte, pred, zero_division=0))
print("3-FOLD CV:", round(float(cv.mean()),4), "+/-", round(float(cv.std()),4))
print("Saved model.joblib + metrics.json")

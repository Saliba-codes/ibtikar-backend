from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="IbtikarAI Toxicity API")

# ---------- Models ----------

class TextsIn(BaseModel):
    texts: List[str]


MODEL_DIR = Path(__file__).parent / "arabert_toxic_classifier"

# Load tokenizer + model once on startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # disable dropout etc.

# Try to find which output index is "toxic"
id2label = model.config.id2label or {}
toxic_index = None
for i, name in id2label.items():
    if "toxic" in str(name).lower():
        toxic_index = int(i)
        break

# Fallback: assume index 1 is toxic
if toxic_index is None:
    toxic_index = 1


@app.post("/predict")
def predict(inp: TextsIn):
    """
    Input:  { "texts": ["...", "..."] }
    Output: { "preds": [ {"label": "harmful"/"safe", "score": float}, ... ] }
    """

    if not inp.texts:
        return {"preds": []}

    enc = tokenizer(
        inp.texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**enc)
        probs = outputs.logits.softmax(dim=-1)

    preds = []
    for p in probs:
        p = p.cpu()
        toxic_prob = float(p[toxic_index])
        label = "harmful" if toxic_prob >= 0.5 else "safe"  # adjust threshold later if you want
        preds.append({"label": label, "score": toxic_prob})

    return {"preds": preds}

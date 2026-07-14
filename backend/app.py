"""
FastAPI backend for the Tiny LLM demo.

Loads the checkpoint trained in Colab (backend/checkpoints/tiny_llm.pt)
and exposes a single /generate endpoint that the frontend calls.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import ModelConfig, TinyGPT

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "tiny_llm.pt")

app = FastAPI(title="Tiny LLM Demo API")

# Allow the frontend (served from anywhere, e.g. file:// or GitHub Pages) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
stoi = None
itos = None


def load_model():
    global model, stoi, itos
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. "
            "Train the model in train_colab.ipynb and place tiny_llm.pt there."
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    config = ModelConfig(**ckpt["config"])
    m = TinyGPT(config)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device)
    m.eval()

    model = m
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}


@app.on_event("startup")
def on_startup():
    load_model()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(200, ge=1, le=500)
    temperature: float = Field(0.8, gt=0, le=2.0)
    top_k: int = Field(40, ge=1, le=200)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model_loaded": model is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    unknown = [c for c in req.prompt if c not in stoi]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt contains characters not seen during training: {unknown[:10]}",
        )

    idx = torch.tensor([[stoi[c] for c in req.prompt]], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    text = "".join(itos[i] for i in out[0].tolist())
    return GenerateResponse(prompt=req.prompt, completion=text)

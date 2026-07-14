"""
Flask backend for PythonAnywhere free tier.

PythonAnywhere's free plan only runs traditional WSGI apps (not the ASGI
server uvicorn uses), so this is a Flask equivalent of backend/app.py with
the same /health and /generate endpoints. Same model.py is reused unchanged.

Deployment: see README.md "Deploy on PythonAnywhere" section.
"""

import os
import torch
from flask import Flask, request, jsonify

from model import ModelConfig, TinyGPT

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "tiny_llm.pt")

app = Flask(__name__)

device = "cpu"  # PythonAnywhere free tier is CPU only
model = None
stoi = None
itos = None


def load_model():
    global model, stoi, itos
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    config = ModelConfig(**ckpt["config"])
    m = TinyGPT(config)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device)
    m.eval()

    model = m
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}


# Load once when the app starts (PythonAnywhere keeps the process warm between requests)
load_model()


@app.after_request
def add_cors_headers(response):
    # Allow the frontend (hosted anywhere) to call this API
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device, "model_loaded": model is not None})


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        # CORS preflight
        return "", 204

    if model is None:
        return jsonify({"detail": "Model not loaded"}), 503

    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "")
    max_new_tokens = min(int(data.get("max_new_tokens", 200)), 500)
    temperature = min(max(float(data.get("temperature", 0.8)), 0.01), 2.0)
    top_k = min(max(int(data.get("top_k", 40)), 1), 200)

    if not prompt:
        return jsonify({"detail": "prompt is required"}), 400

    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        return jsonify({
            "detail": f"Prompt contains characters not seen during training: {unknown[:10]}"
        }), 400

    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = "".join(itos[i] for i in out[0].tolist())

    return jsonify({"prompt": prompt, "completion": text})


if __name__ == "__main__":
    app.run(debug=True)

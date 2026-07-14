# Tiny LLM Demo

A minimal end-to-end example of training a very small character-level GPT
(under 1M parameters) and serving it through a simple web app.

- **Training**: `train_colab.ipynb` — runs on a free Google Colab GPU in a few minutes.
- **Dataset**: [Tiny Shakespeare on Kaggle](https://www.kaggle.com/datasets/thedevastator/the-bards-best-a-character-modeling-dataset)
- **Backend**: `backend/app.py` — a FastAPI server that loads the trained checkpoint and exposes `/generate`.
- **Frontend**: `frontend/index.html` — a single static page that calls the backend and shows the model "typing" its output.

```
tiny-llm-project/
├── train_colab.ipynb        # open this in Google Colab
├── backend/
│   ├── model.py              # model architecture (shared with the notebook)
│   ├── app.py                 # FastAPI server
│   ├── requirements.txt
│   └── checkpoints/           # put tiny_llm.pt here after training
├── frontend/
│   └── index.html
└── README.md
```

## 1. Train the model in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com), choose **Upload**, and select `train_colab.ipynb` from this repo (or open it directly from GitHub once you've pushed the repo, via `File > Open notebook > GitHub`).
2. `Runtime > Change runtime type > T4 GPU`.
3. Run every cell top to bottom.
   - Cell 2 lets you upload a `kaggle.json` API token so the notebook pulls the **real Kaggle dataset** linked above. If you skip the upload, it automatically falls back to a no-login mirror of the same Tiny Shakespeare text so the notebook still runs end-to-end.
   - Training takes roughly 2–5 minutes on a T4 GPU for the default settings (~0.8M parameters, 2000 steps).
4. The last cell saves `tiny_llm.pt` and downloads it to your computer automatically.

To get a `kaggle.json` token: Kaggle → your profile photo → **Settings** → **API** → **Create New Token**.

## 2. Run the backend locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Move the checkpoint you downloaded from Colab into `backend/checkpoints/tiny_llm.pt`, then:

```bash
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000/health` — you should see `"model_loaded": true`.

## 3. Run the frontend

The frontend is a single static HTML file, no build step needed.

- Easiest: just open `frontend/index.html` directly in your browser.
- Or serve it so it behaves like a real site:
  ```bash
  cd frontend
  python -m http.server 5500
  ```
  then visit `http://localhost:5500`.

The page calls `API_URL` (set at the top of the `<script>` block in `index.html`, default `http://localhost:8000`) — change that constant if you deploy the backend elsewhere.

## 4. Push the project to GitHub from VS Code

1. Open the `tiny-llm-project` folder in VS Code (`File > Open Folder…`).
2. Open the built-in terminal (`` Ctrl+` ``) and initialize git:
   ```bash
   git init
   git add .
   git commit -m "Tiny LLM demo: training notebook, backend, frontend"
   ```
3. Create a new empty repo on [github.com/new](https://github.com/new) (don't initialize it with a README).
4. Connect and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
5. Alternatively, use VS Code's **Source Control** panel (left sidebar icon) → **Publish to GitHub**, which does steps 2–4 for you through the UI.

Note: `.gitignore` excludes the trained `.pt` checkpoint by default since model weights don't belong in git history well. If you want the checkpoint versioned in the repo too, remove the `backend/checkpoints/*.pt` line from `.gitignore`, or use [Git LFS](https://git-lfs.com) for it.

## 5. (Optional) Deploy it live

- **Backend**: any host that runs a Python web app works — [Render](https://render.com), [Railway](https://railway.app), or a [Hugging Face Space](https://huggingface.co/spaces) (Docker SDK). Point it at `backend/`, run `uvicorn app:app --host 0.0.0.0 --port $PORT`, and make sure the checkpoint file is included (Git LFS or uploaded separately).
- **Frontend**: [GitHub Pages](https://pages.github.com) — in your repo, `Settings > Pages > Deploy from branch`, pick `main` and the `/frontend` folder. Then update `API_URL` in `index.html` to your deployed backend's URL before pushing.

## Tuning the model

Everything is deliberately tiny for fast iteration. In `train_colab.ipynb`, the config cell exposes:

| Param | Default | Effect |
|---|---|---|
| `n_layer` | 4 | transformer blocks — more = slower, more capacity |
| `n_head` | 4 | attention heads |
| `n_embd` | 128 | embedding width |
| `block_size` | 128 | context length in characters |
| `max_iters` | 2000 | training steps |

Doubling `n_embd` and `n_layer` roughly quadruples parameter count and training time, and noticeably improves coherence if you want a slightly-less-tiny demo.

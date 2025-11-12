# Speech Emotion & Sarcasm Detector — Deployment

This repository contains a Streamlit web app for speech emotion recognition and a basic sarcasm detector. The Streamlit app entrypoint is `app.py` which loads the main app from `results/ph5_streamlit.py`.

## Quick local run (Windows PowerShell)

1. Create and activate a virtual environment

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
2. Install dependencies

```powershell
pip install -r requirements.txt
```
3. Run the app

```powershell
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (recommended quick option)

1. Push this repository to GitHub.
2. On Streamlit Cloud, create a new app and point it to your GitHub repo, branch, and the file path `app.py`.
3. Streamlit Cloud will install from `requirements.txt` and start the app.

Notes:
- If your `models/` directory contains large `.h5` files, consider using Git LFS or hosting models in cloud storage and downloading them at runtime to avoid repository size limits.

## Using Git LFS for model files (optional)

Install Git LFS and track model files before committing large binaries:

```powershell
# Install Git LFS (if not installed)
# On Windows, download from https://git-lfs.github.com/ or use choco if available
git lfs install

# Track model files and commit
git lfs track "models/*.h5"
git add .gitattributes
git add models/*.h5
git commit -m "Add model files via Git LFS"
git push origin main
```

## Deploy with Docker (alternative)

1. Build the image

```powershell
docker build -t speech-sarcasm-app .
```

2. Run the container

```powershell
docker run -p 8501:8501 speech-sarcasm-app
```

## Notes and next steps

- If you prefer another host (Render, Railway, Azure App Service, AWS ECS), I can add a specific deployment guide or provide infrastructure files (service definitions, cloud-specific Dockerfile tweaks, CI/GitHub Actions).
- If you want me to push this to GitHub from your machine, I can provide the exact PowerShell commands or a small script to initialize the repo and push (you must run them locally).

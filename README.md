# VeriCred - Academic Certificate Verification (prototype)

Upload a certificate → OCR + YOLO true/fake regions + QR/URL evidence →
combined status + human-readable report. Blockchain comes later.

## Run
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# image OCR needs the Tesseract binary: sudo apt install tesseract-ocr
.venv/bin/python -m pytest -q
.venv/bin/streamlit run app.py
```

The default fake-region confidence threshold is `0.60`.

## Windows setup (PowerShell)
```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install openai

$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
$env:LLM_MODEL = "your-openrouter-model-id"

.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run tests on Windows with:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Dataset (not in git)
Kaggle/Roboflow “Certificate Forgery Detection”, YOLO format, 0=fake 1=true.
Place at `d1Certificate_forgery_detection/` (train/valid/test + data.yaml).
Train via `notebooks/train_yolo.ipynb` (needs GPU for a real run);
copy `best.pt` to `weights/` (gitignored). Inference reads
`VERICRED_WEIGHTS` or `weights/best.pt`; without weights the UI reports
“detector unavailable” and never invents boxes.

# Kepler Vision

Kepler Vision is a responsive web workspace for exploring Kepler light curves. It is powered by **AstroLens**, an experimental multi-scale CNN that screens for transit-like patterns.

## Start Kepler Vision on Windows

Create a Python 3.10 environment for TensorFlow. From PowerShell in the project folder:

```powershell
py -3.10 -m venv .venv-cnn
.\.venv-cnn\Scripts\Activate.ps1
pip install -e ".[cnn]"
```

Train AstroLens and evaluate it on the held-out CSV:

```powershell
python -m keplervision.train data/exoTrain.csv --test-csv data/exoTest.csv --model multiscale_cnn --output models/multiscale_cnn.keras
```

Start the API and web app:

```powershell
python -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser. API documentation is at **http://127.0.0.1:8000/docs**. The frontend is plain HTML, CSS, and JavaScript served by FastAPI; it sends CSV observations to the prediction API in batches.

## Save predictions with PostgreSQL

The direct `uvicorn` command above runs the API without a database. To run the full local stack, including PostgreSQL, install and start Docker Desktop, then in PowerShell from this project folder run:

```powershell
docker compose up --build
```

Wait for the API startup message, then open **http://127.0.0.1:8000**. Docker Compose starts PostgreSQL first, waits until it is healthy, and then starts the API. Each prediction is saved with its generated observation ID, optional object ID, 3,197 flux features, class, probabilities, AstroLens version, and timestamp. Recent saved predictions are available at **http://127.0.0.1:8000/predictions**; the API docs are at **http://127.0.0.1:8000/docs**.

Press **Ctrl+C** in the PowerShell window to stop the stack. PostgreSQL data is kept in a Docker volume between starts. The included database credentials are for local development only, and the database port is bound to this computer (`127.0.0.1`).

## Data

The included `data/exoTrain.csv` and `data/exoTest.csv` come from the [Kaggle Kepler labelled time-series dataset](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data). Each row contains `FLUX.1` through `FLUX.3197`; `LABEL` 1 means no planet and 2 means planet. `data/README.md` documents the local files and the limits of this split.

### DVC dataset tracking

The two CSVs are tracked by DVC. DVC keeps their large contents in its local cache; the small `data/exoTrain.csv.dvc` and `data/exoTest.csv.dvc` files record their content hashes for Git. Check that the local data matches those records with:

```powershell
dvc status
```

When you are ready to make your first Git commit, add the project files and DVC pointers (the raw CSVs stay ignored by Git):

```powershell
git add .gitignore .dvcignore .dvc/config pyproject.toml README.md data/README.md data/exoTrain.csv.dvc data/exoTest.csv.dvc
git status
```

There is not a DVC remote configured yet, so the cached data is only on this computer. To share or restore the data on another computer, a remote storage location must be configured first. Do not run `dvc push` yet.

## AstroLens evaluation

The current multi-scale CNN evaluation found 1 of 5 planet examples in `exoTest.csv` (20% recall), with 33.3% precision. Its 98.95% accuracy is misleading because the test set is highly imbalanced. Treat this as a baseline and learning project, not a validated detector. Use larger, appropriately separated data before making scientific claims.

Training prints balanced accuracy, Matthews correlation, the fixed 0.50 decision threshold, and a 95% Wilson interval for planet recall alongside the existing metrics. With only five positive test examples, the recall interval is very wide; these extra statistics make the uncertainty visible, but they do not create more evidence or improve the model. Keep `exoTest.csv` held out when comparing training choices.

To evaluate the model artifact you already trained, without retraining or overwriting it:

```powershell
python -m keplervision.evaluate --model models/multiscale_cnn.keras --csv data/exoTest.csv --report reports/astro-lens-evaluation.json
```

The command prints the metrics and saves the same report as JSON. It uses the existing 0.50 threshold and does not tune it on the held-out test file.

## Optional local experiment tracking

Install the MLflow extra once, using the same Python environment as the project:

```powershell
pip install -e ".[mlops]"
```

Add `--track-mlflow` to either the evaluation command above or the training command to record metrics, the fixed threshold, model files, and SHA-256 fingerprints of the source CSV files in a local `mlruns` folder. For example:

```powershell
python -m keplervision.evaluate --model models/multiscale_cnn.keras --csv data/exoTest.csv --report reports/astro-lens-evaluation.json --track-mlflow
python -m mlflow ui
```

Then open **http://127.0.0.1:5000** to browse recorded runs. Tracking is opt-in; ordinary training and evaluation do not require MLflow.

AstroLens uses training-only scaling and minority-class oversampling, then four parallel 1D convolution branches (kernel widths 3, 5, 11, and 21). The model and scaler are saved separately under `models/`.

NASA resources: [MAST Kepler archive](https://archive.stsci.edu/missions-and-data/kepler) · [NASA Exoplanet Archive KOI guidance](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html).

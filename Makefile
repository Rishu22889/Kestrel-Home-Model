PY ?= .venv/bin/python

setup:            ## create venv and install pinned deps
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

features:         ## clean data + build features (needs the data pack in this folder)
	$(PY) -m src.step1_report

backtest:         ## strict walk-forward evidence -> reports/
	$(PY) -m src.backtest && $(PY) -m src.error_analysis && $(PY) -m src.experiments

train: features   ## final model + predictions.csv + models/
	$(PY) -m src.train_final

serve:            ## http://127.0.0.1:8000
	$(PY) -m uvicorn app.main:app --port 8000

all: setup train backtest

.PHONY: setup features backtest train serve all

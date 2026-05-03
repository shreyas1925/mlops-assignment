PYTHONPATH_EXPORT=PYTHONPATH=src:.

.PHONY: install lint test train run-api podman

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) ruff check .

test:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) pytest -q

train:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) python scripts/run_training_pipeline.py

run-api:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) uvicorn api.main:app --host 0.0.0.0 --port 8000

podman:
	bash scripts/podman_build_run.sh


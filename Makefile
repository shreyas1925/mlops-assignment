PYTHONPATH_EXPORT=PYTHONPATH=src:.

.PHONY: install lint test train up down logs predict clean

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) ruff check .

test:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) pytest -q

train:
	. .venv/bin/activate && $(PYTHONPATH_EXPORT) python -m pipeline

up:
	podman compose up -d --build

down:
	podman compose down

logs:
	podman compose logs -f api

predict:
	bash scripts/test_predict.sh

clean:
	podman compose down -v
	rm -rf artifacts/.cache artifacts/.matplotlib

PY := .venv/bin/python

.PHONY: all data features splits baselines train eval train-q train-c eval-q \
	explain test clean

all:
	$(PY) -m src.pipeline all

data:
	$(PY) -m src.pipeline data

features:
	$(PY) -m src.pipeline features

splits:
	$(PY) -m src.pipeline splits

baselines:
	$(PY) -m src.pipeline baselines

train:
	$(PY) -m src.pipeline model-r

eval:
	$(PY) -m src.pipeline eval

train-q:
	$(PY) -m src.pipeline model-q

train-c:
	$(PY) -m src.pipeline model-c

eval-q:
	$(PY) -m src.pipeline eval-q

explain:
	@echo "explain: not implemented yet (Phase 6)"

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim data/processed outputs

PY := .venv/bin/python

.PHONY: all data features splits baselines train eval explain test clean

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

explain:
	@echo "explain: not implemented yet (Phase 6)"

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim data/processed outputs

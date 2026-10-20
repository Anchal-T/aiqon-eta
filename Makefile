PY := .venv/bin/python

.PHONY: all data features train eval explain test clean

all: data

data:
	$(PY) -m src.generate --out data/raw
	$(PY) -m src.ingest
	$(PY) -m src.missingness
	$(PY) -m src.delays

features:
	@echo "features: not implemented yet (Phase 2)"

train:
	@echo "train: not implemented yet (Phase 4)"

eval:
	@echo "eval: not implemented yet (Phase 5)"

explain:
	@echo "explain: not implemented yet (Phase 6)"

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim data/processed outputs

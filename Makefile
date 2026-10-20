PY := .venv/bin/python

.PHONY: all data features splits train eval explain test clean

all: splits

data:
	$(PY) -m src.generate --out data/raw
	$(PY) -m src.ingest
	$(PY) -m src.missingness
	$(PY) -m src.delays

features: data
	$(PY) -m src.features

splits: features
	$(PY) -m src.splits

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

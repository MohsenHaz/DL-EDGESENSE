.PHONY: train eval ablation benchmark test clean

train:
	python scripts/train.py --config configs/standard_mode.yaml

eval:
	python scripts/evaluate.py --config configs/standard_mode.yaml

ablation:
	python scripts/run_ablation.py --config configs/base_config.yaml

benchmark:
	python scripts/run_benchmark.py --config configs/base_config.yaml

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Default tooling
PYTHON ?= python
PIP ?= pip

# Data and training configuration
DATA_DIR ?= data/datasets
CHECKPOINT_ROOT ?= checkpoints
TRAIN_SIZE ?= 1000
VAL_SIZE ?= 200
TEST_SIZE ?= 200
N_ITEMS_MIN ?= 10
N_ITEMS_MAX ?= 50
EPOCHS ?= 50
BATCH_SIZE ?= 32
LEARNING_RATE ?= 0.002
WEIGHT_DECAY ?= 1e-6
HIDDEN_DIM ?= 64
NUM_LAYERS ?= 3
DROPOUT ?= 0.1
SEED ?= 42
GENERATE_DATA ?= 1
DEVICE ?= cpu

empty :=
space := $(empty) $(empty)
comma := ,

# Evaluation configuration
CHECKPOINT_DIR ?= $(shell ls -td $(CHECKPOINT_ROOT)/run_* 2>/dev/null | head -1)
STRATEGY ?= sampling
N_SAMPLES ?= 500
TEMPERATURE ?= 1.0
THRESHOLD ?= 0.5
TEST_ONLY ?= 0
SAMPLING_SCHEDULE ?= 32,64,128
SAMPLING_TOL ?= 1e-3
MAX_SAMPLES ?=
MAX_HINT_ITEMS ?=
LAGRANGIAN_ITERS ?= 30
LAGRANGIAN_TOL ?= 1e-4
LAGRANGIAN_BIAS ?= 0.0
FIX_THRESHOLD ?= 0.9
ILP_TIME_LIMIT ?= 1.0
ILP_THREADS ?=
THREADS ?=
COMPILE ?= 0
QUANTIZE ?= 0
PIPELINE_STRATEGIES ?= sampling warm_start
SKIP_TRAIN ?= 0

# OOD evaluation configuration
OOD_SIZES ?= 100 150 200
OOD_INSTANCES ?= 50

# Extra arguments (leave empty or override per invocation)
TRAIN_ARGS ?=
EVAL_ARGS ?=
OOD_ARGS ?=

ifdef CHECKPOINT
CHECKPOINT_DIR := $(CHECKPOINT)
endif

.PHONY: install train evaluate ood demo quick_train pipeline clean \
        sync-deps check-deps format lint mypy test test-quick docs docs-serve commit \
        validate-configs download-checkpoint generate-datasets verify-reproducibility

install:
	$(PIP) install -r requirements.txt

# ============================================================================
# Development tooling targets
# ============================================================================

# Sync dependencies: regenerate requirements files from pyproject.toml
sync-deps:
	@echo "Syncing dependencies from pyproject.toml..."
	uv pip compile pyproject.toml --extra cpu -o requirements.txt --generate-hashes
	uv pip compile pyproject.toml --extra dev -o requirements-dev.txt --generate-hashes
	@echo "✅ Dependencies synced successfully"
	@echo "Note: For CUDA, install torch from PyTorch index: pip install torch==2.9.0+cu118 --index-url https://download.pytorch.org/whl/cu118"

# Check dependencies: fail if requirements files are out of sync
check-deps:
	@echo "Checking dependency drift..."
	@TMP_DIR=$$(mktemp -d); \
	cp requirements*.txt $$TMP_DIR/ 2>/dev/null || true; \
	$(MAKE) -s sync-deps >/dev/null 2>&1; \
	if ! git diff --quiet requirements*.txt 2>/dev/null; then \
		echo "❌ Dependencies out of sync with pyproject.toml"; \
		echo "Run: make sync-deps"; \
		git diff requirements*.txt; \
		git checkout requirements*.txt 2>/dev/null || true; \
		exit 1; \
	else \
		echo "✅ Dependencies in sync"; \
	fi

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	ruff format src/ experiments/ tests/
	@echo "✅ Code formatted"

# Lint code with ruff
lint:
	@echo "Linting code with ruff..."
	ruff check src/ experiments/ tests/ --output-format=concise
	@echo "✅ Lint check complete"

# Type check with mypy
mypy:
	@echo "Type checking with mypy..."
	mypy src/knapsack_gnn/ experiments/ --ignore-missing-imports
	@echo "✅ Type check complete"

# Run tests with coverage
test:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src/knapsack_gnn --cov-report=term-missing --cov-report=html
	@echo "✅ Tests complete (coverage report: htmlcov/index.html)"

# Run quick tests (exclude slow tests)
test-quick:
	@echo "Running quick tests..."
	pytest tests/ -v -m "not slow" --cov=src/knapsack_gnn --cov-report=term-missing
	@echo "✅ Quick tests complete"

# Build documentation
docs:
	@echo "Building documentation..."
	mkdocs build
	@echo "✅ Documentation built (site/index.html)"

# Serve documentation locally
docs-serve:
	@echo "Serving documentation at http://127.0.0.1:8000"
	mkdocs serve

# Launch Jupyter Lab for interactive notebooks
notebooks:
	@echo "Launching Jupyter Lab..."
	@echo "Notebooks available in: notebooks/"
	jupyter lab notebooks/

# Generate API documentation
api-docs:
	@echo "Generating API documentation..."
	mkdocs build
	@echo "✅ API docs generated in site/"

# Interactive commit message helper (Conventional Commits)
commit:
	@echo "Creating commit with Conventional Commits format..."
	@echo "Tip: Use commitizen for guided commit message creation"
	@if command -v cz >/dev/null 2>&1; then \
		cz commit; \
	else \
		echo "❌ commitizen not found. Install with: pip install commitizen"; \
		echo "Or commit manually following format: <type>(<scope>): <subject>"; \
		exit 1; \
	fi

# Validate configuration files
validate-configs:
	@echo "Validating configuration files..."
	@$(PYTHON) -c "from knapsack_gnn.config import validate_config_file; \
		import sys; \
		from pathlib import Path; \
		configs = list(Path('configs').glob('**/*.yaml')); \
		failed = []; \
		for cfg in configs: \
			is_valid, msg = validate_config_file(cfg); \
			print(msg); \
			if not is_valid: failed.append(cfg); \
		sys.exit(1 if failed else 0)"
	@echo "✅ All configs valid"

# Download pre-trained checkpoint
RUN ?= run_20251020_104533
SOURCE ?= github
download-checkpoint:
	@echo "Downloading checkpoint: $(RUN) from $(SOURCE)"
	$(PYTHON) scripts/download_artifacts.py --checkpoint $(RUN) --source $(SOURCE)

# Generate datasets with specific seed
DATASET_CONFIG ?= configs/train_default.yaml
DATASET_SEED ?= 1337
generate-datasets:
	@echo "Generating datasets from $(DATASET_CONFIG) with seed $(DATASET_SEED)"
	$(PYTHON) scripts/generate_datasets.py --config $(DATASET_CONFIG) --seed $(DATASET_SEED)

# Verify reproducibility of checkpoint
TOLERANCE ?= 1e-6
verify-reproducibility:
	@if [ -z "$(CHECKPOINT_DIR)" ]; then \
		echo "No checkpoint directory specified. Set CHECKPOINT_DIR=<path>"; \
		exit 1; \
	fi
	@echo "Verifying reproducibility of $(CHECKPOINT_DIR) with tolerance $(TOLERANCE)"
	$(PYTHON) scripts/verify_reproducibility.py --checkpoint $(CHECKPOINT_DIR) --tolerance $(TOLERANCE)

train:
	@echo "Training Knapsack GNN (checkpoints saved under $(CHECKPOINT_ROOT))"
	@if [ "$(GENERATE_DATA)" = "1" ] || [ "$(GENERATE_DATA)" = "true" ] || [ "$(GENERATE_DATA)" = "yes" ]; then \
		GENERATE_FLAG="--generate_data"; \
	else \
		GENERATE_FLAG=""; \
	fi; \
	CMD="$(PYTHON) experiments/pipelines/train_pipeline.py $$GENERATE_FLAG --device $(DEVICE) --train_size $(TRAIN_SIZE) --val_size $(VAL_SIZE) --test_size $(TEST_SIZE) --n_items_min $(N_ITEMS_MIN) --n_items_max $(N_ITEMS_MAX) --num_epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --learning_rate $(LEARNING_RATE) --weight_decay $(WEIGHT_DECAY) --seed $(SEED) --data_dir $(DATA_DIR) --checkpoint_dir $(CHECKPOINT_ROOT) $(TRAIN_ARGS)"; \
	echo "$$CMD"; \
	eval "$$CMD"

evaluate:
	@if [ -z "$(CHECKPOINT_DIR)" ]; then \
		echo "No checkpoint directory detected. Set CHECKPOINT_DIR=<path to run_YYYYMMDD_HHMMSS> or train first."; \
		exit 1; \
	fi
	@if [ "$(TEST_ONLY)" = "1" ] || [ "$(TEST_ONLY)" = "true" ] || [ "$(TEST_ONLY)" = "yes" ]; then \
		TEST_FLAG="--test_only"; \
	else \
		TEST_FLAG=""; \
	fi; \
	if [ -n "$(THREADS)" ]; then \
		THREAD_FLAG="--threads $(THREADS)"; \
	else \
		THREAD_FLAG=""; \
	fi; \
	if [ -n "$(MAX_SAMPLES)" ]; then \
		MAX_FLAG="--max_samples $(MAX_SAMPLES)"; \
	else \
		MAX_FLAG=""; \
	fi; \
	if [ -n "$(MAX_HINT_ITEMS)" ]; then \
		HINT_FLAG="--max-hint-items $(MAX_HINT_ITEMS)"; \
	else \
		HINT_FLAG=""; \
	fi; \
	if [ "$(COMPILE)" = "1" ] || [ "$(COMPILE)" = "true" ] || [ "$(COMPILE)" = "yes" ]; then \
		COMPILE_FLAG="--compile"; \
	else \
		COMPILE_FLAG=""; \
	fi; \
	if [ "$(QUANTIZE)" = "1" ] || [ "$(QUANTIZE)" = "true" ] || [ "$(QUANTIZE)" = "yes" ]; then \
		QUANTIZE_FLAG="--quantize"; \
	else \
		QUANTIZE_FLAG=""; \
	fi; \
	if [ -n "$(ILP_THREADS)" ]; then \
		ILP_THREAD_FLAG="--ilp-threads $(ILP_THREADS)"; \
	else \
		ILP_THREAD_FLAG=""; \
	fi; \
	CMD="$(PYTHON) evaluate.py --checkpoint_dir $(CHECKPOINT_DIR) --data_dir $(DATA_DIR) --strategy $(STRATEGY) --n_samples $(N_SAMPLES) --temperature $(TEMPERATURE) --threshold $(THRESHOLD) --sampling_schedule $(SAMPLING_SCHEDULE) --sampling_tolerance $(SAMPLING_TOL) $$MAX_FLAG $$HINT_FLAG --lagrangian_iters $(LAGRANGIAN_ITERS) --lagrangian_tol $(LAGRANGIAN_TOL) --lagrangian_bias $(LAGRANGIAN_BIAS) --fix_threshold $(FIX_THRESHOLD) --ilp_time_limit $(ILP_TIME_LIMIT) --device $(DEVICE) $$TEST_FLAG $$THREAD_FLAG $$ILP_THREAD_FLAG $$COMPILE_FLAG $$QUANTIZE_FLAG $(EVAL_ARGS)"; \
	echo "$$CMD"; \
	eval "$$CMD"

ood:
	@if [ -z "$(CHECKPOINT_DIR)" ]; then \
		echo "No checkpoint directory detected. Set CHECKPOINT_DIR=<path to run_YYYYMMDD_HHMMSS> or train first."; \
		exit 1; \
	fi; \
	if [ -n "$(THREADS)" ]; then \
		THREAD_FLAG="--threads $(THREADS)"; \
	else \
		THREAD_FLAG=""; \
	fi; \
	if [ -n "$(MAX_SAMPLES)" ]; then \
		MAX_FLAG="--max_samples $(MAX_SAMPLES)"; \
	else \
		MAX_FLAG=""; \
	fi; \
	if [ -n "$(MAX_HINT_ITEMS)" ]; then \
		HINT_FLAG="--max-hint-items $(MAX_HINT_ITEMS)"; \
	else \
		HINT_FLAG=""; \
	fi; \
	if [ "$(COMPILE)" = "1" ] || [ "$(COMPILE)" = "true" ] || [ "$(COMPILE)" = "yes" ]; then \
		COMPILE_FLAG="--compile"; \
	else \
		COMPILE_FLAG=""; \
	fi; \
	if [ "$(QUANTIZE)" = "1" ] || [ "$(QUANTIZE)" = "true" ] || [ "$(QUANTIZE)" = "yes" ]; then \
		QUANTIZE_FLAG="--quantize"; \
	else \
		QUANTIZE_FLAG=""; \
	fi; \
	if [ -n "$(ILP_THREADS)" ]; then \
		ILP_THREAD_FLAG="--ilp-threads $(ILP_THREADS)"; \
	else \
		ILP_THREAD_FLAG=""; \
	fi; \
	CMD="$(PYTHON) experiments/pipelines/evaluate_ood_pipeline.py --checkpoint_dir $(CHECKPOINT_DIR) --data_dir $(DATA_DIR) --strategy $(STRATEGY) --n_samples $(N_SAMPLES) --temperature $(TEMPERATURE) --sampling_schedule $(SAMPLING_SCHEDULE) --sampling_tolerance $(SAMPLING_TOL) $$MAX_FLAG $$HINT_FLAG --lagrangian_iters $(LAGRANGIAN_ITERS) --lagrangian_tol $(LAGRANGIAN_TOL) --lagrangian_bias $(LAGRANGIAN_BIAS) --fix_threshold $(FIX_THRESHOLD) --ilp_time_limit $(ILP_TIME_LIMIT) --sizes $(OOD_SIZES) --n_instances_per_size $(OOD_INSTANCES) --device $(DEVICE) $$THREAD_FLAG $$ILP_THREAD_FLAG $$COMPILE_FLAG $$QUANTIZE_FLAG $(OOD_ARGS)"; \
	echo "$$CMD"; \
	eval "$$CMD"

demo:
	$(PYTHON) Knapsack_GNN.py

quick_train:
	$(MAKE) train EPOCHS=10 TRAIN_SIZE=500 VAL_SIZE=100 TEST_SIZE=100 GENERATE_DATA=$(GENERATE_DATA)

pipeline:
	@SKIP_FLAG=""; \
	if [ "$(SKIP_TRAIN)" = "1" ] || [ "$(SKIP_TRAIN)" = "true" ] || [ "$(SKIP_TRAIN)" = "yes" ]; then \
		SKIP_FLAG="--skip-train"; \
	fi; \
	if [ -n "$(THREADS)" ]; then \
		THREAD_FLAG="--threads $(THREADS)"; \
	else \
		THREAD_FLAG=""; \
	fi; \
	if [ -n "$(ILP_THREADS)" ]; then \
		ILP_THREAD_FLAG="--ilp-threads $(ILP_THREADS)"; \
	else \
		ILP_THREAD_FLAG=""; \
	fi; \
	if [ -n "$(MAX_HINT_ITEMS)" ]; then \
		HINT_FLAG="--max-hint-items $(MAX_HINT_ITEMS)"; \
	else \
		HINT_FLAG=""; \
	fi; \
	if [ -n "$(MAX_SAMPLES)" ]; then \
		MAX_FLAG="--max-samples $(MAX_SAMPLES)"; \
	else \
		MAX_FLAG=""; \
	fi; \
	if [ -n "$(CHECKPOINT_DIR)" ]; then \
		CKPT_FLAG="--checkpoint-dir $(CHECKPOINT_DIR)"; \
	else \
		CKPT_FLAG=""; \
	fi; \
	if [ "$(GENERATE_DATA)" = "1" ] || [ "$(GENERATE_DATA)" = "true" ] || [ "$(GENERATE_DATA)" = "yes" ]; then \
		GEN_FLAG="--generate-data"; \
	else \
		GEN_FLAG=""; \
	fi; \
	CMD="$(PYTHON) experiments/pipelines/main.py full --data-dir $(DATA_DIR) --checkpoint-root $(CHECKPOINT_ROOT) --device $(DEVICE) $$SKIP_FLAG $$CKPT_FLAG $$GEN_FLAG --train-size $(TRAIN_SIZE) --val-size $(VAL_SIZE) --test-size $(TEST_SIZE) --n-items-min $(N_ITEMS_MIN) --n-items-max $(N_ITEMS_MAX) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE) --learning-rate $(LEARNING_RATE) --weight-decay $(WEIGHT_DECAY) --hidden-dim $(HIDDEN_DIM) --num-layers $(NUM_LAYERS) --dropout $(DROPOUT) --strategies $(PIPELINE_STRATEGIES) --n-samples $(N_SAMPLES) --temperature $(TEMPERATURE) --sampling-schedule $(subst $(comma),$(space),$(SAMPLING_SCHEDULE)) --sampling-tolerance $(SAMPLING_TOL) $$MAX_FLAG --lagrangian-iters $(LAGRANGIAN_ITERS) --lagrangian-tol $(LAGRANGIAN_TOL) --lagrangian-bias $(LAGRANGIAN_BIAS) --fix-threshold $(FIX_THRESHOLD) --ilp-time-limit $(ILP_TIME_LIMIT) $$HINT_FLAG $$THREAD_FLAG $$ILP_THREAD_FLAG $$([ "$(COMPILE)" = "1" ] || [ "$(COMPILE)" = "true" ] || [ "$(COMPILE)" = "yes" ] && echo "--compile") $$([ "$(QUANTIZE)" = "1" ] || [ "$(QUANTIZE)" = "true" ] || [ "$(QUANTIZE)" = "yes" ] && echo "--quantize")"; \
	echo "$$CMD"; \
	eval "$$CMD"

clean:
	@echo "Cleaning build artifacts and caches..."
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info site/ 2>/dev/null || true
	@echo "✅ Cleanup complete"

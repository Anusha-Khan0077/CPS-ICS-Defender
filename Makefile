# CPS/ICS Defender — Developer Makefile
# Usage: make <target>

PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
PYTEST  ?= $(PYTHON) -m pytest
SRC     := src/cps_defender
TESTS   := tests

.PHONY: help install install-dev test test-cov lint clean \
        generate train-detector train-rl demo zip

# ── Default target ─────────────────────────────────────────────────
help:
	@echo ""
	@echo "  CPS/ICS Defender — available targets"
	@echo "  ─────────────────────────────────────"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install dev + test dependencies"
	@echo "  test            Run the full test suite"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  generate        Generate a synthetic dataset (5 000 flows)"
	@echo "  train-detector  Train the ML IDS detector"
	@echo "  train-rl        Train the DQN mitigation agent (500 ep)"
	@echo "  demo            Run the end-to-end demo"
	@echo "  clean           Remove build artefacts and caches"
	@echo "  zip             Package the project for distribution"
	@echo ""

# ── Install ────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ── Tests ──────────────────────────────────────────────────────────
test:
	$(PYTEST) $(TESTS) -v

test-cov:
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report: htmlcov/index.html"

# Individual test modules
test-ids:
	$(PYTEST) $(TESTS)/test_ids.py -v

test-sdn-rl:
	$(PYTEST) $(TESTS)/test_sdn_rl.py -v

test-integration:
	$(PYTEST) $(TESTS)/test_integration.py -v

# ── Workflow ───────────────────────────────────────────────────────
generate:
	$(PYTHON) scripts/generate_dataset.py \
	    --n-flows 5000 \
	    --output data/processed/dataset \
	    --format both \
	    --augment

train-detector:
	$(PYTHON) scripts/train_detector.py \
	    --dataset data/processed/dataset.npz \
	    --model-out data/models/ml_detector.joblib \
	    --report \
	    --n-estimators 200

train-rl:
	$(PYTHON) scripts/train_rl.py \
	    --episodes 500 \
	    --save data/models/dqn_agent.npz

train-rl-curriculum:
	$(PYTHON) scripts/train_rl.py \
	    --curriculum \
	    --episodes 500 \
	    --save data/models/dqn_agent_curriculum.npz

serve:
	$(PYTHON) scripts/serve.py

serve-auto:
	$(PYTHON) scripts/serve.py --autostart --flow-delay 0.08

serve-trained:
	$(PYTHON) scripts/serve.py \\
	    --ml-model data/models/ids_pipeline.joblib \\
	    --rl-agent data/models/dqn_agent.npz \\
	    --autostart

demo:
	$(PYTHON) scripts/demo.py --n-flows 200

demo-quick:
	$(PYTHON) scripts/demo.py --n-flows 100

# Full pipeline: generate → train detector → train RL → demo
pipeline: generate train-detector train-rl demo

# ── Clean ──────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"      -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "Clean."

# ── Distribution ───────────────────────────────────────────────────
zip:
	@VERSION=$$($(PYTHON) -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['version'])") && \
	 git archive --format=zip --prefix=cps-ics-defender-$$VERSION/ \
	    -o cps-ics-defender-$$VERSION.zip HEAD 2>/dev/null || \
	 zip -r cps-ics-defender.zip . \
	    --exclude "*.pyc" --exclude "__pycache__/*" \
	    --exclude ".git/*" --exclude "data/models/*" \
	    --exclude "htmlcov/*" --exclude "*.egg-info/*" && \
	 echo "Archive: cps-ics-defender.zip"

---
title: SafeGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
---

# SafeGuard AI - Advanced Fraud Detection System

A machine learning-powered fraud detection application using DistilBERT that analyzes messages in real-time to identify potentially fraudulent content. Features a beautiful Flask web interface, comprehensive test suite, and automated CI/CD pipeline.

## 🎯 Features

- **AI-Powered Detection**: Uses DistilBERT transformer model for accurate fraud classification
- **Keyword Analysis**: Supplementary fraud keyword detection for enhanced accuracy
- **Real-time Prediction**: Instant message analysis through web interface
- **Beautiful UI**: Modern, responsive design with gradient styling
- **Enter Key Support**: Quick message submission with Enter key
- **History Management**: Track and clear all previous analyses
- **MLOps Pipeline**: Full data ingestion → training → evaluation pipeline
- **Automated Testing**: Comprehensive pytest suite with 15+ test cases
- **CI/CD Integration**: GitHub Actions workflows for automatic testing and deployment

---

## 📦 Project Structure

```
SafeGuard.AI/
├── app.py                          # Flask web application
├── main.py                         # Main entry point
├── params.yaml                     # Configuration parameters
├── pyproject.toml                  # Project dependencies (UV)
├── dvc.yaml                        # DVC pipeline configuration
├── metrics.json                    # Model evaluation metrics
│
├── src/                            # ML Pipeline stages
│   ├── stage_01_get_data.py        # Data ingestion
│   ├── stage_02_train.py           # Model training
│   └── stage_03_evaluate.py        # Model evaluation
│
├── data/
│   └── raw_data/
│       └── fraud_dataset.csv       # Input fraud dataset
│
├── models/
│   └── fraud_model_final/          # Trained DistilBERT model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── tokenizer.json
│
├── templates/
│   └── index.html                  # Web UI template
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_app.py                 # Flask app tests
│   ├── test_data_ingestion.py      # Pipeline stage 01 tests
│   ├── test_training.py            # Pipeline stage 02 tests
│   ├── test_evaluation.py          # Pipeline stage 03 tests
│   └── test_main.py                # Main module tests
│
└── .github/
    └── workflows/
        ├── tests.yml               # Run pytest tests
        ├── code-quality.yml        # Code quality checks
        ├── build.yml               # Build and deploy
        └── ci-cd.yaml              # Full MLOps pipeline
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (https://astral.sh/uv/)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/SafeGuard.AI.git
   cd SafeGuard.AI
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Run the web application**
   ```bash
   uv run python app.py
   ```

4. **Access the UI**
   - Open browser: `http://localhost:5000`
   - Enter a message to analyze
   - Press Enter or click "Analyze Message"

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_app.py

# Run with coverage report
uv run pytest --cov=src --cov-report=html
```

**Test Coverage:**
- ✅ Flask app routes and functionality (5 tests)
- ✅ Data ingestion pipeline (4 tests)
- ✅ Model training pipeline (3 tests)
- ✅ Model evaluation (2 tests)
- ✅ Main module (1 test)

---

## 🔄 ML Pipeline

### Stage 01: Data Ingestion
```bash
uv run python src/stage_01_get_data.py
```
- Reads configuration from `params.yaml`
- Copies raw data to `data/raw_data/dataset.parquet`
- Creates necessary directories
- Error handling for missing files

### Stage 02: Model Training
```bash
uv run python src/stage_02_train.py
```
- Trains DistilBERT model on fraud dataset
- Splits data: 80% train, 20% test
- Saves model to `models/fraud_model_final/`
- Cleans up intermediate checkpoints
- Training config in `params.yaml`:
  - Model: distilbert-base-uncased
  - Batch size: 4
  - Epochs: 3
  - Learning rate: 5e-5

### Stage 03: Evaluation
```bash
uv run python src/stage_03_evaluate.py
```
- Evaluates trained model
- Generates metrics (accuracy, F1, ROC-AUC)
- Saves results to `metrics.json`

---

## 🤖 How It Works

### Fraud Detection Logic

1. **AI Model Classification** (Primary)
   - DistilBERT transformer analyzes message semantics
   - Returns: FRAUD (confidence > 0.5) or LEGIT

2. **Keyword Analysis** (Supplementary)
   - If model says LEGIT, checks against fraud keyword list
   - If suspicious keywords found → WARNING
   - Otherwise → LEGIT

### Fraud Keywords Detected
- Urgency: "immediately", "24 hours", "asap", "tonight"
- Verification: "click here", "update kyc", "verify now", "log in"
- Financial Schemes: "lottery", "cashback", "winner", "work from home"
- Legal Threats: "police", "arrest", "fir", "court case", "warrant"
- Technical Alerts: "unusual login", "device detected", "password changed"
- Financial: "wire transfer", "refund", "bitcoin", "wallet"

---

## ⚙️ Configuration

Edit `params.yaml` to customize:

```yaml
data_source:
  local_path: data/raw_data/fraud_dataset.csv
  raw_data_dir: data/raw_data
  dataset_name: fraud_dataset.csv

train:
  model_name: distilbert-base-uncased
  batch_size: 4
  epochs: 3
  learning_rate: 5e-5
  save_steps: 500
```

---

## 🔧 GitHub Actions Workflows

Automated workflows run on every push:

### 1. tests.yml - Run Tests
- Runs: `uv run pytest tests/ -v`
- Triggers: Push to `master`/`main`/`develop`
- Status: All tests must pass

### 2. code-quality.yml - Code Quality
- Python syntax validation
- Flake8 linting
- Triggers: Push to `master`/`main`/`develop`

### 3. build.yml - Build & Deploy
- Full test execution
- Creates build artifacts
- Uploads to release artifacts
- Triggers: Push to `master`/`main` or version tags

### 4. ci-cd.yaml - MLOps Pipeline
- Runs full ML pipeline
- Tests → Data ingestion → Training → Evaluation
- Uploads metrics and models
- Triggers: Push to `master`

**View Results**: Go to GitHub repo → Actions tab

---

## 📝 API Routes

### GET / 
- Display analysis page
- Shows message history

### POST /
- Analyze message
- Returns: Message analysis with status

### POST /clear_history
- Clears all history
- Redirects to home

---

## 📊 Model Details

- **Model**: DistilBERT (6 layers, 66M parameters)
- **Dataset**: Fraud SMS/Messages dataset
- **Training**: 3 epochs, batch size 4
- **Device**: Auto-detects GPU/CPU
- **Output**: Binary classification (FRAUD/LEGIT)

---

## ✅ Recent Fixes & Improvements

### Critical Issues Fixed
- ✅ Added missing `redirect` import to Flask app
- ✅ Added `local_path` configuration to params.yaml
- ✅ Comprehensive error handling in all pipeline stages
- ✅ Try-except blocks for file operations
- ✅ Informative error messages

### Features Added
- ✅ Enter key support for message submission
- ✅ Clear all history button with confirmation
- ✅ Beautiful gradient UI with animations
- ✅ Comprehensive test suite (15+ tests)
- ✅ GitHub Actions CI/CD pipelines
- ✅ MLOps pipeline orchestration

---

## 📋 Dependencies

Core dependencies (managed by UV):
- **flask** - Web framework
- **torch** - Deep learning
- **transformers** - DistilBERT model
- **pandas** - Data processing
- **scikit-learn** - ML utilities
- **pytest** - Testing framework
- **pyyaml** - Configuration

---

## 🐛 Known Issues & Workarounds

### Stage 03 Evaluation
- **Issue**: Data format mismatch between stages
- **Status**: Works with proper dataset format
- **Note**: May need adjustment if dataset columns change

---

## 🔐 Security Notes

- Model runs in `eval()` mode (no training during inference)
- No data is stored permanently (history is in-memory)
- HTTPS should be configured for production
- API calls require proper input validation

---

## 🚦 Status

- ✅ Development: Complete
- ✅ Testing: Comprehensive (15+ tests)
- ✅ CI/CD: Active
- ✅ Ready for: Production deployment

---

## 📞 Support

For issues or questions:
1. Check GitHub Issues
2. Review CODE_REVIEW.md for known issues
3. Check test failures in GitHub Actions
4. Review error messages in logs

---

## 📄 License

This project is open source.

---

## 🎓 Learning Resources

- **Flask**: https://flask.palletsprojects.com/
- **Transformers**: https://huggingface.co/docs/transformers/
- **PyTorch**: https://pytorch.org/docs/
- **Pytest**: https://docs.pytest.org/
- **DVC**: https://dvc.org/doc

---

## 🎉 Credits

Built with ❤️ for fraud detection using modern ML practices and best practices.

**Last Updated**: May 2026
**Version**: 1.0.0

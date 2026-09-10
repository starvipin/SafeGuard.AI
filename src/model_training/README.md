# Model training: Steps 01–04

Read the files in numerical order. Run commands from the project root with `python -m`.

| File | What it does | Default input → output |
| --- | --- | --- |
| `step_01_prepare_data.py` | Copy the configured source, converting CSV/Parquet when requested | `data/raw_data/fraud_dataset.csv` → `data/raw_data/dataset.csv` |
| `step_02_train_model.py` | Validate → split 80/20 → tokenize → train DistilBERT → calculate metrics → save model/tokenizer → clean checkpoints | Dataset + `params.yaml` → `models/fraud_model_final/`, `metrics.json` |
| `step_03_evaluate_model.py` | Evaluate the saved model on the same seeded hold-out split | Dataset + saved model → updated `metrics.json` |
| `step_04_upload_to_hf.py` | Create or reuse an HF model repository and upload the local model folder | Trained model + `HF_TOKEN` → updated HF model repository |

Actual paths come from `params.yaml`. `pipeline_helpers.py` handles shared configuration, validation, and metrics. Step 01 copies or converts data; text/label validation takes place during training and evaluation.

## Run order

### Cell-by-cell notebooks

The companion notebooks are not present in the current checkout. The instructions below are a reference for restoring them; use the terminal commands further down to run the available scripts. The existing notebook tests require both `.ipynb` files and will fail while they are absent.

- `step_02_train_model.ipynb`: settings → dataset preview → split → tokenization → batches → base model/optimizer → training → metrics → final save.
- `step_03_evaluate_model.ipynb`: saved model → test batches → predictions → metrics → inspect mistakes → save metrics.

If the notebooks are restored, launch the notebook interface from the project root:

```bash
uv run python -m jupyter lab
```

Open a notebook in `src/model_training/`, select the project's `.venv` Python kernel, and run cells from top to bottom with **Shift+Enter**. The first cell prints the Python executable so you can verify the environment. In VS Code, use **Select Kernel** to choose the same environment.

Prepare the dataset with Step 01 first. After changing the learning rate, **restart the kernel** and run cells in order for fresh training. Rerunning only the training-loop cell continues updating the current weights. Intermediate checkpoints may be saved during training, but the final model is not written until the final save cell. Shut down the training kernel before opening evaluation if you need to free GPU memory.

Notebooks are **alternatives** to running each complete `.py` stage, but they import helpers from those scripts. **Keep the Step 02 and Step 03 Python files.** You do not need to run both the notebook and script for the same training session. DVC and CI use the existing scripts. HF publishing remains an explicit Step 04 operation. Clear notebook outputs before sharing or committing them because previews may contain dataset text.

### Terminal commands

```bash
uv run python -m src.model_training.step_01_prepare_data
uv run python -m src.model_training.step_02_train_model
uv run python -m src.model_training.step_03_evaluate_model
```

The CSV needs `text` and `label` columns (`0` = legitimate, `1` = fraud). Provide enough examples in each class for a stratified 80/20 split. Blank and null text rows are removed.

`metrics.json` contains accuracy, F1, and ROC AUC. Step 03 re-evaluates the same held-out split used for training metrics; it is not an independent unseen benchmark.

Step 02 overwrites the configured local model. Use a separate output directory for experiments if a live process is using the existing local model.

## Step 04: explicit upload

Inspect the metrics and saved files, then set `HF_TOKEN` in the environment or `.env`. Optionally set `HF_MODEL_REPO`; the default is `sainivipin/fraud-model-final`.

```bash
uv run python -m src.model_training.step_04_upload_to_hf
```

This command uploads `train.model_output_dir` from `params.yaml` and updates the remote repository. A token does not cause Step 02 to upload automatically. Python callers can explicitly invoke `upload_model(model_dir=..., repository=..., token=...)`.

## Connection to the website

```text
CSV → Step 01 → dataset.csv → Step 02 → local model
                                         ↓
                               Step 03 → metrics.json
                                         ↓ manually run Step 04
                                HF model repository
                                         ↓ if the local model is missing
                               src/web_app/fraud_detector.py → prediction
```

The website reuses its existing local model/cache; it does not download the latest HF revision on every request. Model publishing and website deployment are separate operations.

With DVC installed, `dvc repro` runs Steps 01–03 and does not upload. Configuration and artifacts retain their root locations: `params.yaml`, `data/`, `models/`, and `metrics.json`.

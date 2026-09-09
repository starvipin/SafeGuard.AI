# Model training: 01 se 04 tak

Files ko number ke order mein padho. Commands project root se `python -m` ke saath chalti hain.

| File | Andar kya hota hai? | Default input → output |
| --- | --- | --- |
| `step_01_prepare_data.py` | Config se source padhta hai, copy karta hai; CSV/Parquet conversion supported | `data/raw_data/fraud_dataset.csv` → `data/raw_data/dataset.csv` |
| `step_02_train_model.py` | Data validate → 80/20 split → tokenize → DistilBERT train → metrics → model/tokenizer save → checkpoints clean | Dataset + `params.yaml` → `models/fraud_model_final/`, `metrics.json` |
| `step_03_evaluate_model.py` | Saved model ko same seed/split ke test portion par evaluate karta hai | Dataset + saved model → updated `metrics.json` |
| `step_04_upload_to_hf.py` | HF model repo create/reuse karke local model folder upload karta hai | Trained model + `HF_TOKEN` → HF model repository updated |

Actual paths `params.yaml` se aate hain. `pipeline_helpers.py` config reading, dataset validation aur metrics ka shared code hai. Step 01 copy/conversion karta hai; text/label validation training/evaluation mein hoti hai.

## Run order

```bash
uv run python -m src.model_training.step_01_prepare_data
uv run python -m src.model_training.step_02_train_model
uv run python -m src.model_training.step_03_evaluate_model
```

CSV mein `text` aur `label` columns chahiye (`0` = legit, `1` = fraud). Dono classes mein enough examples rakho taaki stratified 80/20 split ban sake. Blank/null text rows drop hote hain.

`metrics.json` mein accuracy, F1 aur ROC AUC milte hain. Step 03 separate unseen benchmark nahi hai: training ke metrics wala same held-out split dobara evaluate hota hai.

Step 02 configured local model overwrite karta hai. Live process jis local folder ko use kar raha ho us par training chalane se pehle separate output path configure karo.

## Step 04: upload alag se

Metrics aur saved model check karne ke baad `.env` ya environment mein `HF_TOKEN` set karo. Optional `HF_MODEL_REPO` destination choose karta hai; default `sainivipin/fraud-model-final` hai.

```bash
uv run python -m src.model_training.step_04_upload_to_hf
```

Yeh command `params.yaml` ka `train.model_output_dir` upload karti hai. Remote repo update hota hai. Token present hone se step 02 automatic upload nahi karta. Python se `upload_model(model_dir=..., repository=..., token=...)` explicitly call kar sakte ho.

## Website se connection

```text
CSV → step 01 → dataset.csv → step 02 → local model
                                           ↓
                                 step 03 → metrics.json
                                           ↓ manually run step 04
                                  HF model repository
                                           ↓ if local model missing
                                 src/web_app/fraud_detector.py → prediction
```

Website existing local model/cache use karti hai; har request par latest HF model download nahi hota. Model upload aur website deployment separate operations hain.

`dvc repro` (DVC installed ho to) steps 01–03 chalata hai; upload uska part nahi hai. Root `params.yaml`, `data/`, `models/`, `metrics.json` paths preserve kiye gaye hain.

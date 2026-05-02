# Team 05 Rockets - BKBC ATI Prediction

## Team

**Team name:** Rockets

**Team members:** Adi Almukhamet, Caitlin Eliza Yu, Kian Hong Tan, Samuel Jacob, Uyen Chu

## Project Summary

This project was developed for the Boston Kidney Biopsy Cohort (BKBC) challenge. The task is to predict **acute tubular injury (ATI)** from plasma proteomics and clinical covariates as a binary classification problem.

| Label | Meaning |
|---|---|
| `0` | No ATI |
| `1` | ATI present |

The provided training data contains one row per patient, with anonymized SomaScan plasma protein abundance features plus clinical variables. The model is evaluated on an external held-out cohort using **log loss**, so the primary output is a calibrated probability of ATI rather than only a hard class label.

Our final submission model is a **Lasso Logistic Regression** classifier. We chose this model because the dataset is high-dimensional relative to the number of patients: there are thousands of proteomic features and only hundreds of samples. L1 regularization encourages a sparse model, reducing overfitting risk while still allowing the model to identify informative protein signals.

The original starter-code README has been preserved as [old.md](old.md). It remains useful context for the task description, starter-code structure, data schema, and original XGBoost-based workflow.

## Data

The official training data path is:

```bash
/projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv
```

The expected columns are:

| Column group | Description |
|---|---|
| `sample_id` | Anonymized patient identifier |
| `ati` | Binary ATI label, where `0` = no ATI and `1` = ATI |
| `age` | Age, provided as a 10-year bin midpoint |
| `sex` | Sex, where `1` = male and `2` = female |
| `baseline_egfr_23` | Baseline eGFR |
| `feature_XXXX` | Log2-normalized, ComBat-corrected SomaScan plasma protein abundances |

The submitted model uses all available `feature_` protein columns plus the three clinical covariates: `age`, `sex`, and `baseline_egfr_23`.

## Repository Contents

| File or directory | Description |
|---|---|
| `README.md` | Submission README with setup, file descriptions, and run commands |
| `old.md` | Preserved original README/starter-code documentation |
| `requirements.txt` | Python package requirements |
| `predict.sh` | Main prediction entry point for submission evaluation |
| `predict.py` | Active inference script; loads the final Lasso LR checkpoint and writes predictions |
| `preprocess.py` | Data loading and feature/label construction helpers |
| `model.py` | Shared model definitions, constants, and model factory |
| `evaluate.py` | Stratified k-fold cross-validation script for model comparison |
| `train.py` | Original XGBoost-oriented training script from the starter-code workflow |
| `predict_orig.py` | Original XGBoost-oriented prediction script |
| `train2.py` | Extended training script that supports XGBoost, Lasso LR, Elastic Net, and TopK XGBoost |
| `predict2.py` | Extended prediction-script spinoff for multiple model types; not the active submission path |
| `weights/` | Saved model checkpoints, feature lists, metadata, and feature-importance outputs |
| `predictions.csv` | Example prediction output generated on data with labels |

## Final Model Checkpoint

The final submitted model is stored in:

```bash
weights/lasso_lr_model.pkl
weights/lasso_lr_feature_cols.json
```

`lasso_lr_model.pkl` is a serialized sklearn pipeline containing:

1. `StandardScaler`
2. `LogisticRegression` with L1 regularization

The matching feature list is stored in `lasso_lr_feature_cols.json` and is used by `predict.py` to align incoming CSV columns to the exact order used during training.

## Setup Instructions

On the hackathon environment, load the required modules:

```bash
module load medaihack/spring-2026
module load python3/3.12.4
```

Create and activate a virtual environment:

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For later sessions, reactivate the environment with:

```bash
module load medaihack/spring-2026
module load python3/3.12.4
source .venv/bin/activate
```

## Commands to Run

### 1. Run prediction with the final submitted model

Use `predict.sh` as the main submission entry point:

```bash
bash predict.sh /path/to/input.csv
```

To specify an output file:

```bash
bash predict.sh /path/to/input.csv predictions.csv
```

The prediction script writes a CSV with:

| Column | Description |
|---|---|
| `sample_id` | Patient/sample identifier |
| `prob_ati` | Predicted probability of ATI |
| `pred_label` | Hard class label using a 0.5 threshold |
| `true_label` | Included only if the input CSV contains `ati` |

If the input CSV includes the `ati` column, `predict.py` also prints evaluation metrics including classification report, AUC, and log loss.

### 2. Example prediction command on the training file

```bash
bash predict.sh /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv predictions.csv
```

### 3. Evaluate models with cross-validation

```bash
python evaluate.py --data /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv
```

This evaluates the models defined in `model.py` using stratified k-fold cross-validation and saves per-fold metrics and confusion matrices to the output directory.

### 4. Re-train the final Lasso LR model

The original `train.py` is XGBoost-oriented. To train the final Lasso LR model, use `train2.py`:

```bash
python train2.py \
  --model-name "Lasso LR" \
  --data /projectnb/medaihack/BKBC-hackathon/BKBC_train/train.csv \
  --out weights
```

This writes:

```bash
weights/lasso_lr_model.pkl
weights/lasso_lr_feature_cols.json
weights/lasso_lr_metadata.json
weights/lasso_lr_feature_importance.csv
```

## Source Code and Notebooks

All source code for this submission is contained in the Python files in this directory. No notebook is required to run the submitted model.

The active submission path is:

```text
predict.sh
  -> predict.py
      -> weights/lasso_lr_model.pkl
      -> weights/lasso_lr_feature_cols.json
      -> predictions.csv
```

The model development path was:

```text
preprocess.py
  -> model.py
  -> evaluate.py
  -> train2.py
  -> weights/lasso_lr_model.pkl
```

## Example Output

Example rows from `predictions.csv`:

```csv
sample_id,prob_ati,pred_label,true_label
bkbc_0,0.977625166466342,1,1
bkbc_1,0.9926379638509928,1,1
bkbc_2,0.9797321695488687,1,1
bkbc_3,0.32906222972117904,0,0
```

## Notes

- The final submitted model is Lasso LR, not the original starter-code XGBoost model.
- `predict.sh` is the command intended for external evaluation.
- `old.md` is retained to document the original starter-code assumptions and task framing.
- `train2.py` and `predict2.py` were created as compatibility spinoffs while experimenting with non-XGBoost sklearn models.

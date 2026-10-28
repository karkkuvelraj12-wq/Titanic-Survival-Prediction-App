# Project 05 — Streamlit Interactive ML Web App

An interactive Streamlit app that loads a saved Scikit-learn Decision Tree and estimates Titanic survival probability from passenger class, sex, age, and fare.

## Project files

- `app.py` — interactive interface and prediction display
- `train_model.py` — trains a compatible decision tree and saves `model.joblib`
- `requirements.txt` — Python packages required by the app
- `model.joblib` — generated model artifact; create it before deploying

## Run locally

Use Python 3.10 or newer. From this folder, run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first model-training run downloads Seaborn's Titanic example dataset, so an internet connection may be required. Confirm that `model.joblib` was created in this folder before starting the app.

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `train_model.py`, `requirements.txt`, and the generated `model.joblib` to a public GitHub repository.
2. Open Streamlit Community Cloud at https://streamlit.io/cloud and connect the repository.
3. Select `app.py` as the app entry point and deploy.
4. Open the public app URL on a phone and test multiple inputs.
5. Keep `model.joblib` in the repository root beside `app.py`.

## Feature compatibility

The app expects the exact feature order `pclass`, `sex`, `age`, `fare`. Sex is encoded as `1 = female` and `0 = male`. The training script uses the same feature order and encoding.

## Limitations

This is an educational model based on a historical dataset. Predictions describe patterns in that dataset, are not causal conclusions, and must not be used for real-world safety decisions. The first training run may need internet access to fetch the dataset.

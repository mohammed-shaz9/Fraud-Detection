# Fraud Detection using AI (Python + Django)

A beginner-friendly project to detect credit card fraud using pandas, scikit-learn, matplotlib, seaborn, and Django for deployment. No deep learning required.

## Problem Statement
Build a binary classifier that predicts whether a credit card transaction is fraudulent.

## Dataset
Use the well-known "Credit Card Fraud Detection" dataset (European card transactions, anonymized). Download `creditcard.csv` from Kaggle and place it under `data/creditcard.csv`.

## Project Structure
```
AD/
├─ data/
│  └─ creditcard.csv (you add this)
├─ ml/
│  ├─ train_model.py
│  ├─ figures/
│  └─ metrics/
├─ fraud_detector/
│  ├─ manage.py
│  ├─ fraud_detector/ (Django project)
│  └─ predictor/ (Django app)
├─ fraud_model.pkl (generated after training)
├─ feature_medians.json (generated after training)
└─ requirements.txt
```

## Step 1 – Dataset & Preprocessing
- Load `creditcard.csv` with pandas
- Clean (drop duplicates, fill missing if any)
- Handle imbalance with undersampling (1:1) for simplicity
- Split with `train_test_split`

Run:
```bash
pip install -r requirements.txt
python ml/train_model.py
```
Outputs:
- `ml/figures/` with class distribution, heatmap, amount distribution
- `ml/metrics/model_results.csv`
- `fraud_model.pkl` (best model) and `feature_medians.json`

## Step 2 – EDA
The script prints the dataset shape, missing values, and saves plots to `ml/figures/`.

## Step 3 – Model Training
Trains Logistic Regression and Random Forest. Compares accuracy, precision, recall, F1, and ROC-AUC. Saves the best model.

## Step 4 – Django Deployment
1. Move into the Django project folder
```bash
cd fraud_detector
```
2. Run migrations and start server
```bash
python manage.py migrate
python manage.py runserver
```
3. Open `http://127.0.0.1:8000/` in your browser.

Enter `Time` and `Amount`. Optionally, provide `V1..V28`. Missing ones use dataset medians stored in `feature_medians.json`.

## Step 5 – Documentation
- This README explains the problem, dataset, steps, and how to run.
- See `report.md` for a short 2–3 page report.

## Notes
- Uses SQLite (default) for Django; no extra DB setup needed.
- If you change the features or preprocessing, retrain and regenerate `fraud_model.pkl` and `feature_medians.json`. 

## JSON API
You can also call a JSON API endpoint for programmatic predictions.

- Endpoint: `/api/predict/` (POST)
- Content-Type: `application/json` or form-encoded

Example:
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 10000,
    "Amount": 123.45
  }'
```

Response fields:
- `prediction`: 0 or 1
- `is_fraud`: boolean
- `probability`: float (if the model supports probabilities)
- `defaulted_fields`: list of input keys defaulted to medians
- `clipped_fields`: list of input keys clipped to the training range 
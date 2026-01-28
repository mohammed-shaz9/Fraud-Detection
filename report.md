# Credit Card Fraud Detection – Short Report

## 1. Problem Overview
Credit card fraud is rare but costly. The goal is to identify fraudulent transactions (Class=1) from legitimate ones (Class=0). The dataset is highly imbalanced, so evaluation must focus on recall and precision in addition to accuracy.

## 2. Dataset
- Source: Credit Card Fraud Detection dataset (Kaggle)
- Records: ~284,807 transactions
- Positive class: ~0.172% (fraud)
- Features: PCA-transformed components `V1..V28`, plus `Time` and `Amount`
- Target: `Class` (0 = genuine, 1 = fraud)

## 3. Preprocessing
- Loaded with pandas, dropped duplicates
- Filled missing values with median if any (dataset typically has none)
- Addressed class imbalance with simple undersampling (1:1 ratio) to make training straightforward for beginners.
- Train/test split: 80/20 stratified

## 4. Exploratory Data Analysis (EDA)
- Class distribution plot confirms heavy imbalance
- Correlation heatmap on numeric features
- Amount distribution by class (fraud vs genuine)
- Plots saved under `ml/figures/`

## 5. Models and Training
- Logistic Regression (max_iter=1000)
- Random Forest (200 trees)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC on test set
- We choose the best model primarily by ROC-AUC, then F1 and Recall

## 6. Results
- Training script prints a comparison table and saves it to `ml/metrics/model_results.csv`
- The best model is persisted to `fraud_model.pkl`
- Median feature values used for default inference are saved to `feature_medians.json`

Note: Absolute scores depend on random seed and sampling. Random Forest often offers higher ROC-AUC and Recall in this setup.

## 7. Deployment (Django)
- Simple web interface at `/` accepts `Time` and `Amount`; `V1..V28` are optional
- Backend loads `fraud_model.pkl` and `feature_medians.json`
- Prediction page displays either “✅ Transaction is Genuine” or “❌ Fraud Detected!” and, if available, the predicted probability
- Uses SQLite by default

## 8. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python ml/train_model.py`
3. Start web app:
   - `cd fraud_detector`
   - `python manage.py migrate`
   - `python manage.py runserver`
4. Open `http://127.0.0.1:8000/`

## 9. Limitations and Future Work
- Undersampling discards information; try SMOTE or class weights
- Calibrate probabilities (e.g., Platt scaling) for better thresholding
- Add threshold tuning for business-driven precision/recall trade-offs
- Track model and data versions; add tests and CI
- Add authentication/rate-limiting in production

## 10. Conclusion
This project demonstrates an end-to-end, beginner-friendly fraud detection workflow, from data loading and EDA to model training, evaluation, and a minimal Django deployment. 
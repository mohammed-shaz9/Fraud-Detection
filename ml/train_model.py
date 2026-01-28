import os
import json
import datetime
import hashlib
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score, 
	roc_auc_score, confusion_matrix, classification_report,
	precision_recall_curve
)
import joblib

# ===============================
# Production-Grade Fraud Training
# ===============================

RANDOM_SEED = 42
DATA_DIR = Path("data")
FIGURES_DIR = Path("ml") / "figures"
METRICS_DIR = Path("ml") / "metrics"
MODEL_DIR = Path("models")
DATA_PATH = DATA_DIR / "creditcard.csv"

# Ensure directories exist
for d in [FIGURES_DIR, METRICS_DIR, DATA_DIR, MODEL_DIR]:
	d.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

def generate_synthetic_dataset(n_samples: int = 60000, fraud_ratio: float = 0.006) -> pd.DataFrame:
	"""Generate a synthetic dataset for demonstration if real data is missing."""
	rng = np.random.default_rng(RANDOM_SEED)
	num_fraud = max(1, int(n_samples * fraud_ratio))
	num_genuine = n_samples - num_fraud

	time_genuine = rng.uniform(0, 172800, size=num_genuine)
	time_fraud = rng.uniform(0, 172800, size=num_fraud)

	def gen_components(size: int):
		return {f"V{i}": rng.normal(0, 1.0, size=size) for i in range(1, 29)}

	comps_gen = gen_components(num_genuine)
	comps_fraud = gen_components(num_fraud)

	amount_genuine = rng.gamma(shape=2.0, scale=20.0, size=num_genuine)
	amount_fraud = rng.gamma(shape=4.0, scale=80.0, size=num_fraud)

	# Signal for fraud
	for key in ["V3", "V7", "V10", "V14"]:
		comps_fraud[key] = comps_fraud[key] + rng.normal(2.0, 0.5, size=num_fraud)

	df_gen = pd.DataFrame({"Time": time_genuine, **comps_gen, "Amount": amount_genuine, "Class": 0})
	df_fraud = pd.DataFrame({"Time": time_fraud, **comps_fraud, "Amount": amount_fraud, "Class": 1})

	return pd.concat([df_gen, df_fraud], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

def calculate_feature_stats(df: pd.DataFrame):
	"""Calculate stats ONLY on training data to prevent leakage."""
	stats = {}
	for col in df.columns:
		series = df[col]
		stats[col] = {
			"median": float(series.median()),
			"min": float(series.min()),
			"max": float(series.max()),
			"q1": float(series.quantile(0.25)),
			"q3": float(series.quantile(0.75)),
		}
	return stats

def evaluate_fraud_model(model_name, y_true, y_pred, y_proba):
	"""Production-grade evaluation highlighting business impact."""
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1 = f1_score(y_true, y_pred)
	f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
	auc = roc_auc_score(y_true, y_proba)
	
	# Business Cost Analysis
	false_alarm_cost = fp * 2    # $2 per false alarm (customer support cost)
	missed_fraud_cost = fn * 100 # $100 per missed fraud (direct loss)
	total_cost = false_alarm_cost + missed_fraud_cost
	
	print(f"\n=== {model_name} METRICS ===")
	print(f"True Positives (Fraud Caught): {tp:,}")
	print(f"False Positives (False Alarms): {fp:,}")
	print(f"False Negatives (Fraud Missed): {fn:,}")
	print(f"Precision: {precision:.2%}")
	print(f"Recall (Catch Rate): {recall:.2%}")
	print(f"F2-Score: {f2_score:.3f}")
	print(f"ROC-AUC: {auc:.4f}")
	print(f"💰 COST ANALYSIS: False Alarm=${false_alarm_cost:,.2f} | Missed=${missed_fraud_cost:,.2f} | Total=${total_cost:,.2f}")
	
	return {
		"model": model_name,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"f2": f2_score,
		"roc_auc": auc,
		"total_cost": total_cost,
		"tp": tp, "fp": fp, "fn": fn, "tn": tn
	}

def main():
	if not DATA_PATH.exists():
		df = generate_synthetic_dataset()
		df.to_csv(DATA_PATH, index=False)
	else:
		df = pd.read_csv(DATA_PATH).drop_duplicates()

	X = df.drop(columns=["Class"])
	y = df["Class"]

	# 1. Stratified Split BEFORE calculating stats (No Leakage)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
	)

	# 2. Calculate stats only on training data
	feature_stats = calculate_feature_stats(X_train)
	with open("feature_stats.json", "w") as f:
		json.dump(feature_stats, f)
	
	# Legacy support
	with open("feature_medians.json", "w") as f:
		json.dump({k: v["median"] for k, v in feature_stats.items()}, f)

	# 3. Training with Pipelines & Class Weights
	print(f"Training on {len(X_train)} samples with {y_train.sum()} fraud cases...")

	# Model 1: Balanced Logistic Regression
	lr_pipe = Pipeline([
		('scaler', StandardScaler()),
		('clf', LogisticRegression(class_weight={0: 1, 1: 100}, max_iter=1000, random_state=RANDOM_SEED))
	])
	lr_pipe.fit(X_train, y_train)

	# Model 2: Balanced Random Forest
	rf_pipe = Pipeline([
		('scaler', StandardScaler()),
		('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1))
	])
	rf_pipe.fit(X_train, y_train)

	# 4. Evaluation
	results = []
	for name, pipe in [("Logistic Regression", lr_pipe), ("Random Forest", rf_pipe)]:
		y_pred = pipe.predict(X_test)
		y_proba = pipe.predict_proba(X_test)[:, 1]
		results.append(evaluate_fraud_model(name, y_test, y_pred, y_proba))

	# 5. Model Versioning & Persistence
	best_result = min(results, key=lambda x: x['total_cost'])
	best_model_name = best_result['model']
	best_pipe = lr_pipe if best_model_name == "Logistic Regression" else rf_pipe
	
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	data_hash = hashlib.md5(X_train.values.tobytes()).hexdigest()[:8]
	version = f"v_{timestamp}_{data_hash}"
	
	model_payload = {
		'pipeline': best_pipe,
		'version': version,
		'feature_stats': feature_stats,
		'metrics': best_result,
		'trained_at': timestamp
	}
	
	# Save versioned and "latest"
	v_path = MODEL_DIR / f"fraud_model_{version}.pkl"
	joblib.dump(model_payload, v_path)
	joblib.dump(model_payload, "fraud_model.pkl") # Active model
	
	print(f"\n✅ Best model '{best_model_name}' saved as version {version}")

	# 6. Visualizations
	if True: # Plotting
		plt.figure(figsize=(10, 8))
		sns.heatmap(confusion_matrix(y_test, best_pipe.predict(X_test)), annot=True, fmt='d', cmap='Blues')
		plt.title(f"Confusion Matrix: {best_model_name}")
		plt.savefig(FIGURES_DIR / "confusion_matrix.png")
		plt.close()

if __name__ == "__main__":
	main()
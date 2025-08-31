import os
import json
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
try:
	import matplotlib.pyplot as plt
	import seaborn as sns
	PLOTTING_AVAILABLE = True
except Exception:
	plt = None  # type: ignore
	sns = None  # type: ignore
	PLOTTING_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


# ===============================
# Fraud Detection Training Script
# Steps 1-3: Preprocessing, EDA, Training & Evaluation
# ===============================
# This script is designed for beginners. It includes many comments to explain each step.
# It expects the dataset file at data/creditcard.csv (the well-known ULB credit card fraud dataset).
# If you don't have the dataset yet, download it and place it into the data/ folder before running this.


# -------------------------------
# Configuration and constants
# -------------------------------
RANDOM_SEED = 42  # For reproducibility of results
DATA_DIR = Path("data")
FIGURES_DIR = Path("ml") / "figures"
METRICS_DIR = Path("ml") / "metrics"
MODEL_PATH = Path("fraud_model.pkl")  # Saved best model file
FEATURE_MEDIANS_PATH = Path("feature_medians.json")  # Backward-compat: simple medians
FEATURE_STATS_PATH = Path("feature_stats.json")  # Rich stats for validation & UI
DATA_PATH = DATA_DIR / "creditcard.csv"

# Create necessary folders if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")


def print_header(title: str) -> None:
	"""Print a visual header to separate steps in console output."""
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80 + "\n")


def generate_synthetic_dataset(n_samples: int = 60000, fraud_ratio: float = 0.006) -> pd.DataFrame:
	"""Generate a synthetic dataset with the creditcard.csv schema for demo/training.

	- Highly imbalanced with small fraud_ratio
	- Adds weak signals to some components for fraud class
	"""
	rng = np.random.default_rng(RANDOM_SEED)
	num_fraud = max(1, int(n_samples * fraud_ratio))
	num_genuine = n_samples - num_fraud

	# Time in seconds (0 to two days)
	time_genuine = rng.uniform(0, 172800, size=num_genuine)
	time_fraud = rng.uniform(0, 172800, size=num_fraud)

	def gen_components(size: int):
		return {f"V{i}": rng.normal(0, 1.0, size=size) for i in range(1, 29)}

	comps_genuine = gen_components(num_genuine)
	comps_fraud = gen_components(num_fraud)

	# Amount distributions (fraud tends to be larger)
	amount_genuine = rng.gamma(shape=2.0, scale=20.0, size=num_genuine)
	amount_fraud = rng.gamma(shape=4.0, scale=80.0, size=num_fraud)

	# Introduce weak signal in a few components for fraud
	for key in ["V3", "V7", "V10", "V14"]:
		comps_fraud[key] = comps_fraud[key] + rng.normal(2.0, 0.5, size=num_fraud)

	df_genuine = pd.DataFrame({
		"Time": time_genuine,
		**{k: v for k, v in comps_genuine.items()},
		"Amount": amount_genuine,
		"Class": np.zeros(num_genuine, dtype=int),
	})
	df_fraud = pd.DataFrame({
		"Time": time_fraud,
		**{k: v for k, v in comps_fraud.items()},
		"Amount": amount_fraud,
		"Class": np.ones(num_fraud, dtype=int),
	})

	df = pd.concat([df_genuine, df_fraud], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
	return df


def load_dataset(csv_path: Path) -> pd.DataFrame:
	"""Load the dataset; if missing, generate a synthetic dataset and save it to data/."""
	if not csv_path.exists():
		print(f"Dataset not found at {csv_path}. Generating a synthetic dataset...")
		df_syn = generate_synthetic_dataset(n_samples=60000, fraud_ratio=0.006)
		csv_path.parent.mkdir(parents=True, exist_ok=True)
		df_syn.to_csv(csv_path, index=False)
		print(f"Synthetic dataset saved to {csv_path} with shape {df_syn.shape}.")
		return df_syn
	# Read CSV; this dataset typically has columns: Time, V1..V28, Amount, Class
	df = pd.read_csv(csv_path)
	return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
	"""Perform simple cleaning: drop exact duplicates, handle missing values."""
	# Drop duplicate rows if any
	before = len(df)
	df = df.drop_duplicates()
	after = len(df)
	if before != after:
		print(f"Removed {before - after} duplicate rows.")

	# Check and fill missing values with median for numeric columns (dataset usually has none)
	if df.isnull().sum().sum() > 0:
		for col in df.columns:
			if df[col].isnull().any():
				median_value = df[col].median()
				df[col] = df[col].fillna(median_value)
				print(f"Filled missing values in column '{col}' with median {median_value:.4f}.")
	return df


def perform_eda(df: pd.DataFrame) -> None:
	"""Create simple EDA outputs: shape, missing values, class balance, heatmap, and a few plots.
	Saves figures into ml/figures/.
	"""
	print_header("Step 2 – Exploratory Data Analysis (EDA)")
	print(f"Dataset shape: {df.shape}")
	print("\nMissing values per column:\n", df.isnull().sum())

	# Plot class distribution (fraud vs non-fraud)
	if PLOTTING_AVAILABLE:
		plt.figure(figsize=(6, 4))
		sns.countplot(x="Class", data=df, palette=["#4CAF50", "#F44336"])  # 0=Genuine, 1=Fraud
		plt.title("Transaction Class Distribution (0=Genuine, 1=Fraud)")
		plt.tight_layout()
		(class_distribution_path := FIGURES_DIR / "class_distribution.png")
		plt.savefig(class_distribution_path)
		plt.close()
		print(f"Saved class distribution plot to {class_distribution_path}")
	else:
		print("Plotting libraries not available; skipping class distribution plot.")

	# Correlation heatmap for a subset of features (full heatmap can be large)
	# We'll compute on numeric columns only
	if PLOTTING_AVAILABLE:
		numeric_df = df.select_dtypes(include=[np.number])
		corr = numeric_df.corr()
		plt.figure(figsize=(10, 8))
		sns.heatmap(corr, cmap="coolwarm", center=0, cbar=True)
		plt.title("Correlation Heatmap")
		plt.tight_layout()
		(heatmap_path := FIGURES_DIR / "correlation_heatmap.png")
		plt.savefig(heatmap_path)
		plt.close()
		print(f"Saved correlation heatmap to {heatmap_path}")
	else:
		print("Plotting libraries not available; skipping correlation heatmap.")

	# Amount distributions for fraud vs non-fraud
	if PLOTTING_AVAILABLE:
		plt.figure(figsize=(8, 4))
		sns.kdeplot(data=df[df["Class"] == 0], x="Amount", label="Genuine", fill=True)
		sns.kdeplot(data=df[df["Class"] == 1], x="Amount", label="Fraud", fill=True, color="#F44336")
		plt.legend()
		plt.title("Amount Distribution by Class")
		plt.tight_layout()
		(amount_dist_path := FIGURES_DIR / "amount_distribution.png")
		plt.savefig(amount_dist_path)
		plt.close()
		print(f"Saved amount distribution plot to {amount_dist_path}")
	else:
		print("Plotting libraries not available; skipping amount distribution plot.")


def undersample(df: pd.DataFrame, target_col: str = "Class", ratio: float = 1.0) -> pd.DataFrame:
	"""Perform simple random undersampling of the majority class.
	- ratio=1.0 will create a 1:1 dataset between minority and majority classes.
	"""
	print_header("Step 1 – Dataset & Preprocessing")
	# Separate fraud (1) and genuine (0)
	fraud_df = df[df[target_col] == 1]
	genuine_df = df[df[target_col] == 0]

	minority_count = len(fraud_df)
	majority_count = len(genuine_df)
	print(f"Original class counts -> Genuine: {majority_count}, Fraud: {minority_count}")

	# Number of majority samples to keep
	num_majority_keep = int(minority_count * ratio)
	num_majority_keep = min(num_majority_keep, majority_count)

	genuine_sampled = genuine_df.sample(n=num_majority_keep, random_state=RANDOM_SEED)
	balanced_df = pd.concat([fraud_df, genuine_sampled], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
	print(f"Balanced class counts -> Genuine: {sum(balanced_df[target_col]==0)}, Fraud: {sum(balanced_df[target_col]==1)}")
	return balanced_df


def split_train_test(df: pd.DataFrame, target_col: str = "Class"):
	"""Split the balanced dataset into train and test sets."""
	y = df[target_col]
	X = df.drop(columns=[target_col])
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
	)
	print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
	return X_train, X_test, y_train, y_test


def evaluate_model(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
	"""Compute standard classification metrics."""
	y_pred = model.predict(X_test)
	# For ROC-AUC we need probabilities if available, else fall back to predictions
	if hasattr(model, "predict_proba"):
		y_prob = model.predict_proba(X_test)[:, 1]
	else:
		# Some models might not have predict_proba; use decision_function if present
		if hasattr(model, "decision_function"):
			y_scores = model.decision_function(X_test)
			y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
		else:
			y_prob = y_pred

	metrics = {
		"model": model_name,
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, zero_division=0),
		"recall": recall_score(y_test, y_pred, zero_division=0),
		"f1": f1_score(y_test, y_pred, zero_division=0),
		"roc_auc": roc_auc_score(y_test, y_prob),
	}
	return metrics


def choose_best_model(results: list[dict]) -> dict:
	"""Pick the best model by ROC-AUC, then F1, then Recall."""
	results_sorted = sorted(
		results,
		key=lambda m: (m["roc_auc"], m["f1"], m["recall"]),
		reverse=True,
	)
	return results_sorted[0]


def main() -> None:
	# Load and clean data
	df_raw = load_dataset(DATA_PATH)
	df_clean = basic_cleaning(df_raw)

	# EDA outputs (saved to figures)
	perform_eda(df_clean)

	# Handle class imbalance using undersampling
	df_balanced = undersample(df_clean, target_col="Class", ratio=1.0)

	# Save feature statistics for inference defaults and validation
	features_df = df_balanced.drop(columns=["Class"])  # type: ignore[arg-type]
	feature_medians = features_df.median().to_dict()
	# Compute rich stats per feature
	feature_stats = {}
	for col in features_df.columns:
		series = features_df[col]
		q1 = float(series.quantile(0.25))
		q3 = float(series.quantile(0.75))
		feature_stats[col] = {
			"median": float(series.median()),
			"min": float(series.min()),
			"max": float(series.max()),
			"q1": q1,
			"q3": q3,
		}
	# Write rich stats and legacy medians file for backward compatibility
	with FEATURE_STATS_PATH.open("w") as f_stats:
		json.dump(feature_stats, f_stats)
	with FEATURE_MEDIANS_PATH.open("w") as f_meds:
		json.dump(feature_medians, f_meds)
	print(f"Saved feature stats to {FEATURE_STATS_PATH} and medians to {FEATURE_MEDIANS_PATH}")

	# Split into train and test
	X_train, X_test, y_train, y_test = split_train_test(df_balanced, target_col="Class")

	# -------------------------------
	# Step 3 – Model Training
	# -------------------------------
	print_header("Step 3 – Model Training")

	# Model 1: Logistic Regression
	log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=None)
	log_reg.fit(X_train, y_train)
	log_reg_metrics = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

	# Model 2: Random Forest Classifier
	rf_clf = RandomForestClassifier(
		n_estimators=200,
		random_state=RANDOM_SEED,
		n_jobs=-1,
		class_weight=None  # We already balanced the data
	)
	rf_clf.fit(X_train, y_train)
	rf_metrics = evaluate_model("Random Forest", rf_clf, X_test, y_test)

	# Collect results into a table for easy reading
	results = [log_reg_metrics, rf_metrics]
	results_df = pd.DataFrame(results)
	print("\nModel Comparison (higher is better):\n")
	print(results_df.to_string(index=False))

	# Save metrics to CSV as well
	results_csv_path = METRICS_DIR / "model_results.csv"
	results_df.to_csv(results_csv_path, index=False)
	print(f"\nSaved metrics table to {results_csv_path}")

	# Choose best model by ROC-AUC, then F1, then Recall
	best = choose_best_model(results)
	best_name = best["model"]
	best_model = log_reg if best_name == "Logistic Regression" else rf_clf

	# Persist the best model to disk
	joblib.dump(best_model, MODEL_PATH)
	print(f"\nSaved best model '{best_name}' to {MODEL_PATH}")

	# Final guidance for user
	print_header("What we did")
	print("- Loaded and cleaned the dataset using pandas.")
	print("- Performed EDA and saved plots under ml/figures/.")
	print("- Balanced classes with simple undersampling (1:1).")
	print("- Trained Logistic Regression and Random Forest models.")
	print("- Evaluated with accuracy, precision, recall, F1, ROC-AUC.")
	print("- Saved the best model to 'fraud_model.pkl' and feature stats to 'feature_stats.json'.")


if __name__ == "__main__":
	main() 
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import joblib
import numpy as np

# Paths to the trained model and feature stats
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "fraud_model.pkl"
FEATURE_STATS_PATH = PROJECT_ROOT / "feature_stats.json"
FEATURE_MEDIANS_PATH = PROJECT_ROOT / "feature_medians.json"

_cached_model = None
_cached_medians: Dict[str, float] | None = None
_cached_stats: Dict[str, Dict[str, float]] | None = None


def _load_model_and_medians():
	"""Load the trained model and stats/medians just once and cache them."""
	global _cached_model, _cached_medians, _cached_stats
	if _cached_model is None:
		if not MODEL_PATH.exists():
			raise FileNotFoundError(
				f"Model file not found at {MODEL_PATH}. Please run the training script first."
			)
		_cached_model = joblib.load(MODEL_PATH)
	if _cached_stats is None or _cached_medians is None:
		if FEATURE_STATS_PATH.exists():
			with FEATURE_STATS_PATH.open("r") as f:
				_cached_stats = json.load(f)
			# derive medians from stats
			_cached_medians = {k: float(v.get("median", 0.0)) for k, v in (_cached_stats or {}).items()}
		elif FEATURE_MEDIANS_PATH.exists():
			with FEATURE_MEDIANS_PATH.open("r") as f:
				_cached_medians = json.load(f)
			_cached_stats = {}
		else:
			_cached_medians = {}
			_cached_stats = {}


def home(request: HttpRequest) -> HttpResponse:
	"""Render the prediction form page."""
	# Ensure model is loadable (will show a friendly message if missing)
	model_ready = MODEL_PATH.exists()
	return render(request, "predictor/home.html", {"model_ready": model_ready})


def _build_feature_vector_from_form(post_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
	"""Map form inputs to the model's expected feature order; validate, impute, and clip.
	Returns (feature_vector, metadata about defaults/clips).
	"""
	_load_model_and_medians()

	# Define the expected feature columns as in the creditcard.csv (without the target 'Class')
	expected_columns = [
		"Time",
		"V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
		"V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
		"V21","V22","V23","V24","V25","V26","V27","V28",
		"Amount",
	]

	values: List[float] = []
	defaulted: List[str] = []
	clipped: List[str] = []

	for col in expected_columns:
		raw = post_data.get(col, "")
		if raw is None or str(raw).strip() == "":
			val = float(_cached_medians.get(col, 0.0))
			defaulted.append(col)
		else:
			try:
				val = float(raw)
			except ValueError:
				val = float(_cached_medians.get(col, 0.0))
				defaulted.append(col)

		# Clip to training min/max if available
		stats = (_cached_stats or {}).get(col)
		if stats is not None:
			min_v = float(stats.get("min", val))
			max_v = float(stats.get("max", val))
			if val < min_v:
				val = min_v
				clipped.append(col)
			elif val > max_v:
				val = max_v
				clipped.append(col)

		values.append(val)

	meta = {"defaulted_fields": defaulted, "clipped_fields": clipped}
	return np.array(values, dtype=float).reshape(1, -1), meta


def predict(request: HttpRequest) -> HttpResponse:
	"""Handle prediction requests from the form."""
	if request.method != "POST":
		return render(request, "predictor/home.html", {"error": "Please submit the form.", "model_ready": MODEL_PATH.exists()})

	try:
		_load_model_and_medians()
		x, meta = _build_feature_vector_from_form(request.POST)
		prediction = _cached_model.predict(x)[0]
		is_fraud = bool(prediction == 1)
		message = "❌ Fraud Detected!" if is_fraud else "✅ Transaction is Genuine"
		probability = None
		if hasattr(_cached_model, "predict_proba"):
			probability = float(_cached_model.predict_proba(x)[0][1])
		return render(
			request,
			"predictor/result.html",
			{
				"message": message,
				"is_fraud": is_fraud,
				"probability": probability,
				"defaulted_fields": meta.get("defaulted_fields", []),
				"clipped_fields": meta.get("clipped_fields", []),
			},
		)
	except FileNotFoundError as e:
		return render(request, "predictor/home.html", {"error": str(e), "model_ready": False})
	except Exception as e:
		return render(request, "predictor/home.html", {"error": f"Error: {e}", "model_ready": MODEL_PATH.exists()})


@csrf_exempt
def predict_api(request: HttpRequest) -> JsonResponse:
	"""JSON API endpoint for predictions.
	Accepts POST form or JSON with keys: Time, Amount, optional V1..V28.
	"""
	if request.method != "POST":
		return JsonResponse({"error": "Use POST."}, status=405)

	try:
		if request.content_type and "application/json" in request.content_type:
			try:
				payload = json.loads(request.body.decode("utf-8"))
			except Exception:
				return JsonResponse({"error": "Invalid JSON body."}, status=400)
		else:
			payload = {k: v for k, v in request.POST.items()}

		_load_model_and_medians()
		x, meta = _build_feature_vector_from_form(payload)
		pred = int(_cached_model.predict(x)[0])
		prob = None
		if hasattr(_cached_model, "predict_proba"):
			prob = float(_cached_model.predict_proba(x)[0][1])
		return JsonResponse({
			"prediction": pred,
			"is_fraud": bool(pred == 1),
			"probability": prob,
			"defaulted_fields": meta.get("defaulted_fields", []),
			"clipped_fields": meta.get("clipped_fields", []),
		})
	except FileNotFoundError as e:
		return JsonResponse({"error": str(e)}, status=500)
	except Exception as e:
		return JsonResponse({"error": f"Error: {e}"}, status=500)
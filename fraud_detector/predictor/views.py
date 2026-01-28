from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from django.apps import apps
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache

# Production-grade logging
logger = logging.getLogger('fraud_detection')
# Ensure logs directory exists (optional, depends on settings)

def _get_config():
	return apps.get_app_config('predictor')

def home(request: HttpRequest) -> HttpResponse:
	config = _get_config()
	model_ready = config.pipeline is not None
	return render(request, "predictor/home.html", {"model_ready": model_ready})

def validate_and_sanitize_input(post_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
	"""
	Production-grade validation and sanitization.
	Handles missing values, non-numeric data, and outliers via clipping.
	"""
	config = _get_config()
	stats = config.feature_stats or {}
	
	errors = []
	features = []
	
	# Expected feature columns
	v_cols = [f"V{i}" for i in range(1, 29)]
	expected_cols = ["Time"] + v_cols + ["Amount"]
	
	transformed_values = []
	
	for col in expected_cols:
		raw_value = str(post_data.get(col, "")).strip()
		
		# 1. Handle Missing/Empty
		if not raw_value:
			val = float(stats.get(col, {}).get("median", 0.0))
			errors.append(f"{col} was missing; used median.")
		else:
			# 2. Check Numeric
			try:
				val = float(raw_value)
				if not np.isfinite(val):
					raise ValueError("Non-finite")
			except ValueError:
				val = float(stats.get(col, {}).get("median", 0.0))
				errors.append(f"{col} must be a number; used median.")
		
		# 3. Correct Clipping Logic (Task 5)
		col_stats = stats.get(col)
		if col_stats:
			min_v = float(col_stats.get("min", val))
			max_v = float(col_stats.get("max", val))
			clipped_val = np.clip(val, min_v, max_v)
			
			if val != clipped_val:
				# Log but don't necessarily show as error to user unless relevant
				# errors.append(f"{col} outside normal range ({val:.2f} -> {clipped_val:.2f})")
				val = clipped_val
		
		transformed_values.append(val)
	
	# Amount specific check
	amount_idx = expected_cols.index("Amount")
	if transformed_values[amount_idx] < 0:
		errors.append("Amount cannot be negative; reset to 0.")
		transformed_values[amount_idx] = 0.0

	return np.array(transformed_values).reshape(1, -1), errors

def predict(request: HttpRequest) -> HttpResponse:
	"""Handle prediction requests from the form with monitoring."""
	if request.method != "POST":
		return render(request, "predictor/home.html", {"error": "Please submit the form.", "model_ready": True})

	start_time = datetime.now()
	config = _get_config()
	
	if not config.pipeline:
		return render(request, "predictor/home.html", {"error": "Model not loaded.", "model_ready": False})

	try:
		# Rate limiting (Simple example)
		ip = request.META.get('REMOTE_ADDR', 'unknown')
		cache_key = f'rate_limit_{ip}'
		request_count = cache.get(cache_key, 0)
		if request_count > 100:
			return render(request, "predictor/home.html", {"error": "Rate limit exceeded (100 req/hr).", "model_ready": True})
		cache.set(cache_key, request_count + 1, 3600)

		# Process input
		x_df_data, validation_warnings = validate_and_sanitize_input(request.POST)
		
		# Prediction
		prediction = config.pipeline.predict(x_df_data)[0]
		probability = None
		if hasattr(config.pipeline, "predict_proba"):
			probability = float(config.pipeline.predict_proba(x_df_data)[0][1])
		
		is_fraud = bool(prediction == 1)
		message = "❌ Fraud Detected!" if is_fraud else "✅ Transaction is Genuine"
		
		# Monitoring Log (Task 8)
		latency_ms = (datetime.now() - start_time).total_seconds() * 1000
		logger.info(json.dumps({
			'timestamp': datetime.now().isoformat(),
			'prediction': int(prediction),
			'probability': probability,
			'latency_ms': latency_ms,
			'ip': ip,
			'warnings': validation_warnings
		}))

		return render(
			request,
			"predictor/result.html",
			{
				"message": message,
				"is_fraud": is_fraud,
				"probability": probability,
				"warnings": validation_warnings,
				"latency": f"{latency_ms:.2f}ms"
			},
		)
	except Exception as e:
		logger.error(f"Prediction Error: {e}")
		return render(request, "predictor/home.html", {"error": f"Internal Error: {e}", "model_ready": True})

@csrf_exempt
def predict_api(request: HttpRequest) -> JsonResponse:
	"""JSON API endpoint with monitoring."""
	if request.method != "POST":
		return JsonResponse({"error": "Use POST."}, status=405)

	start_time = datetime.now()
	config = _get_config()
	
	try:
		if request.content_type == "application/json":
			payload = json.loads(request.body.decode("utf-8"))
		else:
			payload = request.POST

		x, warnings = validate_and_sanitize_input(payload)
		pred = int(config.pipeline.predict(x)[0])
		prob = float(config.pipeline.predict_proba(x)[0][1]) if hasattr(config.pipeline, "predict_proba") else None
		
		latency = (datetime.now() - start_time).total_seconds() * 1000
		
		return JsonResponse({
			"prediction": pred,
			"is_fraud": bool(pred == 1),
			"probability": prob,
			"latency_ms": latency,
			"warnings": warnings
		})
	except Exception as e:
		return JsonResponse({"error": str(e)}, status=500)
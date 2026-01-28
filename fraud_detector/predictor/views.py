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

# ==============================================================================
# ENTERPRISE API CORE - FRAUD DETECTION SYSTEM
# ==============================================================================

logger = logging.getLogger('fraud_detection')

def _get_config():
	return apps.get_app_config('predictor')

def home(request: HttpRequest) -> HttpResponse:
	"""Landing page with system status.""""
	config = _get_config()
	model_ready = config.pipeline is not None
	return render(request, "predictor/home.html", {
		"model_ready": model_ready,
		"system_version": config.model_payload.get('version', 'Unknown') if config.model_payload else 'Offline'
	})

def health_check(request: HttpRequest) -> JsonResponse:
	""Internal health monitor for Kubernetes/Status pages.""""
	config = _get_config()
	status = "healthy" if config.pipeline else "unhealthy"
	return JsonResponse({
		"status": status,
		"timestamp": datetime.now().isoformat(),
		"engine_version": config.model_payload.get('version', 'N/A') if config.model_payload else 'N/A'
	})

def validate_and_sanitize_input(post_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
	"""Strict validation and outlier clipping for financial data integrity."""
	config = _get_config()
	stats = config.feature_stats or {}
	
	warnings = []
	
	v_cols = [f"V{i}" for i in range(1, 29)]
	expected_cols = ["Time"] + v_cols + ["Amount"]
	
	transformed_values = []
	
	for col in expected_cols:
		raw_value = str(post_data.get(col, "")).strip()
		
		# Imputation: Use training median for missing or invalid inputs
		if not raw_value:
			val = float(stats.get(col, {}).get("median", 0.0))
			warnings.append(f"Imputed {col} (Missing)")
		else:
			try:
				val = float(raw_value)
				if not np.isfinite(val):
					raise ValueError
			except ValueError:
				val = float(stats.get(col, {}).get("median", 0.0))
				warnings.append(f"Imputed {col} (Invalid Format)")
		
		# Bound Control: Clip to training distribution limits
		col_stats = stats.get(col)
		if col_stats:
			min_v = float(col_stats.get("min", val))
			max_v = float(col_stats.get("max", val))
			clipped_val = np.clip(val, min_v, max_v)
			
			if val != clipped_val:
				warnings.append(f"Clipped {col} (Outlier)")
				val = clipped_val
		
		transformed_values.append(val)
	
	# Business Rule: Negative volumes are logically impossible
	amount_idx = expected_cols.index("Amount")
	if transformed_values[amount_idx] < 0:
		warnings.append("Normalized negative Amount to 0.0")
		transformed_values[amount_idx] = 0.0

	return np.array(transformed_values).reshape(1, -1), warnings

def predict(request: HttpRequest) -> HttpResponse:
	"""Main inference endpoint with high-fidelity logging and rate limiting."""
	if request.method != "POST":
		return render(request, "predictor/home.html", {"error": "Invalid Access Method.", "model_ready": True})

	start_time = datetime.now()
	config = _get_config()
	
	if not config.pipeline:
		return render(request, "predictor/home.html", {"error": "Inference Engine Offline.", "model_ready": False})

	try:
		# DDoS / Scraping Protection
		ip = request.META.get('REMOTE_ADDR', 'unknown')
		cache_key = f'rate_limit_{ip}'
		request_count = cache.get(cache_key, 0)
		if request_count > 100:
			logger.warning(f"Rate limit hit for IP: {ip}")
			return render(request, "predictor/home.html", {"error": "Security Limit Reached. Try again later.", "model_ready": True})
		cache.set(cache_key, request_count + 1, 3600)

		# Data Processing
		x_data, validation_warnings = validate_and_sanitize_input(request.POST)
		
		# Inference
		prediction = config.pipeline.predict(x_data)[0]
		probability = None
		if hasattr(config.pipeline, "predict_proba"):
			probability = float(config.pipeline.predict_proba(x_data)[0][1])
		
		is_fraud = bool(prediction == 1)
		
		# Telemetry
		latency_ms = (datetime.now() - start_time).total_seconds() * 1000
		logger.info(json.dumps({
			'event': 'prediction',
			'version': config.model_payload.get('version'),
			'prediction': int(prediction),
			'probability': probability,
			'latency_ms': latency_ms,
			'ip': ip
		}))

		return render(request, "predictor/result.html", {
			"is_fraud": is_fraud,
			"probability": probability,
			"warnings": validation_warnings,
			"latency": f"{latency_ms:.2f}ms",
			"timestamp": datetime.now()
		})
	except Exception as e:
		logger.error(f"SYSTEM CRASH: {e}", exc_info=True)
		return render(request, "predictor/home.html", {"error": "Internal System Anomaly detected.", "model_ready": True})

@csrf_exempt
def predict_api(request: HttpRequest) -> JsonResponse:
	"""Structured REST API for external integration."""
	if request.method != "POST":
		return JsonResponse({"error": "Method Not Allowed"}, status=405)

	start_time = datetime.now()
	config = _get_config()
	
	try:
		# Payload Extraction
		if request.content_type == "application/json":
			payload = json.loads(request.body.decode("utf-8"))
		else:
			payload = request.POST

		x, warnings = validate_and_sanitize_input(payload)
		
		# Core Analysis
		pred = int(config.pipeline.predict(x)[0])
		prob = float(config.pipeline.predict_proba(x)[0][1]) if hasattr(config.pipeline, "predict_proba") else None
		
		latency = (datetime.now() - start_time).total_seconds() * 1000
		
		return JsonResponse({
			"success": True,
			"data": {
				"prediction": pred,
				"is_fraud": bool(pred == 1),
				"probability": prob,
				"analysis_summary": warnings
			},
			"telemetry": {
				"latency_ms": latency,
				"engine_version": config.model_payload.get('version')
			}
		})
	except Exception as e:
		return JsonResponse({"success": False, "error": str(e)}, status=500)
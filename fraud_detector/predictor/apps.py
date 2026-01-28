from django.apps import AppConfig
import joblib
import os
import json
from pathlib import Path

class PredictorConfig(AppConfig):
	default_auto_field = "django.db.models.BigAutoField"
	name = "predictor"

	# Production: Load model once at startup
	model_payload = None
	pipeline = None
	feature_stats = None

	def ready(self):
		# Avoid loading during migration or shell commands where not needed
		if os.environ.get('RUN_MAIN') == 'true' or not os.environ.get('SERVER_SOFTWARE'):
			project_root = Path(__file__).resolve().parents[2]
			model_path = project_root / "fraud_model.pkl"
			
			if model_path.exists():
				try:
					self.model_payload = joblib.load(model_path)
					if isinstance(self.model_payload, dict):
						self.pipeline = self.model_payload.get('pipeline')
						self.feature_stats = self.model_payload.get('feature_stats')
					else:
						# Fallback for old model format
						self.pipeline = self.model_payload
						stats_path = project_root / "feature_stats.json"
						if stats_path.exists():
							with stats_path.open('r') as f:
								self.feature_stats = json.load(f)
					print(f"✅ Fraud Detection Model Loaded (Version: {self.model_payload.get('version', 'legacy') if isinstance(self.model_payload, dict) else 'legacy'})")
				except Exception as e:
					print(f"⚠️ Error loading model at startup: {e}")
from django.apps import AppConfig
import joblib
import os
import sys
import json
from pathlib import Path

class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    # Singletons
    model_payload = None
    pipeline = None
    feature_stats = None

    def ready(self):
        # SKIP loading if we are running management commands (build phase)
        # This prevents memory crashes during 'collectstatic' or 'migrate' on Render Free Tier
        if any(x in sys.argv for x in ['collectstatic', 'makemigrations', 'migrate', 'check', 'test']):
            print("⚠️ AppConfig: Skipping model load during management command.")
            return

        # Load only in runtime environments (Gunicorn or Dev Server)
        should_load = (
            os.environ.get('RUN_MAIN') == 'true' or 
            'gunicorn' in os.environ.get('SERVER_SOFTWARE', '').lower() or 
            os.environ.get('RENDER') # Render Runtime sets this
        )

        if should_load:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "fraud_model.pkl"
            
            if model_path.exists():
                try:
                    self.model_payload = joblib.load(model_path)
                    if isinstance(self.model_payload, dict):
                        self.pipeline = self.model_payload.get('pipeline')
                        self.feature_stats = self.model_payload.get('feature_stats')
                    else:
                        self.pipeline = self.model_payload
                        stats_path = project_root / "feature_stats.json"
                        if stats_path.exists():
                            with stats_path.open('r') as f:
                                self.feature_stats = json.load(f)
                    print(f"✅ Fraud Detection Engine initialized successfully.")
                except Exception as e:
                    print(f"⚠️ AppConfig: Failed to load model: {e}")
            else:
                print(f"⚠️ AppConfig: Model file not found at {model_path}")
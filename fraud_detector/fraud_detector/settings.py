from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-change-this-in-production"
DEBUG = True
ALLOWED_HOSTS: list[str] = []

INSTALLED_APPS = [
	"django.contrib.admin",
	"django.contrib.auth",
	"django.contrib.contenttypes",
	"django.contrib.sessions",
	"django.contrib.messages",
	"django.contrib.staticfiles",
	"predictor",
]

MIDDLEWARE = [
	"django.middleware.security.SecurityMiddleware",
	"django.contrib.sessions.middleware.SessionMiddleware",
	"django.middleware.common.CommonMiddleware",
	"django.middleware.csrf.CsrfViewMiddleware",
	"django.contrib.auth.middleware.AuthenticationMiddleware",
	"django.contrib.messages.middleware.MessageMiddleware",
	"django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "fraud_detector.urls"

TEMPLATES = [
	{
		"BACKEND": "django.template.backends.django.DjangoTemplates",
		"DIRS": [BASE_DIR / "templates"],
		"APP_DIRS": True,
		"OPTIONS": {
			"context_processors": [
				"django.template.context_processors.debug",
				"django.template.context_processors.request",
				"django.contrib.auth.context_processors.auth",
				"django.contrib.messages.context_processors.messages",
			],
		},
	},
]

WSGI_APPLICATION = "fraud_detector.wsgi.application"

DATABASES = {
	"default": {
		"ENGINE": "django.db.backends.sqlite3",
		"NAME": BASE_DIR / "db.sqlite3",
	}
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Production Hardening: Cache & Logging
CACHES = {
	"default": {
		"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
		"LOCATION": "unique-snowflake",
	}
}

LOGGING = {
	"version": 1,
	"disable_existing_loggers": False,
	"formatters": {
		"json": {
			"format": "%(message)s",
		},
	},
	"handlers": {
		"console": {
			"class": "logging.StreamHandler",
		},
		"file": {
			"level": "INFO",
			"class": "logging.FileHandler",
			"filename": os.path.join(BASE_DIR, "fraud_detection.log"),
			"formatter": "json",
		},
	},
	"loggers": {
		"fraud_detection": {
			"handlers": ["console", "file"],
			"level": "INFO",
			"propagate": True,
		},
	},
}
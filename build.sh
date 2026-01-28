#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Run migrations if you have a database
cd fraud_detector
python manage.py collectstatic --no-input
python manage.py migrate

# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBUG=False
ENV PYTHONDONTWRITEBYTECODE=1

# Run build steps (migrations, static files)
# Note: We assume fraud_model.pkl exists or we run training in a separate CI stage
RUN cd fraud_detector && python manage.py collectstatic --no-input

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "--chdir", "fraud_detector", "--bind", "0.0.0.0:8000", "fraud_detector.wsgi:application"]

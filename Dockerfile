# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Environment variables will be passed at runtime via .env or docker-compose
ENV PYTHONUNBUFFERED=1

# Run the Gradio application
CMD ["python", "app.py"]

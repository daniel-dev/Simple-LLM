# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/cache models/trained models/checkpoints logs outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

# Expose the port
EXPOSE 5000

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "app.py"]

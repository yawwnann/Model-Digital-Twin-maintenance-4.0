# FlexoTwin Smart Maintenance 4.0 - Dockerfile
# Multi-stage build untuk production deployment

# ========================================
# Stage 1: Development Base
# ========================================
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=05_API/07_api_interface.py
ENV FLASK_ENV=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# Stage 2: Dependencies
# ========================================
FROM base as dependencies

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ========================================
# Stage 3: Application
# ========================================
FROM dependencies as application

# Copy project files
COPY 01_Scripts/ 01_Scripts/
COPY 02_Models/ 02_Models/
COPY 03_Data/ 03_Data/
COPY 05_API/ 05_API/
COPY 06_Documentation/ 06_Documentation/
COPY 07_Examples/ 07_Examples/

# Copy configuration files
COPY requirements.txt .
COPY README.md .
COPY PROJECT_INDEX.md .

# Create non-root user untuk security
RUN useradd --create-home --shell /bin/bash flexotwin && \
    chown -R flexotwin:flexotwin /app

USER flexotwin

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command
CMD ["python", "05_API/07_api_interface.py"]

# ========================================
# Stage 4: Production (Optional)
# ========================================
FROM application as production

# Install production WSGI server
USER root
RUN pip install --no-cache-dir gunicorn

USER flexotwin

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "sync", "--timeout", "120", "05_API.07_api_interface:app"]
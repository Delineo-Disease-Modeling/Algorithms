# Delineo Algorithms
# Flask/Waitress server for CZ generation, popgen, patterns
# Port: 1880
#
# Data directory (server/data/) is NOT baked into the image — mount it at
# runtime via docker-compose volume.

FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY server/ server/

EXPOSE 1880

WORKDIR /app/server
CMD ["python", "waitress_app.py"]

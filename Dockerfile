FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8501

WORKDIR /app

# Install system dependencies if required (build-essential, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source and project files
COPY pyproject.toml .
COPY main.py .
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY database/ database/

# Install project in editable/production mode
RUN pip install --no-cache-dir -e .

EXPOSE 8501

# Entrypoint runs Streamlit binding to 0.0.0.0 and listening on $PORT
CMD streamlit run main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

FROM python:3.10-slim

WORKDIR /app

# Install system deps for spacy model download
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy NER model for Presidio
RUN python -m spacy download en_core_web_lg

COPY . .

EXPOSE 8501

# Health check endpoint for CI/CD smoke test
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8501/healthz || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
  "--server.port=8501", \
  "--server.address=0.0.0.0", \
  "--server.headless=true"]

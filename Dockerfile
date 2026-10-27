FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    EEGAGENT_WORKSPACE=/data/workspace EEGAGENT_PROVIDER=anthropic

WORKDIR /app
COPY pyproject.toml README.md ./
COPY eegsi ./eegsi
COPY eegagent ./eegagent
COPY eegmodels ./eegmodels
RUN pip install --no-cache-dir ".[api,ui]"

RUN mkdir -p /data/workspace
VOLUME ["/data"]
EXPOSE 8000 8501

# API by default; run the UI with: docker run ... eegagent ui --api http://api:8000
CMD ["eegagent", "serve", "--host", "0.0.0.0", "--port", "8000"]

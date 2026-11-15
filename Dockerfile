FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
ARG WITH_ENCODER=1
COPY requirements.txt requirements-encoder.txt ./
RUN if [ "$WITH_ENCODER" = "1" ]; then \
    pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu \
    && pip install --no-cache-dir -r requirements-encoder.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/events && chown app /app/data/events

COPY alfarec/ ./alfarec/
COPY artifacts/ ./artifacts/
RUN if [ "$WITH_ENCODER" = "1" ]; then python -m alfarec.pipeline.fetch_encoder; fi
USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2))['status'] == 'ok'"

CMD ["python", "-m", "uvicorn", "alfarec.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

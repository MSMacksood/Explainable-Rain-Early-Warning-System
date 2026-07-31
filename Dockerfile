# -------- Stage 1: builder — compile wheels in an isolated layer --------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# -------- Stage 2: runtime — slim image, non-root, prebuilt wheels --------
FROM python:3.11-slim AS runtime
LABEL org.opencontainers.image.title="sl-rain-early-warning" \
      org.opencontainers.image.description="Monsoon-aware rain early-warning API for Sri Lanka"

RUN useradd --create-home --uid 1001 appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

COPY app ./app
COPY src ./src
COPY config ./config
COPY models ./models

USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

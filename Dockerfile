FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY static ./static
RUN useradd -m app && mkdir -p outputs && chown app outputs
USER app
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"
# LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY (and optional OPENFDA_API_KEY) come from the platform's secrets
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && phishguard prepare && phishguard baseline && rm -rf data
RUN useradd --create-home --uid 10001 runner
USER runner
EXPOSE 8000
CMD ["uvicorn", "phishguard.api:app", "--host", "0.0.0.0", "--port", "8000"]

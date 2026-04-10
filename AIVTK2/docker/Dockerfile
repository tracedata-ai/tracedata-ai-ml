FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv pip install --system .

CMD ["python", "-m", "src.compliance.aivt2_runner", "--tracking-uri", "./mlruns", "--experiment-name", "smoothness-10min-production", "--output-dir", "reports/aivt2"]


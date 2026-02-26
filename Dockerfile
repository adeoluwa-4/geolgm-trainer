FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir .

COPY src /app/src
COPY dashboards /app/dashboards
COPY configs /app/configs

EXPOSE 8501
CMD ["geolgm", "dashboard", "--port", "8501"]

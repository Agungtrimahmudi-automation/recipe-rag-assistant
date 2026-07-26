FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tools/ tools/
COPY config/ config/
COPY data/recipes.jsonl data/recipes.jsonl
COPY data/index/ data/index/

EXPOSE 8000

CMD ["uvicorn", "tools.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run train.py during build to ensure the model exists
RUN python src/train.py

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

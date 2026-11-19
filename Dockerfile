FROM python:3.13-slim

WORKDIR /code

# Install dependencies first so Docker can reuse this layer when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY churn ./churn
COPY app ./app
COPY models ./models

EXPOSE 80
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
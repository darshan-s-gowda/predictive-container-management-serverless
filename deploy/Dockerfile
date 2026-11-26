FROM python:3.11-slim

WORKDIR /app
COPY src/ /app/src/
COPY outputs/ /app/outputs/

RUN pip install --no-cache-dir numpy pandas stable-baselines3 torch --extra-index-url https://download.pytorch.org/whl/cpu || true

EXPOSE 8080
CMD ["python", "src/ecommerce_app/server.py", "8080"]

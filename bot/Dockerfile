FROM python:3.11-slim

WORKDIR /app

COPY bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY bot/ ./bot/

CMD ["python", "/app/bot/bot.py"]

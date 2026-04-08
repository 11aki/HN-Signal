FROM python:3.11-slim

RUN pip install supercronic==0.2.33 --quiet

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY crontab /app/trainer/crontab

CMD ["supercronic", "/app/trainer/crontab"]

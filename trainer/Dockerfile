FROM python:3.11-slim

RUN pip install supercronic==0.2.33 --quiet

WORKDIR /app
COPY trainer/requirements.txt .
RUN pip install -r requirements.txt --quiet

COPY shared/ ./shared/
COPY trainer/ ./trainer/

CMD ["supercronic", "/app/trainer/crontab"]

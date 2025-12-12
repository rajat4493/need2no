FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install .

EXPOSE 9000

CMD ["n2n-serve", "--host", "0.0.0.0", "--port", "9000"]

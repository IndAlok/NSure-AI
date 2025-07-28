# ---- Stage 1: Build Stage ----
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# ---- Stage 2: Final Stage ----
FROM python:3.11-slim

RUN addgroup --system app && adduser --system --group app
WORKDIR /home/app

COPY --from=builder /usr/src/app/wheels /wheels
COPY . .

RUN pip install --no-cache /wheels/*
RUN chown -R app:app /home/app

USER app

# HF Port
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

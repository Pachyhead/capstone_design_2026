From python:3.10.15

WORKDIR /app


RUN apt-get update && apt-get install -y \
    python3-dev build-essential gcc g++ pkg-config \
    libgirepository1.0-dev libcairo2-dev \
    libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev libtiff5-dev \
    libdbus-1-dev libdbus-glib-1-dev \
    tesseract-ocr libtesseract-dev \
    libgl1 libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "./qwen_tts_api/api_server.py"]
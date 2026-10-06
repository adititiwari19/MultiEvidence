FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
COPY run.sh .
RUN mkdir -p data/media data/frames
ENV MEDIA_DIR=/app/data/media FRAME_DIR=/app/data/frames
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["bash", "run.sh"]

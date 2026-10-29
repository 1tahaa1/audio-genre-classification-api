FROM python:3.10-slim

# Installation des dépendances système requises (ffmpef et libsndfile)

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt-get/lists/*


    WORKDIR /app

    # copie et installation des dépendances Pyhton 

    COPY requirements.txt . 
    RUN pip install --no-cache-dir -r requirements.txt

    # Copie du code source


    COPY src/ ./src

    EXPOSE 8000

    # Lancement de l'API 

    CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

    
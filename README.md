# API REST de Classification Audio et Analyse Acoustique

Une solution pour la classification de genres musicaux (dataset GTZAN) et l extraction de parametres acoustiques (BPM, duree, tonalite).

---

## Présentation du projet

Cette API REST combine traitement du signal audio et Deep Learning :
1. Inherence Deep Learning : Classification de genres via Hugging Face (dima806/music_genres_classification).
2. Analyse Acoustique : Extraction avec librosa du tempo/BPM, de la duree et de la tonalite (algorithme de Krumhansl-Schmuckler).
3. Conteneurisation Docker : Isolation et deploiement sous forme de conteneur.

---

## Stack Technique

* Langage : Python 3.10
* API Framework : FastAPI, Uvicorn
* Deep Learning : PyTorch, Transformers (Hugging Face)
* Traitement Audio : Librosa, Soundfile, FFmpeg
* Conteneurisation : Docker

---

## Arborescence du Projet

```text
audio_genre_classification_api/
├── src/
│   ├── __init__.py
│   ├── audio_processing.py  # Calculs acoustiques (BPM, duree, tonalite)
│   ├── model.py             # Inherence Hugging Face / PyTorch
│   └── api.py               # Endpoints FastAPI
├── Dockerfile               # Image Docker (FFmpeg, Libsndfile)
├── requirements.txt         # Dependances Python
└── README.md
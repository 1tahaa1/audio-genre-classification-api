import os
import tempfile
from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.model import get_classifier
from src.audio_processing import analyze_audio_features

app = FastAPI(
    title="API de Classification de Genres Musicaux",
    description="API REST MLOps servant un modèle de prédiction de genres audio."
)

@app.get("/")
def read_root():
    return {"message": "API de Classification Audio prête !"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validation du format
    if not file.filename.endswith(('.wav', '.mp3')):
        return {"error": "Format non valide ! Veuillez envoyer un fichier .wav ou .mp3"}
    
    # Récupérer l'extension du fichier (.wav ou .mp3)
    _, ext = os.path.splitext(file.filename)
    
    # Lecture des octets audio
    contents = await file.read()
    
    # Création d'un fichier temporaire pour librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
        temp_audio.write(contents)
        temp_path = temp_audio.name

    try:
        # 1. Analyse audio (BPM, durée, tonalité)
        audio_analysis = analyze_audio_features(temp_path)
        
        # 2. Prédiction du genre musical
        classifier = get_classifier()
        genre_prediction = classifier.predict(contents, file_extension=ext)
        
    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return {
        "filename": file.filename,
        "status": "Succès",
        "audio_analysis": audio_analysis,
        "genre": genre_prediction
    }

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
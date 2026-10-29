import os
import tempfile
from transformers import pipeline

class AudioClassifier:
    def __init__(self):
        print("Chargement du modèle de classification audio en cours...")
        self.classifier = pipeline(
            "audio-classification",
            model="dima806/music_genres_classification"
        )
        print("Modèle chargé !")

    def predict(self, audio_bytes: bytes, file_extension: str = ".wav"):
        """
        Sauvegarde temporairement les octets audio sur le disque et effectue la prédiction.
        """
        # Création d'un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        try:
            # Prédiction sur le chemin du fichier temporaire
            predictions = self.classifier(temp_path)
            top_prediction = predictions[0]
            
            result = {
                "label": top_prediction["label"],
                "score": round(float(top_prediction["score"]), 4),
                "all_predictions": [
                    {"label": p["label"], "score": round(float(p["score"]), 4)} 
                    for p in predictions[:3]
                ]
            }
        finally:
            # Nettoyage du fichier temporaire
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return result


# Instance globale (Singleton)
classifier_instance = None

def get_classifier():
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = AudioClassifier()
    return classifier_instance
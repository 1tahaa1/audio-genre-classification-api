import librosa
import numpy as np


def estimate_key(y, sr):
    """
    Estime la tonalité (Note + Mode Majeur/Mineur)
    via l'algorithme de Krumhansl-Schmuckler.
    """
    # Extraction du chromagramme
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_vector = np.mean(chroma, axis=1)

    # Profils des gammes majeures et mineures
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.34, 3.17, 3.18])

    notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    best_corr = -1
    estimated_key = "Inconnue"

    # Test des 12 rotations chromatiques
    for i in range(12):
        rotated_chroma = np.roll(chroma_vector, -i)

        # Corrélation Majeure
        corr_major = np.corrcoef(rotated_chroma, major_profile)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            estimated_key = f"{notes[i]} Major"

        # Corrélation Mineure
        corr_minor = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
        if corr_minor > best_corr:
            best_corr = corr_minor
            estimated_key = f"{notes[i]} Minor"

    return estimated_key


def analyze_audio_features(file_path: str):
    """
    Extraction des caractéristiques acoustiques (BPM, durée, tonalité).
    """
    # Chargement du fichier audio
    y, sr = librosa.load(file_path, sr=22050)

    # Tempo / BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Durée en secondes
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Tonalité (Krumhansl-Schmuckler)
    estimated_key_val = estimate_key(y, sr)

    return {
        "bpm": round(tempo, 2),
        "duration": round(duration, 2),
        "estimated_key": estimated_key_val,
        "sample_rate": sr
    }
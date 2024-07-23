from gtts import gTTS
import tempfile
import os

class TextToSpeech:
    def __init__(self, tts_model_name):
        self.tts_model_name = tts_model_name  # Conservez le nom du modèle s'il y a une logique spécifique à ce modèle
        # gTTS n'a pas besoin d'initialisation avec un modèle ici, donc vous pouvez ajuster selon vos besoins
        # Par exemple, vous pouvez initialiser des paramètres supplémentaires si nécessaire

    def synthesize(self, text):
        tts = gTTS(text=text, lang='en')  # Utilisation de gTTS pour générer la parole
        # Enregistrez le résultat dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
            # Vous pouvez retourner le chemin du fichier ou lire le fichier en mémoire
            with open(temp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
        
        os.remove(temp_filename)  # Supprimer le fichier temporaire après utilisation
        return audio_data

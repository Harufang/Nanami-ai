import gradio as gr
from chatbot.chatbot import Chatbot
from training.youtube_training import YouTubeTraining
from optimizations.gpu_optimizations import clear_cache
import speech_recognition as sr
import threading

MODEL_NAME_STT = "openai/whisper-large"
MODEL_NAME_TTS = "tts_model_name"
LLAMA_MODEL_NAME = "meta-llama/Llama-3-7B"  # Remplacez avec un modèle public connu
HUGGINGFACE_TOKEN = "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"  # Remplacez par votre token Hugging Face

chatbot = Chatbot(MODEL_NAME_STT, MODEL_NAME_TTS, LLAMA_MODEL_NAME, HUGGINGFACE_TOKEN)
yt_trainer = YouTubeTraining("downloaded_videos/")

recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_and_respond():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Dites quelque chose...")

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            audio_path = "temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            response = chatbot.process_audio(audio_path)
            clear_cache()
            return response
        except sr.WaitTimeoutError:
            return "Temps d'attente dépassé. Réessayez."
        except sr.UnknownValueError:
            return "Impossible de comprendre l'audio."
        except Exception as e:
            return f"Erreur : {e}"

def continuous_chat():
    response = ""
    while "stop" not in response.lower():
        response = listen_and_respond()
        if "stop" not in response.lower():
            print("Réponse du chatbot:", response)
    print("Chat terminé.")

def train_from_youtube(url):
    video_path = yt_trainer.download_video(url)
    return f"Video downloaded to {video_path}"

def main():
    with gr.Blocks() as demo:
        with gr.Tab("Chatbot"):
            chat_output = gr.Textbox(label="Réponse")
            start_button = gr.Button("Start Conversation")

            def start_conversation():
                threading.Thread(target=continuous_chat).start()

            start_button.click(fn=start_conversation, inputs=None, outputs=None)

        with gr.Tab("YouTube Training"):
            url_input = gr.Textbox(label="YouTube URL")
            train_btn = gr.Button("Download and Train")
            train_output = gr.Textbox()
            train_btn.click(fn=train_from_youtube, inputs=url_input, outputs=train_output)

    demo.launch()

if __name__ == "__main__":
    main()

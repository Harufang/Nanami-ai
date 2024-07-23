import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimizations.gpu_optimizations import accelerator, enable_mixed_precision, optimize_memory

class SpeechToText:
    def __init__(self, model_name):
        self.device = accelerator.device
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model = accelerator.prepare(self.model)
            self.model.to(self.device)  # Transférer le modèle sur le GPU
            self.model.eval()  # Passer le modèle en mode évaluation
        except Exception as e:
            raise RuntimeError(f"Error during model setup: {e}")

    def transcribe(self, audio_path, sampling_rate=16000):
        try:
            # Vérifier si le chemin est un fichier valide
            if not os.path.isfile(audio_path):
                raise ValueError(f"Audio path is not a valid file: {audio_path}")

            # Charger l'audio avec torchaudio
            waveform, original_sampling_rate = torchaudio.load(audio_path, normalize=True)

            # Vérifier si le taux d'échantillonnage est correct
            if original_sampling_rate != sampling_rate:
                # Convertir le taux d'échantillonnage si nécessaire
                resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=sampling_rate)
                waveform = resampler(waveform)

            # Prétraitement de l'audio
            audio_input = self.processor(waveform.squeeze(0).numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_values.to(self.device)

            # Transcription
            with torch.no_grad():
                with enable_mixed_precision():
                    generated_ids = self.model.generate(audio_input)
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return transcription[0]
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")
        finally:
            torch.cuda.empty_cache()  # Libération de la mémoire GPU
            optimize_memory()  # Optimisation de la mémoire GPU

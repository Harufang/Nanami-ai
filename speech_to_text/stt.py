import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimizations.gpu_optimizations import accelerator, optimize_memory
from optimizations.gpu_optimizations import enable_mixed_precision, optimize_memory


# Environment variable to disable oneDNN optimizations if not required
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def process_data():
    with enable_mixed_precision():
        # Process your data using operations that benefit from mixed precision
        pass

    optimize_memory()  # Call after heavy GPU usage

class SpeechToText:
    def __init__(self, model_name, device=None):
        self.device = accelerator.device
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)  # Move model to device
            self.model = accelerator.prepare(self.model)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error during model setup: {e}")

    def load_audio_efficiently(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        return waveform.to(self.device), sr

    def transcribe(self, audio_path, sampling_rate=16000):
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio path is not a valid file: {audio_path}")

        waveform, original_sampling_rate = self.load_audio_efficiently(audio_path)
        if original_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=sampling_rate)
            waveform, original_sampling_rate = torchaudio.load(audio_path, normalize=True)
            waveform = waveform.to(self.device)

        try:
            with torch.cuda.amp.autocast():  # Ensuring autocast is applied correctly
                audio_input = self.processor(
                    waveform.squeeze(0),
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    language='en'
                )

                input_features = audio_input["input_features"].to(self.device)
                attention_mask = (input_features != self.processor.feature_extractor.padding_value).long()

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        max_length=100,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return transcription[0]
        except Exception as e:  # Catch any exception that occurred during transcription
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error during transcription: {str(e)}")
        finally:
            optimize_memory()  # This will run whether an exception occurred or not



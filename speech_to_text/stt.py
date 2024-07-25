import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimizations.gpu_optimizations import accelerator, optimize_memory, enable_mixed_precision
from pydub import AudioSegment
from io import BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SpeechToText:
    def __init__(self, model_name, device=None):
        self.device = device or accelerator.device
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.model = accelerator.prepare(self.model)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error during model setup: {e}")

    def load_audio_efficiently(self, file_path):
        # Use pydub to load and convert the audio file
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_samples = audio.get_array_of_samples()
        waveform = torch.tensor(audio_samples).float().div(32768.0).view(1, -1)
        return waveform.to(self.device), 16000

    def transcribe(self, audio_path, sampling_rate=16000):
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio path is not a valid file: {audio_path}")

        waveform, original_sampling_rate = self.load_audio_efficiently(audio_path)
        if original_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=sampling_rate)
            waveform = resampler(waveform)

        try:
            with torch.cuda.amp.autocast():
                audio_input = self.processor(
                    waveform.squeeze(0).numpy(),
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
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error during transcription: {str(e)}")
        finally:
            optimize_memory()

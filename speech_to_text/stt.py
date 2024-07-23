import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimizations.gpu_optimizations import accelerator, optimize_memory, enable_mixed_precision

# Setting environment variables for better performance and compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SpeechToText:
    def __init__(self, model_name, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)  # Move model to the device
            self.model = accelerator.prepare(self.model)  # Prepare model with Accelerator
            self.model.to(self.device)  # Ensure model is on the correct device
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error during model setup: {e}")
    
    def synthesize(self, text):
            # Example processing; ensure tensors are on the correct device
            input_tensor = torch.tensor([text]).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)  # Synthesize speech from text
            return output

    def load_audio_efficiently(self, file_path, sampling_rate=16000):
        waveform, original_sampling_rate = torchaudio.load(file_path, normalize=True)
        if original_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=sampling_rate)
            waveform = resampler(waveform)
        waveform = waveform.to(self.device)  # Ensure waveform is moved to the correct device
        return waveform

    def transcribe(self, audio_path):
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio path is not a valid file: {audio_path}")

        waveform = self.load_audio_efficiently(audio_path)
        
        try:
            with enable_mixed_precision():  # Use mixed precision for better performance
                audio_input = self.processor(
                    waveform.squeeze(0),
                    return_tensors="pt",
                    sampling_rate=16000,
                    language='en'
                )
                input_features = audio_input.input_values.to(self.device)  # Move to device
                attention_mask = audio_input.attention_mask.to(self.device) if 'attention_mask' in audio_input else None

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_features,
                        attention_mask=attention_mask,
                        max_length=100,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
                transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                return transcription
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error during transcription: {str(e)}")
        finally:
            optimize_memory()  # Optimize memory after heavy operations

    
# Example usage:
# stt = SpeechToText("facebook/wav2vec2-base-960h", device=torch.device('cuda'))
# transcription = stt.transcribe("path/to/audio/file.wav")
# print(transcription)

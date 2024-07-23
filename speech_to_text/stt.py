import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from optimizations.gpu_optimizations import accelerator, enable_mixed_precision, optimize_memory

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SpeechToText:
    def __init__(self, model_name):
        self.device = accelerator.device
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model = accelerator.prepare(self.model)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error during model setup: {e}")

    def transcribe(self, audio_path, sampling_rate=16000):
        try:
            if not os.path.isfile(audio_path):
                raise ValueError(f"Audio path is not a valid file: {audio_path}")

            waveform, original_sampling_rate = torchaudio.load(audio_path, normalize=True)

            if original_sampling_rate != sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=sampling_rate)
                waveform = resampler(waveform)

            with enable_mixed_precision():
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
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Error during transcription: {str(e)}")
        finally:
            optimize_memory()

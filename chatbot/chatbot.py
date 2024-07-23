from transformers import AutoModelForCausalLM, AutoTokenizer
from speech_to_text.stt import SpeechToText
from text_to_speech.tts import TextToSpeech
from optimizations.gpu_optimizations import accelerator, enable_mixed_precision, optimize_memory
import torch

class Chatbot:
    def __init__(self, stt_model_name, tts_model_name, llama_model_name, token):
        self.stt = SpeechToText(stt_model_name)
        self.tts = TextToSpeech(tts_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=token, low_cpu_mem_usage=True)
        self.model = self.model.half()
        self.model = accelerator.prepare(self.model)
        self.model = self.model.to(accelerator.device)
        self.model.eval()

    def process_audio(self, audio_path):
        try:
            text = self.stt.transcribe(audio_path)
            response = self.generate_response(text)
            audio_response = self.tts.synthesize(response)
            return audio_response
        finally:
            optimize_memory()

    def generate_response(self, text):
        """Generate a response from the text input using the model."""
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(inputs.shape, device=self.device)
        try:
            with torch.no_grad():
                with enable_mixed_precision():  # Use the updated autocast
                    outputs = self.model.generate(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        max_length=100,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        finally:
            optimize_memory()

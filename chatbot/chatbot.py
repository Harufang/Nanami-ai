from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech
from optimizations.gpu_optimizations import accelerator, optimize_memory, enable_mixed_precision
from accelerate import Accelerator
import torch
import IPython.display as ipd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

class Chatbot:
    def __init__(self, stt_model_name, tts_model_name, llama_model_name, token):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()

        self.stt = SpeechToText(stt_model_name, self.device)
        self.tts = TextToSpeech(tts_model_name, self.device, token)

        torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=token)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=token, low_cpu_mem_usage=True)
            self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
            self.model.to(self.device).half()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                self.model.to(self.device).half()

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def process_audio(self, audio_path):
        try:
            text = self.stt.transcribe(audio_path)
            response = self.safe_generate_response(text)
            audio_response = self.tts.synthesize(response)
            return audio_response
        finally:
            optimize_memory()

    def generate_response(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(inputs.shape, device=self.device)
        
        with torch.no_grad():
            with enable_mixed_precision():
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

    def safe_generate_response(self, text):
        try:
            return self.generate_response(text)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Encountered CUDA out of memory. Trying to clear cache and reduce sequence length...")
                torch.cuda.empty_cache()
                self.model.config.max_length = 50
                return self.generate_response(text)
            raise

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speech_to_text import SpeechToText
from optimizations.gpu_optimizations import GPUAccelerator, optimize_memory, enable_mixed_precision, check_memory

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class Chatbot:
    def __init__(self, stt_model_name, llama_model_name, token):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = GPUAccelerator()

        self.stt = SpeechToText(stt_model_name, self.device)

        # Clear cache before loading the model
        optimize_memory()

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=token)

        try:
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=token, low_cpu_mem_usage=True)
            self.model.to(self.device).half()

            # Enable gradient checkpointing to save memory
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory. Switching to CPU...")
                self.device = torch.device("cpu")
                self.model = self.model.cpu().half()
                self.model = self.accelerator.prepare(self.model)
            else:
                raise e

        self.model.eval()

    def process_audio(self, audio_path):
        try:
            text = self.stt.transcribe(audio_path)
            response = self.safe_generate_response(text)
            return response
        finally:
            optimize_memory()

    def generate_response(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(inputs.shape, device=self.device)

        with torch.no_grad(), enable_mixed_precision():
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
                optimize_memory()
                self.model.config.max_length = 50
                return self.generate_response(text)
            raise

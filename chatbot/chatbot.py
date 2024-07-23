import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from speech_to_text.stt import SpeechToText
from text_to_speech.tts import TextToSpeech
from optimizations.gpu_optimizations import accelerator, optimize_memory
from optimizations.gpu_optimizations import enable_mixed_precision, optimize_memory
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

accelerator = Accelerator()

# Setting environment variables for CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:4096,expandable_segments:True'
os.environ['USE_FP16'] = '1'  # This is not standard and may not affect Accelerator directly but is used in some contexts for clarity.

def process_data():
    with enable_mixed_precision():
        # Process your data using operations that benefit from mixed precision
        pass

    optimize_memory()  # Call after heavy GPU usage
    
class Chatbot:
    def __init__(self, stt_model_name, tts_model_name, llama_model_name, token):
        self.accelerator = Accelerator()  # Correctly initialized without fp16 argument
        
        self.stt = SpeechToText(stt_model_name)
        self.tts = TextToSpeech(tts_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=token, low_cpu_mem_usage=True)
        self.model = self.model.half()  # Manually setting the model to half precision
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
        scaler = GradScaler()

        with torch.no_grad():
            with autocast():
                outputs = scaler.scale(self.model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.tokenizer.eos_token_id
                ))
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def safe_generate_response(self, text):
        """Attempts to generate a response, adjusting parameters if a memory error occurs."""
        try:
            return self.generate_response(text)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Encountered CUDA out of memory. Trying to clear cache and reduce sequence length...")
                torch.cuda.empty_cache()
                # Adjust model parameters or simplify the task as needed
                self.model.max_length = 50  # Example adjustment
                return self.generate_response(text)
            raise  # Re-raise the exception if it's not a memory issue

# Initialize and use the chatbot
# Example initialization: chatbot = Chatbot("stt_model", "tts_model", "llama_model", "token")

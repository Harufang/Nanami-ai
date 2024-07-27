# model_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

class ShardedModel:
    def __init__(self, model_name, device_ids):
        self.device_ids = device_ids
        self.devices = [torch.device(f"cuda:{i}") for i in device_ids]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the full model and shard it
        self.full_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.shard_model()

    def shard_model(self):
        # Shard the model across the available devices
        num_devices = len(self.device_ids)
        self.sharded_models = []
        model_parts = torch.nn.ModuleList(self.full_model.transformer.h).split(num_devices)
        
        for i in range(num_devices):
            model_part = torch.nn.ModuleList(model_parts[i])
            model_part.to(self.devices[i])
            self.sharded_models.append(model_part)

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the sharded models
        outputs = []
        for i, model_part in enumerate(self.sharded_models):
            input_ids = input_ids.to(self.devices[i])
            attention_mask = attention_mask.to(self.devices[i]) if attention_mask is not None else None
            
            output = model_part(input_ids, attention_mask=attention_mask)
            outputs.append(output)
        
        # Combine the outputs from different shards
        return torch.cat([out.logits for out in outputs], dim=-1)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.devices[0])
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs.get('attention_mask'))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

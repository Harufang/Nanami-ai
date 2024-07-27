# model_utils.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class ShardedModel:
    def __init__(self, model_name, device_ids):
        self.device_ids = device_ids
        self.devices = [torch.device(f"cuda:{i}") for i in device_ids]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the full model and shard it
        self.full_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.shard_model()

    def shard_model(self):
        num_devices = len(self.device_ids)
        self.sharded_models = nn.ModuleList()

        # Access model's transformer or encoder layers correctly
        model_layers = self.full_model.get_encoder() if hasattr(self.full_model, 'get_encoder') else self.full_model.base_model
        
        # Check for attribute name adjustments
        if hasattr(model_layers, 'layers'):
            layers = model_layers.layers
        elif hasattr(model_layers, 'transformer') and hasattr(model_layers.transformer, 'h'):
            layers = model_layers.transformer.h
        else:
            raise AttributeError("Model does not have an expected attribute for layers.")
        
        num_layers = len(layers)
        layers_per_device = num_layers // num_devices
        for i in range(num_devices):
            start = i * layers_per_device
            end = (i + 1) * layers_per_device if i != num_devices - 1 else num_layers
            
            device_model = nn.ModuleList(layers[start:end])
            device_model = device_model.to(self.devices[i])  # Ensure model is correctly moved to the GPU
            self.sharded_models.append(device_model)

    def forward(self, input_ids, attention_mask=None):
        outputs = []
        for i, model_part in enumerate(self.sharded_models):
            input_ids = input_ids.to(self.devices[i])
            attention_mask = attention_mask.to(self.devices[i]) if attention_mask is not None else None
            
            output = model_part(input_ids, attention_mask=attention_mask)
            outputs.append(output)
        
        return torch.cat([out.logits for out in outputs], dim=-1)

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.devices[0])
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs.get('attention_mask'))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

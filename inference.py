# inference.py
from model_utils import ShardedModel

def perform_inference(model_name, device_ids, input_text):
    # Initialize the sharded model
    sharded_model = ShardedModel(model_name, device_ids)
    
    # Perform generation
    response = sharded_model.generate(input_text)
    return response

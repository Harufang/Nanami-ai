import os
import torch
from accelerate import Accelerator

# Define environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:4096'

# Initialize Accelerator with mixed precision enabled
accelerator = Accelerator(mixed_precision="fp16")  # Use "fp16" for floating point 16 precision

# Using Accelerator to manage the device setup automatically
device = accelerator.device

def liberer_memoire_gpu():
    """Clear GPU cache."""
    if torch.cuda.is_available():  # Check if CUDA is available
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("CUDA is not available. Cannot clear GPU cache.")

# Import the autocast context manager
from torch.cuda.amp import autocast

# Example usage in a model
def train_model(model, inputs):
    """Prepare model and inputs with Accelerator and perform training."""
    model, inputs = accelerator.prepare(model, inputs)  # Prepare model and inputs
    with autocast():  # Use autocast for the scope of this function
        outputs = model(inputs)
    return outputs

# If you need to use autocast manually in some parts of your code
def some_computation():
    """Perform computation under autocast to utilize mixed precision."""
    with autocast():  # Correctly using autocast without specifying device_type
        # computation logic here
        pass

def clear_cache():
    """Clear cache to free up memory and synchronize."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear cached memory
        torch.cuda.synchronize()  # Synchronize to ensure completion

def optimize_memory():
    """Optimize memory allocation settings."""
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

def enable_mixed_precision():
    """Enable CUDA mixed precision via autocast."""
    return torch.cuda.amp.autocast()

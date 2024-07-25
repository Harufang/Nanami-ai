import os
import torch
from accelerate import Accelerator
from torch.cuda.amp import autocast

# Define environment variables for better GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:4096,expandable_segments:True'

# Initialize Accelerator with mixed precision enabled
accelerator = Accelerator(mixed_precision="fp16")

# Using Accelerator to manage the device setup automatically
device = accelerator.device

def liberer_memoire_gpu():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("CUDA is not available. Cannot clear GPU cache.")

def clear_cache():
    """Clear cache to free up memory and synchronize."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def optimize_memory():
    """Optimize memory allocation settings."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def enable_mixed_precision():
    """Enable CUDA mixed precision via autocast."""
    return autocast()

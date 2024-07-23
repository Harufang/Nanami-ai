import os
import torch
from accelerate import Accelerator

# Définir les variables d'environnement pour la gestion de la mémoire GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:4096'

# Initialiser Accelerator
accelerator = Accelerator()

# Vérifiez si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nettoyer le cache CUDA
def liberer_memoire_gpu():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def enable_mixed_precision():
    return torch.amp.autocast('cuda')

def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def optimize_memory():
    clear_cache()

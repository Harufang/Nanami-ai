from interface.gradio_interface import main
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

if __name__ == "__main__":
    main()

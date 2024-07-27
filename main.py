# main.py
from inference import perform_inference

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your model name
    device_ids = [0, 1]  # Example device IDs for GPUs
    input_text = "Hello, how are you?"

    # Perform inference
    response = perform_inference(model_name, device_ids, input_text)
    print(response)

if __name__ == "__main__":
    main()

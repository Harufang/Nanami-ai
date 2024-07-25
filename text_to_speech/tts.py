import requests

class TextToSpeech:
    def __init__(self, model_name, token):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}

    def synthesize(self, text):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
        if response.status_code == 200:
            return response.content
        else:
            raise RuntimeError(f"Error during TTS synthesis: {response.text}")

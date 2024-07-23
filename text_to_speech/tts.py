import requests

class TextToSpeech:
    def __init__(self, model_name, token):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}
        
    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Error querying the API: {response.status_code} {response.text}")
        return response.content

    def synthesize(self, text):
        try:
            audio_bytes = self.query({"inputs": text})
            return audio_bytes
        except Exception as e:
            raise RuntimeError(f"Error during synthesis: {e}")

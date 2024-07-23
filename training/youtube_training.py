from pytube import YouTube
import os

class YouTubeTraining:
    def __init__(self, download_path):
        self.download_path = download_path

    def download_video(self, url):
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        output_path = stream.download(output_path=self.download_path)
        return output_path

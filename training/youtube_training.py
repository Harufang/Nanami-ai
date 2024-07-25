from pytube import YouTube
import os

class YouTubeTraining:
    def __init__(self, download_dir):
        self.download_dir = download_dir

    def download_video(self, url):
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=self.download_dir)
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)
        return new_file

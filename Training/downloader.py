import os
from yt_dlp import YoutubeDL

os.makedirs("videos", exist_ok=True)

# Search queries for downloading training videos from YouTube
queries = [
    "dashcam driving city",
    "walking street pedestrians",
    "gopro walking tour",
    "downtown walking tour",
    "city street crowd walking"
]

MAX_SIZE = 1 * 1024 * 1024 * 1024  # 1GB limit per video

def size_filter(info):
    # Filter videos by file size to avoid extremely large downloads
    size = info.get("filesize") or info.get("filesize_approx")
    
    if size is None:
        return None
    
    if size > MAX_SIZE:
        return "File too large"
    
    return None

# Configure download options: 720p max, skip playlists, apply size filter
ydl_opts = {
    "format": "bestvideo[height<=720]",
    "outtmpl": "videos/%(id)s.%(ext)s",
    "ignoreerrors": True,
    "quiet": False,
    "noplaylist": True,
    "match_filter": size_filter,
}

def download(query):
    # Search for top 10 videos matching the query and download
    url = f"ytsearch10:{query}"
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Download videos for all search queries
for q in queries:
    print("Downloading:", q)
    download(q)

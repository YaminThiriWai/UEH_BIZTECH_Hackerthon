from huggingface_hub import snapshot_download
#import whisper
# Download the model and store it locally
snapshot_download(
    repo_id="j-hartmann/emotion-english-distilroberta-base",
    local_dir="local_model",
    local_dir_use_symlinks=False
)

#model = whisper.load_model("base", download_root="./models/whisper")

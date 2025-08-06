## AI Models Required

This project includes two backend Python files:

* `speech_api.py`
* `facial_emotion_api.py`

Each of these uses pretrained AI models. Please read the model requirements below.

---

### `speech_api.py` Requirements

1. **Whisper Model** (`whisper_model/base.pt`)

   * This model is **not included** in the repository due to its large file size.
   * You can generate it using the provided script:

     ```bash
     python export_model.py
     ```
   * Alternatively, download it manually using Whisper:

     ```python
     import whisper
     model = whisper.load_model("base", download_root="./whisper_model")
     ```

2. **Emotion Detection Model**:

   * `emotion_english_distilroberta_model`
   * Already included in the repository (downloaded via Hugging Face).
   * No action needed.

---

### `facial_emotion_api.py` Requirements

1. **Facial Emotion Model**:

   * `model_file_30epochs.h5`
   * Already included in the repository.

2. **Haar Cascade File**:

   * `haarcascade_frontalface_default.xml`
   * Already included in the repository.


# SpeakSense AI – Speak Better. Present Smarter.

AI-powered presentation performance analyzer.

| Signal | Tech |
|---|---|
| Speech (WPM, pauses, fillers) | faster-whisper (Python) |
| Vision (eye contact, posture) | MediaPipe Face Landmarker (in browser) |
| NLP (topic relevance) | TF-IDF cosine similarity + keyword coverage (scikit-learn) |

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Terminal > New Terminal, then:
   ```
   python -m venv .venv
   .venv\Scripts\activate        (Windows)   |   source .venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   python app.py
   ```
3. Open http://127.0.0.1:5000 in Chrome or Edge and allow camera + microphone.

First run downloads the Whisper "base" model (~150 MB) and needs internet. The vision model also loads from the internet.
Use a bigger model with `WHISPER_MODEL=small python app.py`.

If the server is down, the page still works using in-browser speech recognition.

## Structure
- `app.py` – Flask server and `/api/analyze` (Whisper + TF-IDF)
- `static/index.html` – UI, live analysis, MediaPipe vision, report dashboard

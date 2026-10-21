"""SpeakSense AI backend: faster-whisper (speech) + TF-IDF (content).
Vision (MediaPipe face tracking) runs in the browser and is sent along with the audio."""
import json, os, re, tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static", static_url_path="")
print("Loading Whisper model (first run downloads it)...")
model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")

FILLERS = ["um", "umm", "uh", "uhh", "like", "basically", "actually", "literally",
           "you know", "i mean", "kind of", "sort of"]
REFERENCE = {
    "ai": "artificial intelligence machine learning model trained on data neural network deep learning "
          "algorithm prediction classification supervised unsupervised dataset training natural language "
          "processing computer vision",
    "dbms": "database management system relational table schema sql query transaction acid normalization "
            "index primary key foreign key join record data storage concurrency",
}

def clamp(n, lo=0, hi=100):
    return max(lo, min(hi, n))

@app.get("/")
def home():
    return app.send_static_file("index.html")

@app.post("/api/analyze")
def analyze():
    f = request.files.get("audio")
    if not f:
        return jsonify(error="no audio"), 400
    kws = json.loads(request.form.get("keywords", "[]"))
    topic = request.form.get("topic", "custom")

    fd, path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    f.save(path)
    try:
        segments, _ = model.transcribe(
            path, language="en", word_timestamps=True, vad_filter=True,
            initial_prompt="Umm, uh, so, like, you know, I mean, basically, actually.")
        words = [w for s in segments for w in (s.words or [])]
    finally:
        os.remove(path)

    text = " ".join(w.word.strip() for w in words)
    toks = re.findall(r"[a-z']+", text.lower())
    n = len(toks)
    if n < 3:
        return jsonify(error="not enough speech"), 422

    # Speech: pace + long pauses
    span = max(1.0, words[-1].end - words[0].start)
    wpm = n / (span / 60)
    pauses = sum(1 for a, b in zip(words, words[1:]) if b.start - a.end > 1.2)
    pace = 100 if 110 <= wpm <= 160 else 100 - min(70, (110 - wpm if wpm < 110 else wpm - 160) * 1.2)
    speech = clamp(pace - pauses * 4)

    # Fillers
    joined = " " + " ".join(toks) + " "
    fillers = sum(joined.count(f" {f} ") for f in FILLERS)
    filler = clamp(100 - (fillers / n * 100) * 9)

    # Content: TF-IDF cosine similarity to topic reference + keyword coverage
    ref = REFERENCE.get(topic) or " ".join(kws)
    try:
        vec = TfidfVectorizer(stop_words="english").fit(list(REFERENCE.values()) + [ref, text])
        sim = float(cosine_similarity(vec.transform([ref]), vec.transform([text]))[0][0])
    except ValueError:
        sim = 0.0
    hits = sum(1 for k in kws if k in set(toks))
    target = min(8, max(1, -(-len(kws) // 2)))
    content = clamp(60 * min(1, sim / 0.3) + 40 * min(1, hits / target))

    return jsonify(speech=speech, filler=filler, content=content, wpm=wpm, fillers=fillers,
                   pauses=pauses, words=n, hits=hits, total=len(kws), transcript=text)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

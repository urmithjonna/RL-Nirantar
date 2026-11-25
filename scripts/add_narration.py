"""Adds a spoken English narration (Gnani Timbre text-to-speech) to Report/yuva_yodha_rl_simulation.mp4.

Every line describes what the recorded rollout (results/rl_simulation_rollout.csv) shows at that moment of the video.
Lines are voiced once, placed at their target time (never overlapping), mixed into one track and muxed with the video.
The API key is read from the GNANI_API_KEY environment variable or from --key-file (a file holding the key); it is never stored.

Run:  .venv\\Scripts\\python.exe scripts\\add_narration.py --key-file path\\to\\gnani.env
"""
import argparse, io, json, os, shutil, subprocess, tempfile, time, wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "Report" / "yuva_yodha_rl_simulation.mp4"
LINES_JSON = ROOT / "results" / "rl_simulation_narration.json"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
MODEL, VOICE, LANG, RATE, SPEED = "timbre-v3.0", "Kalpana", "en-IN", 24000, 1.1
DURATION, GAP = 50.0, 0.35

# (earliest start in the video, seconds; text) - video times come from the frame timeline of generate_rl_simulation.py
LINES = [
    (0.3, "Nirantar: a reinforcement learning controller for a battery shared by ten homes with frequent power cuts."),
    (6.6, "Every fifteen minutes, the policy picks one of ten battery actions, and a safety shield checks it."),
    (13.6, "Morning: homes start losing supply, and solar and the battery back them up."),
    (24.2, "Midday: solar surplus, plus some grid power, fills the battery by two p.m."),
    (33.4, "Evening peak: the battery powers the street, including the homes in a cut."),
    (40.2, "At nine p.m., the shield keeps a four-hour reserve."),
    (44.4, "Over the day, every home in a cut was served."),
]


def load_key(path):
    if os.environ.get("GNANI_API_KEY"):
        return os.environ["GNANI_API_KEY"].strip()
    txt = Path(path).read_text(encoding="utf-8").strip()
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            return ln.split("=", 1)[1].strip().strip('"') if "=" in ln else ln
    raise SystemExit("no key found")


def speak(session, key, text):
    body = {"text": text, "voice": VOICE, "model": MODEL, "language": LANG, "speed": SPEED,
            "audio_config": {"sample_rate": RATE, "num_channels": 1, "sample_width": 2, "encoding": "linear_pcm", "container": "wav"}}
    for attempt in range(5):
        r = session.post(TTS_URL, headers={"X-API-Key-ID": key, "Content-Type": "application/json"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(15 * (attempt + 1))
    if r.status_code >= 400:
        raise SystemExit(f"Timbre {r.status_code}: {r.text[:200]}")
    b = r.content
    with wave.open(io.BytesIO(b)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE), "unexpected audio format"
    pcm = b[b.find(b"data") + 8:]                       # streamed WAV: header lengths are placeholders
    return np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key-file", default=""); a = ap.parse_args()
    key = load_key(a.key_file); http = requests.Session()
    track = np.zeros(int(DURATION * RATE), np.float32); t_free = 0.0; placed = []
    for target, text in LINES:
        clip = speak(http, key, text)
        clip = clip / max(np.abs(clip).max(), 1) * 0.89 * 32767           # peak-normalise
        start = max(target, t_free); end = start + len(clip) / RATE
        if end > DURATION - 0.2:
            raise SystemExit(f"line at {target}s ends at {end:.1f}s, past the video: shorten it")
        i = int(start * RATE); track[i:i + len(clip)] += clip; t_free = end + GAP
        placed.append({"start_s": round(start, 2), "end_s": round(end, 2), "text": text})
        print(f"{start:5.1f}-{end:5.1f}s  {text}")
    tmp = Path(tempfile.mkdtemp(prefix="narration_"))
    try:
        wav = tmp / "narration.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(np.clip(track, -32768, 32767).astype(np.int16).tobytes())
        silent = tmp / "silent.mp4"; shutil.copy2(VIDEO, silent)
        out = tmp / "narrated.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-t", str(DURATION), "-movflags", "+faststart", str(out)], check=True)
        shutil.copy2(out, VIDEO)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    LINES_JSON.write_text(json.dumps({"voice": f"Gnani Timbre {MODEL} · {VOICE} · {LANG}", "lines": placed}, indent=1), encoding="utf-8")
    print("narrated video ->", VIDEO)


if __name__ == "__main__":
    main()

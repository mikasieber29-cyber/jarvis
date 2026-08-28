#!/usr/bin/env python3
"""
JARVIS Brücke — verbindet das Browser-Interface mit Hermes und Helmut.

Ablauf pro Gespräch:
  Browser nimmt Stimme auf  →  POST /api/talk (webm-Audio)
  → faster-whisper (lokal, Deutsch) macht Text daraus
  → Hermes-API (127.0.0.1:8642) denkt und antwortet
  → ElevenLabs macht Helmuts Stimme daraus
  → JSON zurück an den Browser: {transcript, reply, audio_b64}

Starten mit dem Hermes-venv-Python:
  ~/.hermes/hermes-agent/venv/bin/python jarvis_server.py
Dann im Chrome: http://localhost:8765
"""

import base64
import json
import os
import sys
import tempfile
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- Einstellungen

HOME = os.path.expanduser("~")
ENV_PATH = os.path.join(HOME, ".hermes", ".env")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8765

HERMES_URL = "http://127.0.0.1:8642/v1/chat/completions"
ELEVEN_VOICE_ID = "g1jpii0iyvtRs8fqXsd1"          # Helmut
ELEVEN_MODEL = "eleven_multilingual_v2"
WHISPER_MODEL = "base"                              # später: "small" für bessere Erkennung
WHISPER_LANG = "de"

SYSTEM_HINT = (
    "Du bist JARVIS und sprichst gerade PER SPRACHE mit Mika über sein "
    "Iron-Man-Interface. Antworte auf Deutsch, nenn ihn Chef, halte dich "
    "kurz (2-3 Sätze), ausser er will Details. Keine Markdown-Zeichen, "
    "keine Aufzählungszeichen — deine Antwort wird laut vorgelesen."
)


def read_env(path):
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    vals[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return vals


ENV = read_env(ENV_PATH)
ELEVEN_KEY = ENV.get("ELEVENLABS_API_KEY", "")
HERMES_KEY = ENV.get("API_SERVER_KEY", "")

if not ELEVEN_KEY:
    print("WARNUNG: kein ELEVENLABS_API_KEY in ~/.hermes/.env — Helmut bleibt stumm.")
if not HERMES_KEY:
    print("WARNUNG: kein API_SERVER_KEY in ~/.hermes/.env — Hermes-Anschluss fehlt.")

# ---------------------------------------------------------------- Spracherkennung

_whisper = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            print(f"Lade Spracherkennung ({WHISPER_MODEL}, Deutsch) …")
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            print("Spracherkennung bereit.")
        return _whisper


def transcribe(audio_path):
    model = get_whisper()
    segments, _info = model.transcribe(
        audio_path, language=WHISPER_LANG, beam_size=5, vad_filter=True
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------- Hermes & Helmut

def ask_hermes(text):
    payload = json.dumps({
        "model": "hermes-agent",
        "messages": [
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": text},
        ],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        HERMES_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HERMES_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def helmut_speaks(text):
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
        "?output_format=mp3_44100_128"
    )
    payload = json.dumps({"text": text, "model_id": ELEVEN_MODEL}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "xi-api-key": ELEVEN_KEY},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ---------------------------------------------------------------- HTTP-Server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("  " + fmt % args)

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/app.html"):
            try:
                with open(os.path.join(APP_DIR, "app.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"app.html fehlt neben jarvis_server.py", "text/plain")
        elif self.path == "/api/health":
            ok = {"server": True, "elevenlabs": bool(ELEVEN_KEY), "hermes_key": bool(HERMES_KEY)}
            self._send(200, json.dumps(ok).encode())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/talk":
            self._send(404, b"not found", "text/plain")
            return

        length = int(self.headers.get("Content-Length", "0"))
        audio = self.rfile.read(length)
        step = "aufnahme"
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tf:
                tf.write(audio)
                tmp = tf.name
            try:
                step = "spracherkennung"
                text = transcribe(tmp)
            finally:
                os.unlink(tmp)

            if not text:
                self._send(200, json.dumps({
                    "transcript": "", "reply": "", "audio_b64": "",
                    "note": "nichts verstanden"
                }).encode())
                return

            print(f"  Chef sagte: {text}")
            step = "hermes"
            reply = ask_hermes(text)
            print(f"  Jarvis: {reply[:120]}")

            audio_b64 = ""
            if ELEVEN_KEY:
                step = "helmut"
                try:
                    audio_b64 = base64.b64encode(helmut_speaks(reply)).decode()
                except Exception as e:
                    print(f"  Helmut-Fehler (Antwort kommt als Text): {e}")

            self._send(200, json.dumps({
                "transcript": text, "reply": reply, "audio_b64": audio_b64
            }).encode())

        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            print(f"  FEHLER bei {step}: HTTP {e.code} — {detail}")
            self._send(502, json.dumps({"error": step, "detail": f"HTTP {e.code}: {detail}"}).encode())
        except Exception as e:
            print(f"  FEHLER bei {step}: {e}")
            self._send(500, json.dumps({"error": step, "detail": str(e)}).encode())


def main():
    print("=" * 56)
    print("  JARVIS Brücke — http://localhost:%d" % PORT)
    print("  Hermes: %s" % HERMES_URL)
    print("  Stimme: Helmut (%s)" % ELEVEN_VOICE_ID)
    print("=" * 56)
    # Spracherkennung im Hintergrund vorladen, damit das erste Gespräch flott ist
    threading.Thread(target=get_whisper, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

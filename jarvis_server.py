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
import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
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
ELEVEN_MODEL = "eleven_flash_v2_5"   # schnelles Gesprächs-Modell; für Studio-Qualität: eleven_multilingual_v2
WHISPER_MODEL = "base"                              # später: "small" für bessere Erkennung
WHISPER_LANG = "de"

SYSTEM_HINT = (
    "Du bist JARVIS und sprichst gerade PER SPRACHE mit Mika über sein "
    "Iron-Man-Interface. Antworte auf Deutsch, nenn ihn Chef, halte dich "
    "kurz (2-3 Sätze), ausser er will Details. Keine Markdown-Zeichen, "
    "keine Aufzählungszeichen — deine Antwort wird laut vorgelesen."
)

TEXT_HINT = (
    "Du bist JARVIS und Mika (nenn ihn Chef) SCHREIBT dir gerade über sein "
    "Interface. Antworte auf Deutsch, kurz und klar — deine Antwort wird "
    "gelesen, nicht vorgelesen. Keine Markdown-Zeichen, keine Sternchen."
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

def ask_hermes(text, hint=None):
    payload = json.dumps({
        "model": "hermes-agent",
        "messages": [
            {"role": "system", "content": hint or SYSTEM_HINT},
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


# ---------------------------------------------------------------- Dashboard-Daten

GOOGLE_TOKEN = os.path.join(HOME, ".hermes", "google_token.json")
CITY = "Zürich"          # fürs Wetter — auf Wunsch anpassen
_geo = {}
_dash = {"t": 0, "data": None}
_rem = {"t": 0, "data": None}


def google_creds():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return creds
    except Exception as e:
        print("  google_creds:", e)
        return None


def gapi(creds, url):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + creds.token})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def fetch_mails(creds):
    base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    lst = gapi(creds, base + "?maxResults=6&labelIds=INBOX")
    out = []
    for m in lst.get("messages", [])[:6]:
        d = gapi(creds, base + "/" + m["id"] +
                 "?format=metadata&metadataHeaders=From&metadataHeaders=Subject")
        hdr = {h["name"]: h["value"] for h in d.get("payload", {}).get("headers", [])}
        sender = hdr.get("From", "?").split("<")[0].strip().strip('"') or hdr.get("From", "?")
        ts = int(d.get("internalDate", "0")) / 1000
        out.append({
            "from": sender[:34],
            "subject": hdr.get("Subject", "(kein Betreff)")[:60],
            "unread": "UNREAD" in d.get("labelIds", []),
            "time": datetime.datetime.fromtimestamp(ts).astimezone().strftime("%H:%M") if ts else "",
        })
    return out


def fetch_events(creds):
    now = datetime.datetime.now().astimezone()
    t0 = urllib.parse.quote(now.isoformat())
    t1 = urllib.parse.quote((now + datetime.timedelta(hours=40)).isoformat())
    url = ("https://www.googleapis.com/calendar/v3/calendars/primary/events"
           f"?timeMin={t0}&timeMax={t1}&singleEvents=true&orderBy=startTime&maxResults=8")
    out = []
    for ev in gapi(creds, url).get("items", []):
        st = ev.get("start", {})
        if "dateTime" in st:
            dt = datetime.datetime.fromisoformat(st["dateTime"]).astimezone()
            when, allday = dt.strftime("%H:%M"), False
        else:
            dt = datetime.datetime.fromisoformat(st.get("date", now.date().isoformat())).astimezone()
            when, allday = "ganztags", True
        day = "heute" if dt.date() == now.date() else "morgen"
        out.append({"title": ev.get("summary", "(ohne Titel)")[:44], "when": when,
                    "day": day, "allday": allday})
    return out


def fetch_weather():
    global _geo
    if not _geo:
        g = json.loads(urllib.request.urlopen(
            "https://geocoding-api.open-meteo.com/v1/search?count=1&language=de&name="
            + urllib.parse.quote(CITY), timeout=10).read())
        r = g["results"][0]
        _geo = {"lat": r["latitude"], "lon": r["longitude"], "name": r["name"]}
    w = json.loads(urllib.request.urlopen(
        f"https://api.open-meteo.com/v1/forecast?latitude={_geo['lat']}"
        f"&longitude={_geo['lon']}&current_weather=true"
        "&daily=temperature_2m_max,temperature_2m_min&timezone=auto", timeout=10).read())
    code = w["current_weather"]["weathercode"]
    txt = ("klar" if code == 0 else "leicht bewölkt" if code <= 2 else "bewölkt" if code == 3
           else "Nebel" if code in (45, 48) else "Regen" if code < 70 else "Schnee" if code < 80
           else "Schauer" if code < 90 else "Gewitter")
    return {"city": _geo["name"], "temp": round(w["current_weather"]["temperature"]),
            "desc": txt,
            "max": round(w["daily"]["temperature_2m_max"][0]),
            "min": round(w["daily"]["temperature_2m_min"][0])}


def fetch_reminders():
    if time.time() - _rem["t"] < 300 and _rem["data"] is not None:
        return _rem["data"]
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "Reminders" to get name of every reminder whose completed is false'],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:120] or "osascript-Fehler")
        names = [n.strip() for n in r.stdout.strip().split(",") if n.strip()]
        _rem.update(t=time.time(), data=names[:6])
    except Exception as e:
        _rem.update(t=time.time(), data={"error": str(e)[:120]})
    return _rem["data"]


def build_dashboard():
    if time.time() - _dash["t"] < 60 and _dash["data"]:
        return _dash["data"]
    data = {}
    creds = google_creds()
    for key, fn in (("mails", lambda: fetch_mails(creds)),
                    ("events", lambda: fetch_events(creds))):
        try:
            data[key] = fn() if creds else {"error": "Google-Zugang fehlt"}
        except Exception as e:
            data[key] = {"error": str(e)[:120]}
    try:
        data["weather"] = fetch_weather()
    except Exception as e:
        data["weather"] = {"error": str(e)[:120]}
    data["reminders"] = fetch_reminders()
    _dash.update(t=time.time(), data=data)
    return data


# ---------------------------------------------------------------- Unterseiten (Posteingang, Woche, Aufgaben)

_mails = {"t": 0, "data": None}
_week = {"t": 0, "data": None}
_remall = {"t": 0, "data": None}
_bodies = {}

WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def _when_label(ts, now):
    d = datetime.datetime.fromtimestamp(ts).astimezone()
    if d.date() == now.date():
        return d.strftime("%H:%M")
    if d.date() == (now - datetime.timedelta(days=1)).date():
        return "Gestern"
    return d.strftime("%d.%m.")


def fetch_mail_list(creds, n=30):
    if time.time() - _mails["t"] < 60 and _mails["data"] is not None:
        return _mails["data"]
    base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    lst = gapi(creds, base + f"?maxResults={n}&labelIds=INBOX")
    now = datetime.datetime.now().astimezone()
    out = []
    for m in lst.get("messages", [])[:n]:
        d = gapi(creds, base + "/" + m["id"] +
                 "?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date")
        hdr = {h["name"]: h["value"] for h in d.get("payload", {}).get("headers", [])}
        raw_from = hdr.get("From", "?")
        sender = raw_from.split("<")[0].strip().strip('"') or raw_from
        ts = int(d.get("internalDate", "0")) / 1000
        out.append({
            "id": m["id"],
            "from": sender[:40],
            "email": raw_from,
            "subject": hdr.get("Subject", "(kein Betreff)")[:120],
            "snippet": d.get("snippet", "")[:140],
            "unread": "UNREAD" in d.get("labelIds", []),
            "time": _when_label(ts, now) if ts else "",
            "date": datetime.datetime.fromtimestamp(ts).astimezone().strftime("%a %d.%m.%Y %H:%M") if ts else "",
        })
    _mails.update(t=time.time(), data=out)
    return out


def _b64url(data):
    data = data.replace("-", "+").replace("_", "/")
    data += "=" * (-len(data) % 4)
    return base64.b64decode(data).decode("utf-8", "replace")


def _html_to_text(html):
    import re, html as h
    html = re.sub(r"(?is)<(style|script|head)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h[1-6]>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    txt = h.unescape(html)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
    return txt.strip()


def _walk_parts(part, found):
    mime = part.get("mimeType", "")
    body = part.get("body", {}).get("data")
    if body and mime in ("text/plain", "text/html"):
        found.setdefault(mime, _b64url(body))
    for p in part.get("parts", []) or []:
        _walk_parts(p, found)


def fetch_mail_body(creds, mid):
    if mid in _bodies:
        return _bodies[mid]
    base = "https://gmail.googleapis.com/gmail/v1/users/me/messages/" + mid + "?format=full"
    d = gapi(creds, base)
    hdr = {h["name"]: h["value"] for h in d.get("payload", {}).get("headers", [])}
    found = {}
    _walk_parts(d.get("payload", {}), found)
    text = found.get("text/plain") or (_html_to_text(found["text/html"]) if "text/html" in found else d.get("snippet", ""))
    ts = int(d.get("internalDate", "0")) / 1000
    out = {
        "id": mid,
        "from": hdr.get("From", "?"),
        "to": hdr.get("To", ""),
        "subject": hdr.get("Subject", "(kein Betreff)"),
        "date": datetime.datetime.fromtimestamp(ts).astimezone().strftime("%A, %d. %B %Y, %H:%M") if ts else "",
        "body": text[:30000],
    }
    if len(_bodies) > 60:
        _bodies.clear()
    _bodies[mid] = out
    return out


def fetch_week(creds, days=7):
    if time.time() - _week["t"] < 120 and _week["data"] is not None:
        return _week["data"]
    now = datetime.datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t0 = urllib.parse.quote(start.isoformat())
    t1 = urllib.parse.quote((start + datetime.timedelta(days=days)).isoformat())
    url = ("https://www.googleapis.com/calendar/v3/calendars/primary/events"
           f"?timeMin={t0}&timeMax={t1}&singleEvents=true&orderBy=startTime&maxResults=60")
    daysout = []
    byday = {}
    for i in range(days):
        d = (start + datetime.timedelta(days=i)).date()
        label = "Heute" if i == 0 else "Morgen" if i == 1 else WOCHENTAGE[d.weekday()]
        entry = {"date": d.isoformat(), "label": label, "sub": d.strftime("%d.%m."), "events": []}
        daysout.append(entry)
        byday[d.isoformat()] = entry
    for ev in gapi(creds, url).get("items", []):
        st, en = ev.get("start", {}), ev.get("end", {})
        if "dateTime" in st:
            dt = datetime.datetime.fromisoformat(st["dateTime"]).astimezone()
            et = datetime.datetime.fromisoformat(en["dateTime"]).astimezone() if "dateTime" in en else None
            when, allday = dt.strftime("%H:%M"), False
            until = et.strftime("%H:%M") if et else ""
            key = dt.date().isoformat()
        else:
            dt = datetime.datetime.fromisoformat(st.get("date", start.date().isoformat()))
            when, allday, until = "ganztags", True, ""
            key = dt.date().isoformat()
        item = {"title": ev.get("summary", "(ohne Titel)")[:80], "when": when, "until": until,
                "allday": allday, "location": (ev.get("location") or "")[:60],
                "past": (not allday) and dt < now}
        if key in byday:
            byday[key]["events"].append(item)
    _week.update(t=time.time(), data=daysout)
    return daysout


def fetch_reminders_all():
    if time.time() - _remall["t"] < 300 and _remall["data"] is not None:
        return _remall["data"]
    script = '''
set out to ""
tell application "Reminders"
  repeat with l in lists
    set ln to name of l
    repeat with r in (reminders of l whose completed is false)
      set dd to ""
      try
        set dd to (due date of r) as string
      end try
      set out to out & ln & tab & (name of r) & tab & dd & linefeed
    end repeat
  end repeat
end tell
return out
'''
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:120] or "osascript-Fehler")
        lists = {}
        for line in r.stdout.split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            ln = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            due = parts[2].strip() if len(parts) > 2 else ""
            if name:
                lists.setdefault(ln, []).append({"name": name, "due": due})
        data = [{"list": k, "items": v} for k, v in lists.items()]
        _remall.update(t=time.time(), data=data)
    except Exception as e:
        _remall.update(t=time.time(), data={"error": str(e)[:120]})
    return _remall["data"]


def _as_str(v):
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def complete_reminder(list_name, name):
    """Hakt eine Erinnerung in Apple Erinnerungen ab (completed = true)."""
    script = (
        'tell application "Reminders"\n'
        '  set rs to (' + ('reminders of list ' + _as_str(list_name) if list_name else 'reminders') + ' whose name is ' + _as_str(name) + ' and completed is false)\n'
        '  if (count of rs) > 0 then\n'
        '    set completed of item 1 of rs to true\n'
        '    return "ok"\n'
        '  end if\n'
        '  return "notfound"\n'
        'end tell'
    )
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:160] or "osascript-Fehler")
    ok = r.stdout.strip() == "ok"
    # Caches leeren, damit Übersicht und Aufgaben-Seite sofort stimmen
    _remall.update(t=0, data=None)
    _rem.update(t=0, data=None)
    _dash.update(t=0, data=None)
    return {"ok": ok, "error": None if ok else "Erinnerung nicht gefunden (schon erledigt?)"}


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

    def _json_call(self, fn):
        creds = google_creds()
        if not creds:
            self._send(200, json.dumps({"error": "Google-Zugang fehlt"}).encode())
            return
        try:
            self._send(200, json.dumps(fn(creds)).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)[:200]}).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html", "/app.html"):
            try:
                with open(os.path.join(APP_DIR, "app.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"app.html fehlt neben jarvis_server.py", "text/plain")
        elif self.path == "/api/dashboard":
            try:
                self._send(200, json.dumps(build_dashboard()).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
        elif self.path.startswith("/api/mails"):
            self._json_call(lambda c: fetch_mail_list(c))
        elif self.path.startswith("/api/mail?"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            mid = (q.get("id") or [""])[0]
            self._json_call(lambda c: fetch_mail_body(c, mid))
        elif self.path.startswith("/api/events"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            days = max(1, min(14, int((q.get("days") or ["7"])[0])))
            self._json_call(lambda c: fetch_week(c, days))
        elif self.path.startswith("/api/reminders"):
            try:
                self._send(200, json.dumps(fetch_reminders_all()).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
        elif self.path == "/api/health":
            ok = {"server": True, "elevenlabs": bool(ELEVEN_KEY), "hermes_key": bool(HERMES_KEY)}
            self._send(200, json.dumps(ok).encode())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/ask":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                text = (body.get("text") or "").strip()
                if not text:
                    self._send(400, json.dumps({"error": "leere Frage"}).encode())
                    return
                reply = ask_hermes(text, TEXT_HINT)
                self._send(200, json.dumps({"reply": reply}).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
            return
        if self.path == "/api/reminder/done":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                res = complete_reminder(body.get("list", ""), body.get("name", ""))
                self._send(200, json.dumps(res).encode())
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "error": str(e)[:200]}).encode())
            return
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
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        print("  Vom iPhone (gleiches WLAN): http://%s:%d" % (s.getsockname()[0], PORT))
        s.close()
    except Exception:
        pass
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

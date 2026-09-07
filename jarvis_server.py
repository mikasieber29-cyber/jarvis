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
import re
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


def helmut_speaks(text, voice_id=None):
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id or ELEVEN_VOICE_ID}"
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
WORK_START, WORK_END = 8, 18     # Zeitfenster, in dem freie Lücken gesucht werden
GAP_MIN = 45                     # kürzere Lücken als das interessieren nicht
WAIT_DAYS = 2                    # Mails, die so lange auf Antwort warten
FOLLOW_DAYS = 7                  # eigene Mails ohne Antwort seit so vielen Tagen
_geo = {}
_dash = {"t": 0, "data": None}
_rem = {"t": 0, "data": None}
_follow = {"t": 0, "data": None}
_quota = {"t": 0, "data": None}


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
                    "day": day, "allday": allday, "iso": dt.isoformat()})
    return out


def free_gaps(creds, days=2):
    """Freie Blöcke von mindestens GAP_MIN Minuten innerhalb der Arbeitszeit."""
    now = datetime.datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t0 = urllib.parse.quote(start.isoformat())
    t1 = urllib.parse.quote((start + datetime.timedelta(days=days)).isoformat())
    url = ("https://www.googleapis.com/calendar/v3/calendars/primary/events"
           f"?timeMin={t0}&timeMax={t1}&singleEvents=true&orderBy=startTime&maxResults=60")
    busy = []
    for ev in gapi(creds, url).get("items", []):
        st, en = ev.get("start", {}), ev.get("end", {})
        if "dateTime" not in st or "dateTime" not in en:
            continue                                    # ganztägige Einträge blockieren nicht
        if (ev.get("transparency") == "transparent"):
            continue                                    # "frei" markierte Termine zählen nicht
        busy.append((datetime.datetime.fromisoformat(st["dateTime"]).astimezone(),
                     datetime.datetime.fromisoformat(en["dateTime"]).astimezone()))
    busy.sort()
    out = []
    for d in range(days):
        day0 = (start + datetime.timedelta(days=d)).replace(hour=WORK_START)
        day1 = (start + datetime.timedelta(days=d)).replace(hour=WORK_END)
        cur = max(day0, now) if d == 0 else day0
        if cur >= day1:
            continue
        for b0, b1 in busy:
            if b1 <= cur or b0 >= day1:
                continue
            if b0 - cur >= datetime.timedelta(minutes=GAP_MIN):
                out.append((cur, min(b0, day1)))
            cur = max(cur, b1)
        if day1 - cur >= datetime.timedelta(minutes=GAP_MIN):
            out.append((cur, day1))
    return [{"day": "heute" if a.date() == now.date() else "morgen",
             "from": a.strftime("%H:%M"), "to": b.strftime("%H:%M"),
             "min": int((b - a).total_seconds() // 60)} for a, b in out[:6]]


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
        "&daily=temperature_2m_max,temperature_2m_min"
        "&hourly=precipitation_probability&forecast_days=2&timezone=auto", timeout=10).read())
    code = w["current_weather"]["weathercode"]
    txt = ("klar" if code == 0 else "leicht bewölkt" if code <= 2 else "bewölkt" if code == 3
           else "Nebel" if code in (45, 48) else "Regen" if code < 70 else "Schnee" if code < 80
           else "Schauer" if code < 90 else "Gewitter")
    rain = None
    try:
        now = datetime.datetime.now()
        times = w["hourly"]["time"]
        probs = w["hourly"]["precipitation_probability"]
        for tstr, pr in zip(times, probs):
            h = datetime.datetime.fromisoformat(tstr)
            if h < now or (h - now).total_seconds() > 12 * 3600:
                continue
            if pr is not None and pr >= 60:
                mins = int((h - now).total_seconds() // 60)
                rain = {"at": h.strftime("%H:%M"), "prob": pr, "in_min": mins}
                break
    except Exception:
        rain = None
    return {"city": _geo["name"], "temp": round(w["current_weather"]["temperature"]),
            "desc": txt, "rain": rain,
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
    try:
        data["gaps"] = free_gaps(creds) if creds else {"error": "Google-Zugang fehlt"}
    except Exception as e:
        data["gaps"] = {"error": str(e)[:120]}
    data["reminders"] = fetch_reminders()
    data["quota"] = fetch_quota()
    _dash.update(t=time.time(), data=data)
    return data




# ---------------------------------------------------------------- Länder auf dem Globus

# Name -> (Breite, Länge, Schreibweisen die Mika benutzen könnte)
# Die Koordinaten sind grobe Mittelpunkte — sie bestimmen, wohin sich die Erde dreht.
LAENDER = {
    "Schweiz":         (46.8,   8.2, ["schweiz", "helvetia", "eidgenossenschaft"]),
    "Deutschland":     (51.2,  10.4, ["deutschland", "brd"]),
    "Österreich":      (47.6,  14.1, ["österreich", "oesterreich"]),
    "Italien":         (42.8,  12.6, ["italien", "italy"]),
    "Frankreich":      (46.6,   2.4, ["frankreich", "france"]),
    "Spanien":         (40.2,  -3.7, ["spanien"]),
    "Portugal":        (39.5,  -8.0, ["portugal"]),
    "Grossbritannien": (54.0,  -2.5, ["grossbritannien", "großbritannien", "england", "uk", "britannien", "london"]),
    "Irland":          (53.2,  -8.0, ["irland"]),
    "Niederlande":     (52.2,   5.5, ["niederlande", "holland"]),
    "Belgien":         (50.6,   4.6, ["belgien"]),
    "Liechtenstein":   (47.2,   9.6, ["liechtenstein"]),
    "Polen":           (52.1,  19.4, ["polen"]),
    "Tschechien":      (49.8,  15.5, ["tschechien", "tschechei"]),
    "Ungarn":          (47.2,  19.4, ["ungarn"]),
    "Kroatien":        (45.1,  15.2, ["kroatien"]),
    "Serbien":         (44.0,  21.0, ["serbien"]),
    "Rumänien":        (45.9,  25.0, ["rumänien", "rumaenien"]),
    "Griechenland":    (39.0,  22.0, ["griechenland"]),
    "Schweden":        (62.0,  15.0, ["schweden"]),
    "Norwegen":        (64.5,  12.0, ["norwegen"]),
    "Dänemark":        (56.0,  10.0, ["dänemark", "daenemark"]),
    "Finnland":        (64.0,  26.0, ["finnland"]),
    "Türkei":          (39.0,  35.2, ["türkei", "tuerkei"]),
    "Ukraine":         (48.4,  31.2, ["ukraine"]),
    "Russland":        (61.5,  90.0, ["russland", "russia", "moskau"]),
    "USA":             (39.5, -98.5, ["usa", "amerika", "vereinigte staaten", "washington"]),
    "Kanada":          (56.1,-106.3, ["kanada"]),
    "Mexiko":          (23.6,-102.5, ["mexiko"]),
    "Brasilien":      (-14.2, -51.9, ["brasilien"]),
    "Argentinien":    (-38.4, -63.6, ["argentinien"]),
    "Chile":          (-35.7, -71.5, ["chile"]),
    "Kolumbien":       ( 4.6, -74.3, ["kolumbien"]),
    "Venezuela":       ( 6.4, -66.6, ["venezuela"]),
    "China":           (35.9, 104.2, ["china", "peking"]),
    "Japan":           (36.2, 138.3, ["japan", "tokio"]),
    "Indien":          (20.6,  78.9, ["indien"]),
    "Südkorea":        (35.9, 127.8, ["südkorea", "suedkorea"]),
    "Nordkorea":       (40.3, 127.5, ["nordkorea"]),
    "Taiwan":          (23.7, 121.0, ["taiwan"]),
    "Indonesien":     ( -0.8, 113.9, ["indonesien"]),
    "Vietnam":         (14.1, 108.3, ["vietnam"]),
    "Thailand":        (15.9, 101.0, ["thailand"]),
    "Pakistan":        (30.4,  69.3, ["pakistan"]),
    "Afghanistan":     (33.9,  67.7, ["afghanistan"]),
    "Iran":            (32.4,  53.7, ["iran", "teheran"]),
    "Irak":            (33.2,  43.7, ["irak"]),
    "Syrien":          (34.8,  39.0, ["syrien"]),
    "Israel":          (31.5,  34.9, ["israel", "gaza", "nahost"]),
    "Saudi-Arabien":   (23.9,  45.1, ["saudi", "saudi-arabien"]),
    "Ägypten":         (26.8,  30.8, ["ägypten", "aegypten"]),
    "Marokko":         (31.8,  -7.1, ["marokko"]),
    "Algerien":        (28.0,   1.7, ["algerien"]),
    "Tunesien":        (33.9,   9.5, ["tunesien"]),
    "Libyen":          (26.3,  17.2, ["libyen"]),
    "Nigeria":         ( 9.1,   8.7, ["nigeria"]),
    "Äthiopien":       ( 9.1,  40.5, ["äthiopien", "aethiopien"]),
    "Kenia":           ( 0.0,  37.9, ["kenia"]),
    "Südafrika":      (-30.6,  22.9, ["südafrika", "suedafrika"]),
    "Australien":     (-25.3, 133.8, ["australien"]),
    "Neuseeland":     (-40.9, 174.9, ["neuseeland"]),
}

# Wo ich eine gute deutschsprachige Quelle kenne, nehme ich die statt der Suche.
LAND_FEEDS = {
    "Schweiz": [("SRF", "https://www.srf.ch/news/bnf/rss/1890"),
                ("NZZ", "https://www.nzz.ch/schweiz.rss")],
    "Deutschland": [("Tagesschau", "https://www.tagesschau.de/xml/rss2/")],
}


def land_erkennen(text):
    """Findet das erste Land, das in Mikas Satz vorkommt.
    Ganze Woerter — sonst steckt "uk" in "Zucker" und "usa" in "Klausur"."""
    t = (text or "").lower()
    treffer = None
    for name, (lat, lon, worte) in LAENDER.items():
        for w in worte:
            m = re.search(r"(?<![a-zäöüß])" + re.escape(w) + r"(?![a-zäöüß])", t)
            if m and (treffer is None or m.start() < treffer[0]):
                treffer = (m.start(), name, lat, lon)
    if not treffer:
        return None
    _, name, lat, lon = treffer
    return {"land": name, "lat": lat, "lon": lon}


# Nur der Landesname holt Streuware — jeder deutsche Artikel, der das Land
# erwaehnt (Formel 1, Reisetipps). Mit Themenwoertern kommt heraus, was im Land
# wirklich laeuft.
NEWS_THEMEN = "(Regierung OR Politik OR Wirtschaft OR Wahl OR Krise OR Parlament OR Präsident)"


def _google_news(land):
    q = urllib.parse.quote("%s %s" % (land, NEWS_THEMEN))
    return [("Google News", "https://news.google.com/rss/search?q=%s&hl=de&gl=CH&ceid=CH:de" % q)]


# ---------------------------------------------------------------- Nachrichten (RSS)

# Reihenfolge egal — die Meldungen werden am Schluss nach Zeit sortiert.
# Zum Ändern einfach Zeilen austauschen: (Anzeigename, RSS-Adresse)
NEWS_FEEDS = [
    ("SRF",         "https://www.srf.ch/news/bnf/rss/1922"),      # International
    ("NZZ",         "https://www.nzz.ch/international.rss"),
    ("Tagesschau",  "https://www.tagesschau.de/xml/rss2/"),
]

_news = {}                      # Land (oder "" für die Welt) -> {"t":…, "data":…}


def _rss_zeit(roh):
    """pubDate eines Feeds in ein datetime umwandeln — Feeds sind da unzuverlässig."""
    if not roh:
        return None
    try:
        import email.utils
        d = email.utils.parsedate_to_datetime(roh)
        return d.astimezone() if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc).astimezone()
    except Exception:
        return None


def _alter(d):
    if not d:
        return ""
    sek = (datetime.datetime.now(d.tzinfo) - d).total_seconds()
    if sek < 90:
        return "gerade eben"
    if sek < 3600:
        return "vor %d Min" % (sek // 60)
    if sek < 86400:
        std = int(sek // 3600)
        return "vor %d Std" % std
    tage = int(sek // 86400)
    return "gestern" if tage == 1 else "vor %d Tagen" % tage


def _feed_lesen(name, url):
    import xml.etree.ElementTree as ET
    req = urllib.request.Request(url, headers={"User-Agent": "Jarvis/1.0 (persoenlicher Assistent)"})
    roh = urllib.request.urlopen(req, timeout=12).read()
    wurzel = ET.fromstring(roh)
    raus = []
    for it in wurzel.iter("item"):
        titel = (it.findtext("title") or "").strip()
        if not titel:
            continue
        quelle = name
        if name == "Google News" and " - " in titel:
            titel, _, hinten = titel.rpartition(" - ")
            if 2 < len(hinten) < 28:
                quelle = hinten
        d = _rss_zeit(it.findtext("pubDate"))
        raus.append({
            "title": titel,
            "url": (it.findtext("link") or "").strip(),
            "source": quelle,
            "when": _alter(d),
            "ts": d.timestamp() if d else 0,
        })
    return raus


def fetch_news(land=None):
    """Schlagzeilen aus mehreren Feeds, zusammengelegt und nach Zeit sortiert.
    Ohne Land: die Welt. Mit Land: dessen eigene Quellen oder eine Suche."""
    schluessel = land or ""
    topf = _news.setdefault(schluessel, {"t": 0, "data": None})
    if time.time() - topf["t"] < 900 and topf["data"] is not None:
        return topf["data"]
    if land:
        quellen = LAND_FEEDS.get(land) or _google_news(land)
    else:
        quellen = NEWS_FEEDS
    proQuelle, fehler = [], []
    for name, url in quellen:
        try:
            m = _feed_lesen(name, url)
            m.sort(key=lambda x: x["ts"], reverse=True)
            proQuelle.append(m[:12])
        except Exception as e:
            fehler.append("%s: %s" % (name, str(e)[:60]))
    if not any(proQuelle):
        topf.update(t=time.time(), data={"error": "; ".join(fehler) or "keine Meldungen"})
        return topf["data"]

    # Abwechselnd aus jeder Quelle ziehen — sonst überschwemmt ein einzelner
    # Feed mit vielen frischen Meldungen die ganze Liste.
    gesehen, raus = set(), []
    for runde in range(12):
        for liste in proQuelle:
            if runde >= len(liste) or len(raus) >= 24:
                continue
            m = liste[runde]
            schluessel = m["title"].lower()[:55]
            if schluessel in gesehen:
                continue
            gesehen.add(schluessel)
            raus.append(m)
    topf.update(t=time.time(), data=raus)
    return raus


NEWS_WOERTER = ("news", "nachricht", "schlagzeile", "welt", "weltgeschehen", "aktuell",
                "passiert", "neuigkeit", "politik", "krieg", "wirtschaftslage",
                "meldung", "presse", "medien",
                # so redet Mika wirklich
                "was geht", "geht ab", "abgeht", "was läuft", "was lauft", "läuft gerade",
                "was ist los", "was los", "los in", "lage in", "situation in", "steht an in",
                "läuft es in", "lauft es in", "läuft in", "steht es in", "sieht es in",
                "sieht es aus", "siehts aus", "sieht's aus", "wie sieht", "steht es um",
                "gibt es neues", "gibts neues", "gibt's neues")


def orte_im_text(text):
    """Jedes Land, das im Text vorkommt, mit seiner relativen Stelle (0 bis 1).
    Damit kann die Oberfläche die Erde mitdrehen, während Jarvis spricht."""
    t = (text or "").lower()
    if not t:
        return []
    treffer = []
    for name, (lat, lon, worte) in LAENDER.items():
        for w in worte:
            for m in re.finditer(r"(?<![a-zäöüß])" + re.escape(w) + r"(?![a-zäöüß])", t):
                treffer.append((m.start(), name, lat, lon))
    treffer.sort()
    raus, letztes = [], None
    for pos, name, lat, lon in treffer:
        if name == letztes:                      # nicht zweimal hintereinander dasselbe Land
            continue
        letztes = name
        raus.append({"land": name, "lat": lat, "lon": lon, "pos": round(pos / len(t), 4)})
    return raus[:8]


def regie(reply, morgen, news_ab, fokus):
    """Sagt der Oberflaeche, ab wann die Weltkarte kommt und wohin sie zeigen soll.
    karte_ab ist ein Anteil der Antwort (0 = sofort, None = gar nicht),
    orte sind Stationen mit derselben Zeitrechnung."""
    if morgen:
        if not news_ab:
            return {"karte_ab": None, "orte": []}
        # Beim Briefing kommt die Karte erst zu den Nachrichten — vorher bleibt
        # Jarvis auf der Uebersicht bei Mails, Terminen und Aufgaben.
        orte = [{"land": "Schweiz", "lat": 46.8, "lon": 8.2, "pos": news_ab}]
        for o in orte_im_text(reply):
            if o["pos"] > news_ab + .02 and o["land"] != orte[-1]["land"]:
                orte.append(o)
        return {"karte_ab": news_ab, "orte": orte[:6]}
    if not fokus:
        return {"karte_ab": None, "orte": []}
    orte = orte_im_text(reply)
    if fokus.get("land"):
        # Das gefragte Land zuerst, auch wenn Jarvis es erst spaeter nennt
        orte = ([{"land": fokus["land"], "lat": fokus["lat"], "lon": fokus["lon"], "pos": 0.0}]
                + [o for o in orte if o["land"] != fokus["land"]])
    return {"karte_ab": 0.0, "orte": orte[:6]}


def fokus_bestimmen(text):
    """Fragt Mika nach Nachrichten? Dann sagen wir der Oberfläche, wohin die Erde soll.
    Mit Land: dorthin. Ohne Land: zurück auf die ganze Welt."""
    if not any(w in (text or "").lower() for w in NEWS_WOERTER):
        return None
    return land_erkennen(text) or {"land": None, "lat": 15.0, "lon": 5.0}

def news_kontext(text, fokus=None):
    """Nur wenn Mika nach Nachrichten fragt, bekommt das Gehirn die Schlagzeilen mit.
    Nennt er ein Land, sind es die Schlagzeilen von dort."""
    t = (text or "").lower()
    if not any(w in t for w in NEWS_WOERTER):
        return ""
    try:
        n = fetch_news(fokus.get("land") if fokus else None)
    except Exception:
        return ""
    if not isinstance(n, list) or not n:
        return ""
    zeilen = "; ".join("%s (%s, %s)" % (m["title"], m["source"], m["when"]) for m in n[:8])
    woher = ("aus " + fokus["land"]) if (fokus and fokus.get("land")) else "aus der Welt"
    return (" Das sind die aktuellen Schlagzeilen " + woher + ": " + zeilen + ". "
            "Nenn höchstens drei davon, fass sie in eigenen Worten zusammen und sag dazu, "
            "wie alt die Meldung ist. Schreib das als zusammenhängenden Text — keine "
            "Aufzählung, keine Nummerierung, keine Spiegelstriche.")


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


# ---------------------------------------------------------------- Wartet auf dich / Nachfassen

def _me(creds):
    """Eigene Mailadresse (einmal holen, dann gemerkt)."""
    if not hasattr(_me, "addr"):
        try:
            _me.addr = gapi(creds, "https://gmail.googleapis.com/gmail/v1/users/me/profile").get("emailAddress", "").lower()
        except Exception:
            _me.addr = ""
    return _me.addr


def _thread_summary(creds, tid, me):
    """Letzte Nachricht eines Gesprächs: von wem, wann, worum."""
    t = gapi(creds, "https://gmail.googleapis.com/gmail/v1/users/me/threads/" + tid +
             "?format=metadata&metadataHeaders=From&metadataHeaders=Subject")
    msgs = t.get("messages", [])
    if not msgs:
        return None
    last = msgs[-1]
    hdr = {h["name"]: h["value"] for h in last.get("payload", {}).get("headers", [])}
    raw_from = hdr.get("From", "")
    from_me = me and me in raw_from.lower()
    ts = int(last.get("internalDate", "0")) / 1000
    when = datetime.datetime.fromtimestamp(ts).astimezone()
    days = max(0, (datetime.datetime.now().astimezone() - when).days)
    who = raw_from.split("<")[0].strip().strip('"') or raw_from
    return {"id": msgs[0]["id"], "thread": tid, "from_me": bool(from_me),
            "who": who[:40], "subject": hdr.get("Subject", "(kein Betreff)")[:90],
            "days": days, "date": when.strftime("%d.%m."), "snippet": (last.get("snippet") or "")[:120]}


def _threads(creds, q, n):
    url = ("https://gmail.googleapis.com/gmail/v1/users/me/threads?maxResults=" + str(n)
           + "&q=" + urllib.parse.quote(q))
    return [t["id"] for t in gapi(creds, url).get("threads", [])]


def fetch_followups(creds):
    """Zwei Listen: Gespraeche, die auf DICH warten — und solche, wo DU wartest."""
    if time.time() - _follow["t"] < 600 and _follow["data"] is not None:
        return _follow["data"]
    me = _me(creds)
    waiting, follow = [], []
    try:
        for tid in _threads(creds, f"in:inbox older_than:{WAIT_DAYS}d newer_than:45d -category:promotions -category:social", 18):
            s = _thread_summary(creds, tid, me)
            if s and not s["from_me"]:
                waiting.append(s)
    except Exception as e:
        waiting = {"error": str(e)[:120]}
    try:
        for tid in _threads(creds, f"from:me older_than:{FOLLOW_DAYS}d newer_than:70d", 18):
            s = _thread_summary(creds, tid, me)
            if s and s["from_me"]:
                follow.append(s)
    except Exception as e:
        follow = {"error": str(e)[:120]}
    if isinstance(waiting, list):
        waiting.sort(key=lambda x: -x["days"]); waiting = waiting[:8]
    if isinstance(follow, list):
        follow.sort(key=lambda x: -x["days"]); follow = follow[:8]
    data = {"waiting": waiting, "followup": follow}
    _follow.update(t=time.time(), data=data)
    return data


def fetch_quota():
    """Wie viel Stimm-Kontingent bei ElevenLabs diesen Monat noch uebrig ist."""
    if time.time() - _quota["t"] < 900 and _quota["data"] is not None:
        return _quota["data"]
    out = None
    try:
        req = urllib.request.Request("https://api.elevenlabs.io/v1/user/subscription",
                                     headers={"xi-api-key": ELEVEN_KEY})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        used, lim = d.get("character_count", 0), d.get("character_limit", 0) or 1
        out = {"used": used, "limit": lim, "left_pct": max(0, round(100 - used * 100.0 / lim))}
    except Exception as e:
        out = {"error": str(e)[:80]}
    _quota.update(t=time.time(), data=out)
    return out


# ---------------------------------------------------------------- Das Team

TEAM_FILE = os.path.join(APP_DIR, "team.json")

TEAM_DEFAULT = [
    {"id": "jarvis", "name": "Jarvis", "role": "Chef-Assistent", "lead": True,
     "desc": "Kalender, Mail, Aufgaben, Alltag — der Kopf des Ganzen.",
     "voice_id": ELEVEN_VOICE_ID, "voice_name": "Helmut",
     "hint": "Du bist JARVIS, Mikas persönlicher Assistent. Du kümmerst dich um "
             "Kalender, Mail, Aufgaben und Alltagsfragen. Nenn ihn Chef. Du bist "
             "loyal, ruhig und trocken-humorvoll."},
    {"id": "akquise", "name": "Rachel", "role": "Akquise", "lead": False,
     "desc": "Firmen recherchieren, Anschreiben entwerfen, Nachfassen.",
     "voice_id": "", "voice_name": "",
     "hint": "Du bist Rachel und zuständig für Neukunden-Akquise bei Mikas "
             "Webdesign-Firma. Du recherchierst passende Firmen, entwirfst kurze "
             "persönliche Anschreiben und erinnerst ans Nachfassen. Du schreibst "
             "knapp, konkret und ohne Marketing-Floskeln. Nenn ihn Chef. "
             "Mails werden immer nur als Entwurf vorbereitet, nie gesendet."},
    {"id": "technik", "name": "Ben", "role": "Technik", "lead": False,
     "desc": "Websites, Code, Fehlersuche, technische Einschätzungen.",
     "voice_id": "", "voice_name": "",
     "hint": "Du bist Ben und der technische Kopf in Mikas Webdesign-Firma. "
             "Du beantwortest Fragen zu Websites, Code, Hosting und Fehlersuche. "
             "Du erklärst verständlich, ohne Fachjargon-Nebel, und sagst klar, "
             "wenn etwas eine schlechte Idee ist. Nenn ihn Chef."},
    {"id": "content", "name": "Lina", "role": "Content", "lead": False,
     "desc": "LinkedIn-Posts, Texte, Ideen, Formulierungen.",
     "voice_id": "", "voice_name": "",
     "hint": "Du bist Lina und zuständig für Texte und Content bei Mikas Firma. "
             "Du schreibst LinkedIn-Beiträge, Website-Texte und Formulierungen. "
             "Du schreibst wie ein Mensch: konkret, kurze Sätze, keine Buzzwords, "
             "keine Emoji-Wände. Nenn ihn Chef."},
]


def load_team():
    try:
        with open(TEAM_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        by_id = {m.get("id"): m for m in saved if isinstance(m, dict)}
        out = []
        for d in TEAM_DEFAULT:
            m = dict(d)
            if d["id"] in by_id:          # nur Stimme und Name uebernehmen, Rest bleibt gepflegt
                for k in ("voice_id", "voice_name", "name"):
                    if by_id[d["id"]].get(k):
                        m[k] = by_id[d["id"]][k]
            out.append(m)
        return out
    except Exception:
        return [dict(d) for d in TEAM_DEFAULT]


def save_team(members):
    keep = [{"id": m.get("id"), "name": m.get("name", ""),
             "voice_id": m.get("voice_id", ""), "voice_name": m.get("voice_name", "")}
            for m in members if m.get("id")]
    with open(TEAM_FILE, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    return load_team()


def team_kontext():
    """Kurzer Hinweis, damit jede Person weiss, wer sonst noch da ist."""
    ms = load_team()
    liste = ", ".join(f"{m['name']} ({m['role']})" for m in ms)
    return (" Du gehörst zu Mikas Team: " + liste + ". "
            "Wenn eine Frage klar zu einer anderen Person gehört, sag kurz, "
            "wer besser passt — Mika wechselt dann mit «Hey <Name>».")


def member(who):
    for m in load_team():
        if m["id"] == (who or "jarvis"):
            return m
    return load_team()[0]


# Wenn Mika jemanden beim Namen anspricht, uebernimmt diese Person
ANRUFE = ("hey", "hallo", "he", "ey", "ok", "okay", "jo", "sag mal", "du")

NAMENS_VARIANTEN = {
    "jarvis":  ["jarvis", "travis", "jervis", "jarwis", "charvis", "dscharvis", "sarvis"],
    "rachel":  ["rachel", "rachael", "rachelle", "raechel", "räschel", "reichel", "rejchel", "raschel"],
    "ben":     ["ben", "benn", "bän", "beno"],
    "lina":    ["lina", "lena", "leena", "liena", "linna"],
    "vera":    ["vera", "wera"],
}


def _aliase(m):
    n = (m.get("name") or "").strip().lower().split()[0] if m.get("name") else ""
    out = set([n]) if n else set()
    out |= set(NAMENS_VARIANTEN.get(n, []))
    return sorted((a for a in out if len(a) >= 2), key=len, reverse=True)


def wer_ist_gemeint(text, fallback):
    """Sucht am Satzanfang eine Anrede wie 'Hey Rachel' und gibt die Person zurueck."""
    import re
    t = (text or "").strip().lower()
    if not t:
        return fallback, False
    anrede = "(?:%s)" % "|".join(re.escape(a) for a in ANRUFE)
    for m in load_team():
        for a in _aliase(m):
            if re.match(r"^\s*(?:%s[\s,]+)?%s\b" % (anrede, re.escape(a)), t):
                return m, (m["id"] != fallback["id"])
    return fallback, False


_voices = {"t": 0, "data": None}


def eleven_voices():
    """Welche Stimmen im ElevenLabs-Konto verfuegbar sind."""
    if time.time() - _voices["t"] < 900 and _voices["data"] is not None:
        return _voices["data"]
    out = []
    try:
        req = urllib.request.Request("https://api.elevenlabs.io/v1/voices",
                                     headers={"xi-api-key": ELEVEN_KEY})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        for v in d.get("voices", []):
            lab = v.get("labels") or {}
            out.append({"id": v.get("voice_id"), "name": v.get("name", "?"),
                        "desc": ", ".join(x for x in [lab.get("gender"), lab.get("age"),
                                                      lab.get("accent"), lab.get("description")] if x)})
    except Exception as e:
        out = {"error": str(e)[:120]}
    _voices.update(t=time.time(), data=out)
    return out


# ---------------------------------------------------------------- Morgenbriefing

GRUSS_MUSTER = ("guten morgen", "gueten morge", "guete morge", "gute morgen",
                "morgen zusammen", "guten morgen zusammen", "moin", "guete morgä")


KURZ_GRUSS = ("morgen", "morge", "morgä", "moin", "moin moin", "guten tag", "grüezi")


def _ohne_namen(t):
    """Anrede vorne und hinten wegschneiden: 'hey jarvis guten morgen chef' → 'guten morgen'."""
    namen = []
    for m in load_team():
        namen.extend(_aliase(m))
    namen.append("chef")
    aenderung = True
    while aenderung:
        aenderung = False
        for w in ("hey", "hallo", "he", "ey", "ok", "okay", "jo", "du", "sag mal"):
            if t.startswith(w + " "):
                t = t[len(w) + 1:].strip(); aenderung = True
        for a in namen:
            if t == a or (t.startswith(a) and t[len(a):len(a) + 1] in (" ", ",", ".")):
                t = t[len(a):].lstrip(" ,.").strip(); aenderung = True
            if t.endswith(" " + a) or t.endswith(", " + a):
                t = t[:t.rfind(a)].rstrip(" ,.").strip(); aenderung = True
    return t


def ist_morgengruss(text):
    t = (text or "").strip().lower().rstrip(".!?,")
    if not t:
        return False
    t = _ohne_namen(t)
    fueller = ("zusammen", "zäme", "miteinander", "alle", "allerseits", "auch")
    for g in GRUSS_MUSTER:
        if t.startswith(g):
            rest = t[len(g):].strip(" ,.!")     # nur der blosse Gruss, kein ganzer Satz
            return rest == "" or rest.split()[0] in fueller
    return t in KURZ_GRUSS                      # blosses "Morgen" nur allein, nie im Satz


def _anzahl(n, eins, viele):
    return ("eine " + eins) if n == 1 else (f"{n} {viele}" if n else f"keine {viele}")


def briefing_text(name):
    """Kurzes gesprochenes Briefing aus den echten Daten."""
    d = build_dashboard()
    h = datetime.datetime.now().hour
    teile = [f"Guten Morgen, Chef. Hier ist {name}."] if h < 11 else [f"Hallo Chef, hier ist {name}."]

    mails = d.get("mails")
    if isinstance(mails, list):
        neu = [m for m in mails if m.get("unread")]
        if neu:
            satz = _anzahl(len(neu), "neue Mail", "neue Mails").capitalize()
            absender = ", ".join(m["from"] for m in neu[:2])
            teile.append(f"{satz} im Posteingang, von {absender}.")
        else:
            teile.append("Im Posteingang ist nichts Neues.")

    evs = d.get("events")
    if isinstance(evs, list):
        heute = [e for e in evs if e.get("day") == "heute" and not e.get("allday")]
        if heute:
            erste = heute[0]
            rest = len(heute) - 1
            satz = f"Heute {_anzahl(len(heute), 'Termin', 'Termine')}, der nächste um {erste['when']}: {erste['title']}."
            if rest > 0:
                satz = satz[:-1] + f", danach noch {rest} weitere."
            teile.append(satz)
        else:
            teile.append("Heute stehen keine Termine mehr an.")

    rem = d.get("reminders")
    if isinstance(rem, list) and rem:
        teile.append(f"Offen sind {_anzahl(len(rem), 'Aufgabe', 'Aufgaben')}, zum Beispiel {rem[0]}.")

    w = d.get("weather")
    if isinstance(w, dict) and not w.get("error"):
        satz = f"Draussen {w['temp']} Grad, {w['desc']}."
        if w.get("rain"):
            satz += f" Ab {w['rain']['at']} soll es regnen."
        teile.append(satz)

    f = _follow.get("data")                      # nur wenn ohnehin schon geladen
    if isinstance(f, dict) and isinstance(f.get("waiting"), list) and f["waiting"]:
        teile.append(f"Und {len(f['waiting'])} Gespräche warten noch auf deine Antwort.")

    # Bis hierher: Mails, Termine, Aufgaben, Wetter. Jetzt kommt die Welt.
    vor_news = " ".join(teile)

    news_satz = ""
    try:
        n = fetch_news("Schweiz")
        if isinstance(n, list) and n:
            def sauber(t):
                t = t.replace(" – ", ", ").replace(" — ", ", ").replace(" - ", ", ")
                return t.strip().rstrip(".")
            koepfe = [sauber(m["title"]) for m in n[:2]]
            news_satz = "Und noch kurz aus der Schweiz: " + koepfe[0] + "."
            if len(koepfe) > 1:
                news_satz += " Ausserdem: " + koepfe[1] + "."
    except Exception:
        news_satz = ""

    schluss = "Womit fangen wir an?"
    ganz = " ".join([vor_news, news_satz, schluss]).replace("  ", " ").strip()
    # Anteil des Textes, ab dem die Schweizer Nachrichten kommen — daran hängt
    # die Oberfläche den Wechsel auf die Welt-Seite auf.
    anteil = (len(vor_news) + 1) / len(ganz) if news_satz else None
    return {"text": ganz, "news_ab": round(anteil, 4) if anteil else None}


# ---------------------------------------------------------------- Beiträge (LinkedIn)

POST_FILE = os.path.join(APP_DIR, "posts.json")
LI_TOKEN_FILE = os.path.join(HOME, ".hermes", "linkedin_token.json")
LI_ID = ENV.get("LINKEDIN_CLIENT_ID", "")
LI_SECRET = ENV.get("LINKEDIN_CLIENT_SECRET", "")
LI_REDIRECT = f"http://localhost:{PORT}/api/linkedin/callback"

POST_HINT = (
    "Du schreibst einen LinkedIn-Beitrag für Mika. Er führt mit Nicola und Timo "
    "eine kleine Agentur für Webdesign und Content (trendingmedia.ch) in der Schweiz. "
    "Schreib auf Deutsch, in der Ich-Form, wie Mika selbst spricht: klar, "
    "bodenständig, ohne Werbefloskeln und ohne Übertreibungen. Kein Markdown, "
    "keine Sternchen, keine Überschriften. Erster Satz muss neugierig machen. "
    "Kurze Absätze, 120 bis 200 Wörter. Am Schluss eine echte Frage an die Leser "
    "und höchstens drei passende Hashtags. Gib NUR den Beitrag aus, keine Erklärung."
)


def load_posts():
    try:
        with open(POST_FILE) as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def save_posts(liste):
    with open(POST_FILE, "w") as f:
        json.dump(liste[-60:], f, ensure_ascii=False, indent=1)
    return liste


def post_neu(thema, plattform="linkedin"):
    text = ask_hermes(f"Thema: {thema}", POST_HINT)
    p = {
        "id": str(int(time.time() * 1000)),
        "plattform": plattform,
        "thema": thema,
        "text": text,
        "status": "entwurf",
        "erstellt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "url": "",
    }
    liste = load_posts()
    liste.append(p)
    save_posts(liste)
    return p


def post_aendern(pid, anweisung):
    liste = load_posts()
    for p in liste:
        if p["id"] == pid:
            p["text"] = ask_hermes(
                "Hier ist der bisherige Beitrag:\n\n" + p["text"] +
                "\n\nÄndere ihn so: " + anweisung, POST_HINT)
            save_posts(liste)
            return p
    raise RuntimeError("Beitrag nicht gefunden")


def li_token():
    """Gültiges Zugangs-Token holen; erneuert sich still, solange der
    Erneuerungsschlüssel gilt (ein Jahr)."""
    try:
        with open(LI_TOKEN_FILE) as f:
            t = json.load(f)
    except Exception:
        return None
    if t.get("expires_at", 0) > time.time() + 120:
        return t
    if not (t.get("refresh_token") and LI_ID and LI_SECRET):
        return None
    daten = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": t["refresh_token"],
        "client_id": LI_ID,
        "client_secret": LI_SECRET,
    }).encode()
    req = urllib.request.Request("https://www.linkedin.com/oauth/v2/accessToken", data=daten)
    with urllib.request.urlopen(req, timeout=20) as r:
        neu = json.loads(r.read())
    t["access_token"] = neu["access_token"]
    t["expires_at"] = time.time() + int(neu.get("expires_in", 5184000))
    if neu.get("refresh_token"):
        t["refresh_token"] = neu["refresh_token"]
    with open(LI_TOKEN_FILE, "w") as f:
        json.dump(t, f)
    os.chmod(LI_TOKEN_FILE, 0o600)
    return t


def li_status():
    if not (LI_ID and LI_SECRET):
        return {"bereit": False, "grund": "LINKEDIN_CLIENT_ID/SECRET fehlen in ~/.hermes/.env"}
    t = li_token()
    if not t:
        return {"bereit": False, "grund": "noch nicht angemeldet", "start": "/api/linkedin/start"}
    return {"bereit": True, "name": t.get("name", ""), "bis": int(t.get("expires_at", 0))}


def li_veroeffentlichen(text):
    t = li_token()
    if not t:
        raise RuntimeError("kein LinkedIn-Zugang — zuerst anmelden")
    koerper = json.dumps({
        "author": "urn:li:person:" + t["sub"],
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }).encode()
    req = urllib.request.Request(
        "https://api.linkedin.com/v2/ugcPosts", data=koerper,
        headers={"Authorization": "Bearer " + t["access_token"],
                 "Content-Type": "application/json",
                 "X-Restli-Protocol-Version": "2.0.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        kopf = dict(r.getheaders())
        urn = kopf.get("x-restli-id") or kopf.get("X-RestLi-Id") or ""
    return "https://www.linkedin.com/feed/update/" + urn if urn else ""


def post_freigeben(pid):
    """Wird NUR nach ausdrücklicher Freigabe im Interface aufgerufen."""
    liste = load_posts()
    for p in liste:
        if p["id"] == pid:
            if p["status"] == "veroeffentlicht":
                return p
            p["url"] = li_veroeffentlichen(p["text"])
            p["status"] = "veroeffentlicht"
            p["gepostet"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            save_posts(liste)
            return p
    raise RuntimeError("Beitrag nicht gefunden")


# Sprachbefehl: "schreib einen Post über ..." (Wortstellung egal)
WORT_POST = re.compile(r"\b(post|posting|beitrag|beiträge)\b", re.I)
WORT_TU = re.compile(r"\b(schreib\w*|erstell\w*|verfass\w*|formulier\w*|entwirf|mach\w*)\b", re.I)
WORT_THEMA = re.compile(r"\b(?:über|ueber|zu|zum|zur|wegen|thema)\s+(.{2,})$", re.I)


def ist_postauftrag(text):
    t = (text or "").strip().rstrip(".!?,")
    if not t:
        return None
    t = _ohne_namen(t.lower())                  # "Jarvis, schreib mir …" → "schreib mir …"
    if not (WORT_POST.search(t) and WORT_TU.search(t)):
        return None
    if re.search(r"\b(mail|email|e-mail|nachricht|aufgabe|erinnerung)\b", t):
        return None                             # das ist etwas anderes
    m = WORT_THEMA.search(t)
    thema = m.group(1).strip(" .!?,") if m else ""
    return thema or "ein aktuelles Thema aus meiner Arbeit"


# ---------------------------------------------------------------- Musik (Spotify)

INTRO_TRACK = os.environ.get("JARVIS_INTRO_TRACK", "spotify:track:08mG3Y1vljYA6bvDt4Wqkj")
MUSIK_LAUT = int(os.environ.get("JARVIS_MUSIK_LAUT", "100"))    # Intro
MUSIK_LEISE = int(os.environ.get("JARVIS_MUSIK_LEISE", "65"))  # während gesprochen wird
_musik = {"lief": False, "vol": None, "lauf": 0}


def _osa(script, timeout=8):
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "").strip()[:160] or "osascript-Fehler")
    return r.stdout.strip()


def spotify_da():
    try:
        return _osa('tell application "System Events" to return (exists file "/Applications/Spotify.app")') == "true"
    except Exception:
        return False


def musik_start():
    """Song von vorne, laut. Merkt sich die alte Lautstärke.
    Beliebig oft hintereinander: ein laufendes Ausblenden wird abgebrochen."""
    _musik["lauf"] += 1                          # bricht ein laufendes Ausblenden ab
    if not _musik["lief"]:                       # eigene Lautstärke nur einmal merken
        try:
            _musik["vol"] = int(_osa('tell application "Spotify" to return sound volume') or "70")
        except Exception:
            _musik["vol"] = None
    _osa('tell application "Spotify"\n'
         'activate\n'
         f'set sound volume to {MUSIK_LAUT}\n'
         f'play track "{INTRO_TRACK}"\n'
         'end tell', timeout=15)
    _musik["lief"] = True
    return {"ok": True, "quelle": "spotify"}


def musik_leiser():
    _osa(f'tell application "Spotify" to set sound volume to {MUSIK_LEISE}')
    return {"ok": True}


def _ausblenden(marke):
    try:
        for v in range(MUSIK_LEISE, -1, -4):
            if _musik["lauf"] != marke:          # neues Briefing gestartet → abbrechen
                return
            _osa(f'tell application "Spotify" to set sound volume to {max(v, 0)}')
            time.sleep(0.12)
        if _musik["lauf"] != marke:
            return
        _osa('tell application "Spotify" to pause')
        if _musik.get("vol"):
            _osa(f'tell application "Spotify" to set sound volume to {_musik["vol"]}')
        _musik["lief"] = False
    except Exception as e:
        print(f"  Musik-Ausblenden: {e}")
        _musik["lief"] = False


def musik_stop():
    threading.Thread(target=_ausblenden, args=(_musik["lauf"],), daemon=True).start()
    return {"ok": True}


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
        elif self.path.startswith("/api/team"):
            try:
                self._send(200, json.dumps({"team": load_team(), "voices": eleven_voices()}).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
        elif self.path.startswith("/api/followups"):
            self._json_call(lambda c: fetch_followups(c))
        elif self.path.startswith("/api/news"):
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                roh = (q.get("land", [""])[0] or "").strip()
                treffer = land_erkennen(roh) if roh else None
                land = treffer["land"] if treffer else None
                antwort = {"land": land, "meldungen": fetch_news(land)}
                if treffer:
                    antwort["lat"] = treffer["lat"]; antwort["lon"] = treffer["lon"]
                self._send(200, json.dumps(antwort).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
        elif self.path.startswith("/api/welt"):
            try:
                with open(os.path.join(APP_DIR, "welt-raster.json"), "rb") as f:
                    self._send(200, f.read())
            except FileNotFoundError:
                self._send(404, json.dumps({"error": "welt-raster.json fehlt"}).encode())
        elif self.path.startswith("/api/laender"):
            self._send(200, json.dumps(
                [{"land": k, "lat": v[0], "lon": v[1]} for k, v in sorted(LAENDER.items())]).encode())
        elif self.path.startswith("/api/reminders"):
            try:
                self._send(200, json.dumps(fetch_reminders_all()).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
        elif self.path.startswith("/api/intro"):
            pfad = None
            for name in ("intro.mp3", "intro.m4a", "intro.wav"):
                p = os.path.join(APP_DIR, name)
                if os.path.exists(p):
                    pfad = p
                    break
            if not pfad:
                self._send(404, json.dumps({"error": "keine Datei intro.mp3 im Jarvis-Ordner"}).encode())
                return
            try:
                with open(pfad, "rb") as f:
                    daten = f.read()
                typ = "audio/mpeg" if pfad.endswith((".mp3", ".m4a")) else "audio/wav"
                self._send(200, daten, typ)
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:120]}).encode())
        elif self.path.startswith("/api/posts"):
            self._send(200, json.dumps({"posts": load_posts()[::-1],
                                        "linkedin": li_status()}).encode())
        elif self.path.startswith("/api/linkedin/start"):
            if not (LI_ID and LI_SECRET):
                self._send(200, json.dumps({"error": "LINKEDIN_CLIENT_ID/SECRET fehlen in ~/.hermes/.env"}).encode())
                return
            url = ("https://www.linkedin.com/oauth/v2/authorization?response_type=code"
                   f"&client_id={urllib.parse.quote(LI_ID)}"
                   f"&redirect_uri={urllib.parse.quote(LI_REDIRECT)}"
                   "&scope=" + urllib.parse.quote("openid profile w_member_social"))
            self.send_response(302)
            self.send_header("Location", url)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path.startswith("/api/linkedin/callback"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code = (q.get("code") or [""])[0]
            if not code:
                self._send(400, b"kein Code von LinkedIn erhalten", "text/plain; charset=utf-8")
                return
            try:
                daten = urllib.parse.urlencode({
                    "grant_type": "authorization_code", "code": code,
                    "redirect_uri": LI_REDIRECT,
                    "client_id": LI_ID, "client_secret": LI_SECRET}).encode()
                req = urllib.request.Request("https://www.linkedin.com/oauth/v2/accessToken", data=daten)
                with urllib.request.urlopen(req, timeout=20) as r:
                    t = json.loads(r.read())
                t["expires_at"] = time.time() + int(t.get("expires_in", 5184000))
                req2 = urllib.request.Request("https://api.linkedin.com/v2/userinfo",
                                              headers={"Authorization": "Bearer " + t["access_token"]})
                with urllib.request.urlopen(req2, timeout=20) as r:
                    me = json.loads(r.read())
                t["sub"] = me.get("sub", "")
                t["name"] = me.get("name", "")
                with open(LI_TOKEN_FILE, "w") as f:
                    json.dump(t, f)
                os.chmod(LI_TOKEN_FILE, 0o600)
                self._send(200, ("<meta charset='utf-8'><body style='font:16px system-ui;padding:40px'>"
                                 f"<h2>LinkedIn verbunden</h2><p>Angemeldet als {me.get('name','')}. "
                                 "Du kannst dieses Fenster schliessen.</p>").encode(),
                           "text/html; charset=utf-8")
            except urllib.error.HTTPError as e:
                self._send(502, ("LinkedIn-Fehler: " + e.read().decode(errors="replace")[:400]).encode(),
                           "text/plain; charset=utf-8")
        elif self.path.startswith("/vorschau"):
            try:
                with open(os.path.join(APP_DIR, "kopf-vorschau.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"keine kopf-vorschau.html im Jarvis-Ordner", "text/plain")
        elif self.path == "/api/health":
            ok = {"server": True, "elevenlabs": bool(ELEVEN_KEY), "hermes_key": bool(HERMES_KEY)}
            self._send(200, json.dumps(ok).encode())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/api/music"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            was = (q.get("do") or ["start"])[0]
            try:
                if was == "start":
                    if not spotify_da():
                        self._send(200, json.dumps({"ok": False, "error": "Spotify nicht installiert"}).encode())
                        return
                    res = musik_start()
                elif was == "duck":
                    res = musik_leiser()
                else:
                    res = musik_stop()
                self._send(200, json.dumps(res).encode())
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": str(e)[:200]}).encode())
            return
        if self.path.startswith("/api/post/"):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                was = self.path.split("/api/post/", 1)[1].split("?")[0]
                if was == "neu":
                    res = post_neu(body.get("thema", "").strip() or "meine Arbeit",
                                   body.get("plattform", "linkedin"))
                elif was == "aendern":
                    res = post_aendern(body.get("id", ""), body.get("anweisung", "kürzer"))
                elif was == "text":
                    liste = load_posts()
                    res = None
                    for p in liste:
                        if p["id"] == body.get("id"):
                            p["text"] = body.get("text", p["text"]); res = p
                    save_posts(liste)
                    if res is None:
                        raise RuntimeError("Beitrag nicht gefunden")
                elif was == "freigabe":
                    # nur nach ausdrücklichem Klick im Interface
                    res = post_freigeben(body.get("id", ""))
                elif was == "loeschen":
                    liste = [p for p in load_posts() if p["id"] != body.get("id")]
                    save_posts(liste); res = {"ok": True}
                else:
                    raise RuntimeError("unbekannt")
                self._send(200, json.dumps(res).encode())
            except urllib.error.HTTPError as e:
                self._send(200, json.dumps({"error": "LinkedIn: " + e.read().decode(errors="replace")[:300]}).encode())
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)[:250]}).encode())
            return
        if self.path == "/api/team":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, json.dumps({"team": save_team(body.get("team") or [])}).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)[:200]}).encode())
            return
        if self.path == "/api/ask":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                text = (body.get("text") or "").strip()
                if not text:
                    self._send(400, json.dumps({"error": "leere Frage"}).encode())
                    return
                m, gewechselt = wer_ist_gemeint(text, member(body.get("who")))
                # Nennt Mika ein Land in einer Nachrichtenfrage, dreht sich die Erde dorthin
                fokus = fokus_bestimmen(text)
                morgen = ist_morgengruss(text)
                thema = ist_postauftrag(text)
                entwurf = None
                news_ab = None
                if morgen:
                    br = briefing_text(m["name"])
                    reply = br["text"]; news_ab = br["news_ab"]
                elif thema:
                    entwurf = post_neu(thema)
                    reply = "Entwurf liegt auf der Seite Beitraege bereit. Lies ihn durch, dann gibst du ihn frei."
                else:
                    reply = ask_hermes(text, m["hint"] + team_kontext() + news_kontext(text, fokus) + " " + TEXT_HINT)
                self._send(200, json.dumps(dict({"reply": reply, "who": m["id"], "name": m["name"],
                                                 "switched": gewechselt, "briefing": morgen,
                                                 "post": entwurf, "focus": fokus},
                                                **regie(reply, morgen, news_ab, fokus))).encode())
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
        if not self.path.startswith("/api/talk"):
            self._send(404, b"not found", "text/plain")
            return

        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        who = member((q.get("who") or ["jarvis"])[0])
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
            who, gewechselt = wer_ist_gemeint(text, who)
            if gewechselt:
                print(f"  → {who['name']} übernimmt")
            morgen = ist_morgengruss(text)
            thema = ist_postauftrag(text)
            entwurf = None
            # Nennt Mika ein Land in einer Nachrichtenfrage, dreht sich die Erde dorthin
            fokus = fokus_bestimmen(text)
            news_ab = None
            if morgen:
                step = "briefing"
                br = briefing_text(who["name"])
                reply = br["text"]; news_ab = br["news_ab"]
            elif thema:
                step = "beitrag"
                entwurf = post_neu(thema)
                reply = ("Ich habe einen Entwurf geschrieben, er liegt auf der Seite Beiträge. "
                         "Schau ihn dir an — veröffentlicht wird er erst, wenn du ihn freigibst.")
            else:
                step = "hermes"
                reply = ask_hermes(text, who["hint"] + team_kontext() + news_kontext(text, fokus) + " " + SYSTEM_HINT)
            print(f"  {who['name']}: {reply[:120]}")

            audio_b64 = ""
            if ELEVEN_KEY:
                step = "helmut"
                try:
                    audio_b64 = base64.b64encode(helmut_speaks(reply, who.get("voice_id"))).decode()
                except Exception as e:
                    print(f"  Helmut-Fehler (Antwort kommt als Text): {e}")

            self._send(200, json.dumps({
                "transcript": text, "reply": reply, "audio_b64": audio_b64,
                "who": who["id"], "name": who["name"], "switched": gewechselt,
                "briefing": morgen, "post": entwurf, "focus": fokus,
                **regie(reply, morgen, news_ab, fokus)
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

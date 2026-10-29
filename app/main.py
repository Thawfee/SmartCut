import copy
import json
import os
import shutil
import threading
import time
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "workspace")
UPL = os.path.join(WORK, "uploads")
PJ = os.path.join(WORK, "project.json")
os.makedirs(UPL, exist_ok=True)

COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#fb923c", "#22d3ee", "#a3e635"]
DEFAULT_VERSIONS = {
    "landscape": {"label": "Presentation / YouTube", "ratio": "16:9", "width": 1280, "height": 720, "max_seconds": None},
    "portrait": {"label": "Reels / Shorts", "ratio": "9:16", "width": 720, "height": 1280, "max_seconds": None},
    "short": {"label": "Social teaser", "ratio": "9:16", "width": 720, "height": 1280, "max_seconds": 30},
}

if os.environ.get("LITE"):  # small cloud servers: render at 480p so it stays fast
    for _v in DEFAULT_VERSIONS.values():
        _v["width"], _v["height"] = (854, 480) if _v["width"] > _v["height"] else (480, 854)

DEMO = os.path.join(ROOT, "demo_media")

app = FastAPI(title="SmartCut")
LOCK = threading.RLock()
JOB = {"id": 0, "kind": None, "status": "idle", "pct": 0, "msg": "", "error": None}


# ---------------------------------------------------------------- storage
def new_project():
    return {"brief": "", "glossary": {}, "language": "", "accuracy": "base", "sources": [], "music": None,
            "music_volume": 0.35, "segments": [], "order": [], "versions": copy.deepcopy(DEFAULT_VERSIONS),
            "dismissed": [], "log": [], "renders": {}, "history": []}


def load():
    with LOCK:
        if not os.path.exists(PJ):
            save(new_project())
        with open(PJ, encoding="utf-8") as f:
            return json.load(f)


def save(p):
    with LOCK:
        tmp = PJ + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=1, ensure_ascii=False)
        os.replace(tmp, PJ)


def log(p, msg, invalidate=True):
    p["log"].append(f'{time.strftime("%H:%M:%S")}  {msg}')
    p["log"] = p["log"][-200:]
    if invalidate:
        for r in p["renders"].values():
            r["stale"] = True  # every version must be rebuilt from the updated timeline


def snapshot(p):
    """Undo point: only the editable parts."""
    p["history"].append(json.dumps({
        "segments": [{k: s[k] for k in ("id", "keep", "reason", "text", "cover", "user")} for s in p["segments"]],
        "order": p["order"], "glossary": p["glossary"], "dismissed": p["dismissed"],
        "versions": p["versions"], "music_volume": p.get("music_volume", 0.35)}))
    p["history"] = p["history"][-40:]


def view(p):
    data = {k: v for k, v in p.items() if k != "history"}
    data["can_undo"] = bool(p["history"])
    data["warnings"] = P.warnings(p) if p["segments"] else []
    data["durations"] = {v: P.timeline(p, v)[1] for v in p["versions"]} if p["segments"] else {}
    data["job"] = JOB
    data["engine"] = engine()
    data["demo_available"] = os.path.exists(os.path.join(DEMO, "transcripts.json"))
    for s in data["segments"]:
        s.pop("words", None)  # keep the UI payload small
    return data


def engine():
    if os.environ.get("GROQ_API_KEY"):
        return "cloud"
    import importlib.util
    return "local" if importlib.util.find_spec("faster_whisper") else "none"


def busy():
    if JOB["status"] == "running":
        raise HTTPException(409, "Please wait — still working on: " + (JOB["msg"] or JOB["kind"] or ""))


# ---------------------------------------------------------------- background jobs (UI shows progress)
def start_job(kind, fn):
    busy()
    JOB.update(id=JOB["id"] + 1, kind=kind, status="running", pct=0, msg="Starting…", error=None)

    def progress(pct, msg):
        JOB.update(pct=int(pct), msg=msg)

    def runner():
        try:
            fn(progress)
            JOB.update(status="done", pct=100, msg="Done")
        except Exception as e:
            traceback.print_exc()
            JOB.update(status="error", error=friendly_error(e), msg="Failed")
    threading.Thread(target=runner, daemon=True).start()


def friendly_error(e):
    t = str(e)
    if "No such file" in t and "ffmpeg" in t.lower() or isinstance(e, FileNotFoundError):
        return "FFmpeg was not found. Install FFmpeg and add it to PATH, then restart the app."
    if "faster_whisper" in t or "No module named" in t:
        return "Speech model not installed. Run: pip install -r requirements.txt"
    if "huggingface" in t.lower() or "connection" in t.lower() or "resolve" in t.lower():
        return "Could not download the speech model. Connect to the internet once and try again."
    return t[-600:]


# ---------------------------------------------------------------- API
@app.get("/api/project")
def get_project():
    return view(load())


@app.get("/api/job")
def get_job():
    return JOB


@app.post("/api/demo")
def load_demo():
    """One click demo: sample recordings with ready transcripts, so it works without any API key."""
    meta = json.load(open(os.path.join(DEMO, "transcripts.json"), encoding="utf-8"))

    def work(progress):
        with LOCK:
            shutil.rmtree(WORK, ignore_errors=True)
            os.makedirs(UPL, exist_ok=True)
        q = new_project()
        q["brief"], q["glossary"], q["language"] = meta["brief"], meta["glossary"], "en"
        files = meta["files"]
        for n, (fname, words) in enumerate(files.items()):
            progress(10 + 60 * n / len(files), f"Loading {fname}")
            name = "demo_" + fname
            shutil.copy(os.path.join(DEMO, fname), os.path.join(UPL, name))
            info = P.probe(os.path.join(UPL, name))
            sid = f"R{n + 1}"
            q["sources"].append({"id": sid, "file": name, "name": fname, **info, "transcribed": True,
                                 "lufs": P.measure_loudness(os.path.join(UPL, name)) if info["has_audio"] else None,
                                 "speech": bool(words), "color": COLORS[n % len(COLORS)]})
            for s in P.words_to_segments(words, sid, info["duration"]):
                s["id"] = f"S{len(q['segments']) + 1}"
                q["segments"].append(s)
                q["order"].append(s["id"])
        shutil.copy(os.path.join(DEMO, meta["music"]), os.path.join(UPL, "demo_" + meta["music"]))
        q["music"] = "demo_" + meta["music"]
        progress(80, "Finding pauses, repeats, false starts and duplicate camera angles")
        P.analyse(q)
        pre = os.path.join(DEMO, "prerendered")
        for v in q["versions"]:
            mp4, srt = os.path.join(pre, f"out_{v}.mp4"), os.path.join(pre, f"out_{v}.srt")
            if os.path.exists(mp4) and os.path.exists(srt):
                shutil.copy(mp4, WORK)
                shutil.copy(srt, WORK)
                q["renders"][v] = {"mp4": f"out_{v}.mp4", "srt": f"out_{v}.srt", "clips_rendered": 0,
                                   "clips_reused": 0, "duration": P.timeline(q, v)[1], "captions_burned": True,
                                   "at": "demo", "stale": False}
        log(q, "Demo project loaded: 5 recordings + music, rough cut built from the brief", invalidate=False)
        save(q)
    start_job("process", work)
    return view(load())


@app.post("/api/reset")
def reset():
    busy()
    with LOCK:
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(UPL, exist_ok=True)
        save(new_project())
    return view(load())


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), kind: str = Form("source")):
    busy()
    p = load()
    for uf in files:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(uf.filename or "file"))
        name = f"{int(time.time() * 1000)}_{safe}"
        path = os.path.join(UPL, name)
        with open(path, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        info = P.probe(path)
        if not info["has_audio"] and not info["has_video"]:
            os.remove(path)
            raise HTTPException(400, f"{uf.filename} is not a video or audio file FFmpeg can read.")
        if kind == "music":
            p["music"] = name
            log(p, f"Music set: {uf.filename}")
        else:
            n = len(p["sources"]) + 1
            sid = f"R{n}"
            while any(s["id"] == sid for s in p["sources"]):
                n += 1
                sid = f"R{n}"
            p["sources"].append({"id": sid, "file": name, "name": uf.filename, **info, "transcribed": False,
                                 "lufs": None, "speech": None, "color": COLORS[(n - 1) % len(COLORS)]})
            log(p, f"Recording {sid} added: {uf.filename} ({info['duration']:.0f}s)")
    save(p)
    return view(p)


@app.post("/api/remove_source/{sid}")
def remove_source(sid: str):
    busy()
    p = load()
    if sid == "music":
        p["music"] = None
        log(p, "Music removed")
    else:
        snapshot(p)
        p["sources"] = [s for s in p["sources"] if s["id"] != sid]
        gone = {s["id"] for s in p["segments"] if s["source"] == sid}
        p["segments"] = [s for s in p["segments"] if s["source"] != sid]
        p["order"] = [i for i in p["order"] if i not in gone]
        for s in p["segments"]:
            if s.get("cover") and s["cover"]["source"] == sid:
                s["cover"] = None
        p["history"] = []  # structural change: undo history no longer matches
        log(p, f"Recording {sid} removed")
    save(p)
    return view(p)


class ProcessReq(BaseModel):
    brief: str = ""
    glossary: str = ""   # lines like:  wrong => right
    language: str = ""   # "" = auto, "en", "ta", ...
    accuracy: str = "base"


def parse_glossary(text):
    g = {}
    for line in (text or "").splitlines():
        for sep in ("=>", "->", "="):
            if sep in line:
                a, b = line.split(sep, 1)
                if a.strip() and b.strip():
                    g[a.strip()] = b.strip()
                break
    return g


@app.post("/api/process")
def process(req: ProcessReq):
    p = load()
    if not p["sources"]:
        raise HTTPException(400, "Upload at least one recording first.")

    def work(progress):
        q = load()
        q["brief"], q["glossary"] = req.brief, parse_glossary(req.glossary)
        q["language"], q["accuracy"] = req.language, req.accuracy if req.accuracy in ("tiny", "base", "small", "medium") else "base"
        todo = [s for s in q["sources"] if not s.get("transcribed")]
        for n, src in enumerate(todo):
            path = os.path.join(UPL, src["file"])
            base = 5 + 80 * n / max(1, len(todo))
            if src["has_audio"]:
                progress(base, f"Balancing audio level: {src['name']}")
                src["lufs"] = P.measure_loudness(path)
                progress(base + 5, f"Listening to {src['name']} ({n + 1}/{len(todo)}) — first time downloads the speech model")
                words = P.transcribe(path, q["language"] or None, os.environ.get("FORCE_MODEL") or q["accuracy"])
            else:
                words = []
            new = P.words_to_segments(words, src["id"], src["duration"])
            nid = max([int(s["id"][1:]) for s in q["segments"]] + [0])
            for s in new:
                nid += 1
                s["id"] = f"S{nid}"
                q["segments"].append(s)
                q["order"].append(s["id"])
            src["transcribed"], src["speech"] = True, bool(new)
            log(q, f"{src['id']} transcribed: {len(new)} speech segments" + ("" if new else " (no speech — usable as B-roll)"))
        progress(90, "Finding pauses, repeats, false starts and duplicate camera angles")
        P.analyse(q)
        q["history"] = []
        log(q, "Rough cut built from the brief")
        save(q)
    start_job("process", work)
    return view(p)


class EditReq(BaseModel):
    op: str
    seg: str | None = None
    value: str | None = None


@app.post("/api/edit")
def edit(req: EditReq):
    busy()
    p = load()
    by_id = {s["id"]: s for s in p["segments"]}
    s = by_id.get(req.seg) if req.seg else None
    op = req.op
    if op == "undo":
        if not p["history"]:
            raise HTTPException(400, "Nothing to undo.")
        snap = json.loads(p["history"].pop())
        for x in snap["segments"]:
            if x["id"] in by_id:
                by_id[x["id"]].update(x)
        p["order"], p["glossary"], p["dismissed"] = snap["order"], snap["glossary"], snap["dismissed"]
        p["versions"], p["music_volume"] = snap["versions"], snap["music_volume"]
        log(p, "Undo")
        save(p)
        return view(p)
    snapshot(p)
    if op == "toggle" and s:
        s["keep"], s["user"] = not s["keep"], True
        s["reason"] = "Kept by editor" if s["keep"] else "Removed by editor"
        log(p, f'{s["id"]} {"restored" if s["keep"] else "removed"} → captions, music & all versions follow')
    elif op == "text" and s:
        s["text"] = (req.value or "").strip() or s["original_text"]
        log(p, f'{s["id"]} caption corrected → every version updated (audio untouched)')
    elif op in ("up", "down", "intro") and s:
        o = p["order"]
        i = o.index(s["id"])
        if op == "intro":
            o.insert(0, o.pop(i))
            s["keep"], s["user"] = True, True
            s["reason"] = "Chosen as the new introduction"
            log(p, f'{s["id"]} set as the introduction → all versions start with it')
        else:
            kept = [x for x in o if by_id[x]["keep"]] if s["keep"] else o
            k = kept.index(s["id"])
            k2 = k - 1 if op == "up" else k + 1
            if 0 <= k2 < len(kept):
                a, b = o.index(kept[k]), o.index(kept[k2])
                o[a], o[b] = o[b], o[a]
                log(p, f'{s["id"]} moved {op}')
    elif op == "cover" and s:
        if req.value:
            c = next((x for x in p["sources"] if x["id"] == req.value and x["has_video"]), None)
            if not c:
                raise HTTPException(400, "Choose a video recording for B-roll.")
            s["cover"] = {"source": c["id"], "offset": 0}
            log(p, f'{s["id"]} covered with B-roll from {c["id"]} (speech kept)')
        else:
            s["cover"] = None
            log(p, f'{s["id"]} B-roll removed')
    elif op == "dismiss" and req.value:
        p["dismissed"].append(req.value)
        log(p, f"Warning checked by editor: {req.value}", invalidate=False)
    elif op == "glossary":
        p["glossary"] = parse_glossary(req.value)
        log(p, "Caption fixes applied → names corrected in every version")
    elif op == "brief":
        p["brief"] = req.value or ""
        log(p, "Brief updated", invalidate=False)
    elif op == "music_volume":
        p["music_volume"] = max(0.0, min(1.0, float(req.value or 0.35)))
        log(p, f"Music volume set to {int(p['music_volume'] * 100)}%")
    elif op == "version" and req.seg in p["versions"]:
        p["versions"][req.seg]["max_seconds"] = int(req.value) if req.value else None
        log(p, f"{p['versions'][req.seg]['label']} length set to {req.value + 's' if req.value else 'full'}")
    elif op == "reanalyse":
        for x in p["segments"]:
            x["user"] = False
        P.analyse(p)
        log(p, "Rough cut rebuilt from the brief (editor choices reset)")
    else:
        p["history"].pop()
        raise HTTPException(400, "Unknown edit.")
    save(p)
    return view(p)


@app.post("/api/render/{version}")
def render(version: str):
    p = load()
    targets = list(p["versions"]) if version == "all" else [version]
    if any(v not in p["versions"] for v in targets):
        raise HTTPException(404, "Unknown version")

    def work(progress):
        q = load()
        for n, v in enumerate(targets):
            label = q["versions"][v]["label"]
            res = P.render(q, v, WORK, lambda pct, msg: progress((n + pct / 100) * 100 / len(targets), f"{label}: {msg}"))
            res.update(at=time.strftime("%H:%M:%S"), stale=False)
            q = load()
            q["renders"][v] = res
            log(q, f"Rendered {label}: {res['clips_rendered']} new clips, {res['clips_reused']} reused from cache",
                invalidate=False)
            save(q)
    start_job("render", work)
    return view(p)


@app.get("/files/{name}")
def files(name: str):
    path = os.path.join(WORK, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "File not found — render it first.")
    return FileResponse(path, filename=os.path.basename(name))


app.mount("/uploads", StaticFiles(directory=UPL), name="uploads")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "app", "static"), html=True), name="ui")

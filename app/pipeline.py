"""
SmartCut pipeline (v2).

ONE project.json is the single source of truth:
  sources  -> uploaded recordings (never modified)
  segments -> pieces of speech: source file + start/end (provenance), caption text, keep/drop + reason
  order    -> story order
  versions -> output formats, all DERIVED from the same segments

Edit once -> captions, music timing and every version are regenerated.
Rendering is incremental: each clip is rendered once and cached, so a change
only re-renders the clips it touches instead of rebuilding the whole production.
"""
import difflib
import hashlib
import json
import os
import re
import subprocess
import urllib.request

STOP = set("""a an the and or but so to of in on at for with is are was were be been it this that
these those i you he she we they my your our their me him her them as by from not no yes
do does did have has had will would can could should just very really like um uh okay ok
yeah well about there here what when where which who how all some any more most also then
hi hello make video""".split())
FILLERS = {"um", "uh", "hmm", "ah", "er", "erm", "okay", "ok", "so", "yeah", "like", "right", "you know", "i mean"}
CHATTER = ("is the mic", "mic working", "mic on", "can you hear", "start again", "start over", "one more time",
           "let me try again", "are we rolling", "are we recording", "testing testing", "cut it", "from the top",
           "sorry let me", "take two", "can we redo")
CONNECTORS = ("and", "so", "but", "because", "that", "which", "this", "it", "he", "she", "they",
              "then", "also", "as i said", "like i said", "therefore", "thats why", "that is why")

PAUSE_SPLIT = 0.7      # seconds of silence that becomes a cut point
PAD_START, PAD_END = 0.08, 0.18
LOW_CONF = 0.60
TARGET_LUFS = -18.0


def run(args, cwd=None):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "")[-1200:])
    return r


# ---------------------------------------------------------------- media info
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height:stream_tags=rotate:stream_side_data=rotation:format=duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    info = json.loads(out or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    w, h = (v.get("width", 0), v.get("height", 0)) if v else (0, 0)
    rot = 0
    if v:
        rot = abs(int(float(v.get("tags", {}).get("rotate", 0) or 0)))
        for sd in v.get("side_data_list", []) or []:
            rot = abs(int(float(sd.get("rotation", rot) or 0)))
    if rot in (90, 270):
        w, h = h, w
    # cover-art images inside mp3 show up as a 1-frame "video": treat as audio-only
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    has_video = bool(v) and not path.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"))
    return {"duration": round(dur, 2), "has_video": has_video,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "width": w if has_video else 0, "height": h if has_video else 0}


def measure_loudness(path):
    """Integrated loudness (EBU R128, LUFS). Used to balance recordings made at different levels."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af",
                            "loudnorm=print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = r.stderr
        data = json.loads(txt[txt.rfind("{"): txt.rfind("}") + 1])
        val = float(data["input_i"])
        return None if val < -70 else round(val, 1)
    except Exception:
        return None


# ---------------------------------------------------------------- transcription
_models = {}


def transcribe(path, language=None, model_size="base"):
    """Words with timestamps + confidence.
    Cloud mode (GROQ_API_KEY set): Whisper large-v3-turbo on Groq's free API.
    Laptop mode: faster-whisper on this computer (offline after first download)."""
    if os.environ.get("GROQ_API_KEY"):
        return transcribe_groq(path, language)
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("No speech engine available. Set GROQ_API_KEY (free at console.groq.com) "
                           f"or install faster-whisper. ({e})")
    if model_size not in _models:
        _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = _models[model_size].transcribe(path, word_timestamps=True, vad_filter=True,
                                              language=language or None, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            if w.word.strip():
                words.append({"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2),
                              "p": round(w.probability, 2)})
    return words


def transcribe_groq(path, language=None):
    """Send a small mono audio copy (well under the 25 MB limit) to Groq's Whisper API."""
    import math
    import tempfile
    import uuid
    tmp = os.path.join(tempfile.gettempdir(), f"sc_{uuid.uuid4().hex}.mp3")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", tmp])
    try:
        with open(tmp, "rb") as f:
            audio = f.read()
    finally:
        os.remove(tmp)
    fields = [("model", "whisper-large-v3-turbo"), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment"),
              ("temperature", "0")]
    if language:
        fields.append(("language", language))
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.mp3"\r\n'
             f'Content-Type: audio/mpeg\r\n\r\n').encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", body, {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "SmartCut/2"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Speech API error {e.code}: {e.read().decode(errors='replace')[:300]}")
    segs = data.get("segments") or []

    def conf(t):  # word confidence ≈ confidence of the segment it falls in
        for s in segs:
            if s["start"] - 0.05 <= t <= s["end"] + 0.05:
                return round(min(0.99, math.exp(s.get("avg_logprob", -0.1))), 2)
        return 0.9
    return [{"w": w["word"].strip(), "s": round(w["start"], 2), "e": round(w["end"], 2), "p": conf(w["start"])}
            for w in data.get("words") or [] if w.get("word", "").strip()]


def words_to_segments(words, source_id, duration):
    """Group words into segments. Long silences and sentence ends become cut points."""
    groups, cur = [], []
    for w in words:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            sentence_end = re.search(r"[.?!]$", cur[-1]["w"]) is not None
            if gap > PAUSE_SPLIT or (sentence_end and gap > 0.25):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    segs = []
    for g in groups:
        text = " ".join(x["w"] for x in g).strip()
        segs.append({
            "source": source_id,
            "start": round(max(0, g[0]["s"] - PAD_START), 2),
            "end": round(min(duration or (g[-1]["e"] + 1), g[-1]["e"] + PAD_END), 2),
            "words": g, "text": text, "original_text": text,
            "confidence": round(sum(x["p"] for x in g) / len(g), 2),
            "keep": True, "reason": "", "user": False, "cover": None,
        })
    return segs


# ---------------------------------------------------------------- analysis
def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", "", t.lower())).strip()


def keywords(t):
    return {w for w in norm(t).split() if len(w) > 2 and w not in STOP}


def similar(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    short, long_ = sorted((a, b), key=len)
    if len(short) > 15 and short in long_:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def analyse(project):
    """Auto keep/drop with a human-readable reason for every decision. Editor choices are never overwritten."""
    segs = project["segments"]
    brief_kw = keywords(project.get("brief", ""))
    auto = [s for s in segs if not s.get("user")]
    for s in segs:
        kw = keywords(s["text"])
        s["score"] = round((len(kw & brief_kw) / (len(brief_kw) or 1)) + min(len(kw), 12) / 40, 3)
        s["duplicate_of"] = None
    for s in auto:
        n = norm(s["text"])
        s["keep"], s["reason"] = True, ("Matches the brief" if keywords(s["text"]) & brief_kw else "Relevant speech")
        if n in FILLERS or len(n.split()) <= 1:
            s["keep"], s["reason"] = False, "Filler / too short"
        elif any(c in n for c in CHATTER):
            s["keep"], s["reason"] = False, "Production chatter (mic check / restart), not part of the story"

    by_src = {}
    for s in segs:
        by_src.setdefault(s["source"], []).append(s)
    # false starts: "Our main goal" ... "Our main goal was to solve ..."
    for lst in by_src.values():
        lst.sort(key=lambda x: x["start"])
        for a, b in zip(lst, lst[1:]):
            na, nb = norm(a["text"]), norm(b["text"])
            if not a.get("user") and a["keep"] and na and nb.startswith(na) and len(nb) > len(na):
                a["keep"] = False
                a["reason"] = (f"False start — {b['id']} says it completely" if len(na.split()) <= 4
                               else f"Repeat of {b['id']} — kept the more complete take")

    # repetition: same idea twice -> keep the more complete take
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if not (a["keep"] and b["keep"]) or similar(a["text"], b["text"]) < 0.72:
                continue
            if a["source"] == b["source"]:
                loser, winner = (a, b) if len(norm(b["text"])) >= len(norm(a["text"])) else (b, a)
                if loser.get("user"):
                    continue
                loser["keep"] = False
                loser["reason"] = f"Repeat of {winner['id']} — kept the more complete take"
            else:
                # same moment in two recordings: multi-camera / overlapping explanation
                loser, winner = (a, b) if (b["confidence"], b["end"] - b["start"]) > (a["confidence"], a["end"] - a["start"]) else (b, a)
                loser["duplicate_of"], winner["duplicate_of"] = winner["id"], loser["id"]
                if loser.get("user"):
                    continue
                loser["keep"] = False
                loser["reason"] = f"Same moment also recorded in {winner['id']} ({winner['source']}) — kept the clearer one"
            if not a["keep"]:
                break

    if os.environ.get("GEMINI_API_KEY"):
        try:
            llm_select(project)
        except Exception as e:  # AI is optional; rules already produced a valid cut
            project.setdefault("log", []).append(f"AI selection skipped: {e}")


def llm_select(project):
    """Optional: Gemini picks which kept segments tell the brief's story. It can only drop, never invent."""
    lines = "\n".join(f'{s["id"]}: {s["text"]}' for s in project["segments"] if s["keep"] and not s.get("user"))
    prompt = (f"Brief: {project.get('brief')}\nTranscript segments:\n{lines}\n\n"
              "Which segments are off-topic for the brief? Never drop a segment that another kept segment "
              "depends on for context. Reply ONLY JSON: {\"drop\": {\"<id>\": \"<short reason>\"}}")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    txt = resp["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    by_id = {s["id"]: s for s in project["segments"]}
    for sid, why in data.get("drop", {}).items():
        if sid in by_id and not by_id[sid].get("user"):
            by_id[sid]["keep"], by_id[sid]["reason"] = False, f"AI: {why}"


# ---------------------------------------------------------------- warnings (re-checked after EVERY edit)
def warnings(project):
    out = []
    by_id = {s["id"]: s for s in project["segments"]}
    src = {s["id"]: s for s in project["sources"]}
    dismissed = set(project.get("dismissed", []))
    order = project["order"]
    kept = [by_id[i] for i in order if by_id[i]["keep"]]

    def add(seg, typ, msg, level="warn"):
        if f"{seg}:{typ}" not in dismissed:
            out.append({"seg": seg, "type": typ, "msg": msg, "level": level})

    for s in kept:
        # context: kept line opens with a connector but the line it followed in the recording is gone
        same = sorted([x for x in project["segments"] if x["source"] == s["source"]], key=lambda x: x["start"])
        i = same.index(s)
        prev = same[i - 1] if i else None
        first = norm(s["text"])
        if prev and not prev["keep"] and any(first == c or first.startswith(c + " ") for c in CONNECTORS) \
                and not prev["reason"].startswith(("Repeat", "Filler", "False start", "Production")):
            add(s["id"], "context", f'Starts with "{s["text"].split()[0]}" but the line before it ({prev["id"]}) '
                                    f'was removed — the meaning may change. Restore {prev["id"]} or remove this.')
        if s["confidence"] < LOW_CONF:
            add(s["id"], "caption", "Speech recognition was unsure here — check names / technical words in the caption.")
        if not src[s["source"]]["has_video"] and not s.get("cover"):
            add(s["id"], "visual", "Audio only — there is no picture for this part. A black frame is used "
                                   "(nothing is invented). Tip: cover it with B-roll.")
        if s.get("duplicate_of") and by_id.get(s["duplicate_of"], {}).get("keep"):
            add(s["id"], "duplicate", f'The same moment is also kept from {s["duplicate_of"]} — it will play twice.')

    # order: recorded events shown in a different order than they happened
    last = {}
    for s in kept:
        if s["source"] in last and s["start"] < last[s["source"]]:
            add(s["id"], "order", "Moved before something that was said earlier in the same recording — "
                                  "make sure this does not change what really happened.", "info")
        last[s["source"]] = s["start"]

    # continuity: framing jumps between neighbouring clips
    for a, b in zip(kept, kept[1:]):
        va = src[a["cover"]["source"]] if a.get("cover") else src[a["source"]]
        vb = src[b["cover"]["source"]] if b.get("cover") else src[b["source"]]
        if va["has_video"] and vb["has_video"] and (va["width"] > va["height"]) != (vb["width"] > vb["height"]):
            add(b["id"], "framing", "Framing changes (portrait phone ↔ landscape camera) at this cut — "
                                    "check how it looks in each version.", "info")
        elif a["source"] == b["source"] and b["start"] - a["end"] > 3.0 and va["has_video"] \
                and not a.get("cover") and not b.get("cover"):
            add(b["id"], "jump", "Jump cut (part of the same shot was removed). Tip: cover this clip with B-roll.", "info")
    if not kept and project["segments"]:
        out.append({"seg": None, "type": "empty", "msg": "Nothing is selected for the timeline.", "level": "warn"})
    return out


# ---------------------------------------------------------------- timeline per version
def timeline(project, version):
    """Kept segments mapped onto an output timeline. Every version derives from the same segments."""
    vcfg = project["versions"][version]
    by_id = {s["id"]: s for s in project["segments"]}
    kept = [by_id[i] for i in project["order"] if by_id[i]["keep"]]
    limit = vcfg.get("max_seconds")
    if limit and kept:  # short version: intro + best-scoring segments, kept in story order
        chosen, total = {kept[0]["id"]}, kept[0]["end"] - kept[0]["start"]
        for s in sorted(kept[1:], key=lambda x: -x["score"]):
            d = s["end"] - s["start"]
            if total + d <= limit:
                chosen.add(s["id"])
                total += d
        kept = [s for s in kept if s["id"] in chosen]
    t, items = 0.0, []
    for s in kept:
        d = round(s["end"] - s["start"], 2)
        items.append({"seg": s, "out_start": round(t, 2), "out_end": round(t + d, 2)})
        t += d
    return items, round(t, 2)


def apply_glossary(text, glossary):
    for wrong, right in sorted(glossary.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", right, text, flags=re.I)
    return text


def captions(project, version, max_words=7):
    """Caption cues re-timed onto the output timeline; they move automatically when clips change."""
    items, _ = timeline(project, version)
    gl = project.get("glossary", {})
    cues = []
    for it in items:
        s, off = it["seg"], it["out_start"] - it["seg"]["start"]
        if s["text"] == s["original_text"] and s.get("words"):
            ws = s["words"]
            for i in range(0, len(ws), max_words):
                chunk = ws[i:i + max_words]
                cues.append([max(it["out_start"], chunk[0]["s"] + off),
                             min(it["out_end"], chunk[-1]["e"] + off + 0.15),
                             apply_glossary(" ".join(w["w"] for w in chunk), gl)])
        else:  # caption corrected by the editor: spread evenly over the clip
            words = apply_glossary(s["text"], gl).split()
            n = max(1, -(-len(words) // max_words))
            step = (it["out_end"] - it["out_start"]) / n
            for k in range(n):
                cues.append([it["out_start"] + k * step, it["out_start"] + (k + 1) * step,
                             " ".join(words[k * max_words:(k + 1) * max_words])])
    for i in range(1, len(cues)):  # no overlapping cues
        if cues[i][0] < cues[i - 1][1]:
            cues[i - 1][1] = cues[i][0]
    return [c for c in cues if c[1] - c[0] > 0.05 and c[2].strip()]


def to_srt(cues):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1))


# ---------------------------------------------------------------- render (incremental)
def _clip(project, seg, W, H, workdir, src):
    """Render one segment as a normalised clip (cached by content hash)."""
    s_info = src[seg["source"]]
    cover = seg.get("cover")
    d = round(seg["end"] - seg["start"], 3)
    gain = 0.0
    if s_info.get("lufs") is not None:
        gain = max(-12.0, min(20.0, TARGET_LUFS - s_info["lufs"]))
    key = json.dumps([s_info["file"], seg["start"], seg["end"], W, H, cover, round(gain, 1), 2])
    name = os.path.join("clips", hashlib.md5(key.encode()).hexdigest()[:16] + ".mkv")
    if os.path.exists(os.path.join(workdir, name)):
        return name, False
    os.makedirs(os.path.join(workdir, "clips"), exist_ok=True)
    args = ["ffmpeg", "-y", "-v", "error"]
    v_info = src[cover["source"]] if cover else s_info
    crop = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if v_info.get("width") and (v_info["width"] > v_info["height"]) != (W > H):
        # orientation mismatch (phone portrait in a landscape version, or vice versa):
        # show the whole frame over a blurred copy instead of cutting off faces
        vfit = (f"split[bg][fg];[bg]{crop},boxblur=20:2[bgb];"
                f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps=30,setsar=1,format=yuv420p")
    else:
        vfit = f"{crop},fps=30,setsar=1,format=yuv420p"
    # video input
    if cover:
        c_info = src[cover["source"]]
        args += ["-stream_loop", "-1", "-ss", str(cover.get("offset", 0)), "-t", str(d),
                 "-i", os.path.join("uploads", c_info["file"])]
        vsrc = "0:v"
    elif s_info["has_video"]:
        args += ["-ss", str(seg["start"]), "-t", str(d), "-i", os.path.join("uploads", s_info["file"])]
        vsrc = "0:v"
    else:  # missing visual: honest black frame, never fabricated imagery
        args += ["-f", "lavfi", "-t", str(d), "-i", f"color=c=black:s={W}x{H}:r=30"]
        vsrc = "0:v"
    # audio input (always the speech source)
    if s_info["has_audio"]:
        if cover or not s_info["has_video"]:
            args += ["-ss", str(seg["start"]), "-t", str(d), "-i", os.path.join("uploads", s_info["file"])]
            asrc = "1:a"
        else:
            asrc = "0:a"
        af = (f"[{asrc}]aresample=48000,aformat=channel_layouts=stereo,volume={gain:.1f}dB,"
              f"afade=t=in:d=0.03,afade=t=out:st={max(0, d - 0.06):.3f}:d=0.06,apad,atrim=0:{d}[a]")
    else:
        af = f"anullsrc=r=48000:cl=stereo,atrim=0:{d}[a]"
    fc = f"[{vsrc}]{vfit},trim=0:{d},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={d}[vt];[vt]trim=0:{d}[v];{af}"
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "18", "-g", "30", "-c:a", "pcm_s16le", "-t", str(d), name]
    run(args, cwd=workdir)
    return name, True


def render(project, version, workdir, progress=lambda pct, msg: None):
    vcfg = project["versions"][version]
    W, H = vcfg["width"], vcfg["height"]
    items, total = timeline(project, version)
    if not items:
        raise ValueError("No clips selected — restore at least one clip.")
    src = {s["id"]: s for s in project["sources"]}
    clips, made = [], 0
    for k, it in enumerate(items):
        progress(int(80 * k / len(items)), f"Preparing clip {k + 1} of {len(items)}")
        name, new = _clip(project, it["seg"], W, H, workdir, src)
        clips.append(name)
        made += new
    with open(os.path.join(workdir, f"list_{version}.txt"), "w", encoding="utf-8") as f:
        f.write("".join(f"file '{c.replace(os.sep, '/')}'\n" for c in clips))
    srt = f"out_{version}.srt"
    with open(os.path.join(workdir, srt), "w", encoding="utf-8") as f:
        f.write(to_srt(captions(project, version)))
    progress(85, "Mixing music, balancing audio, adding captions")
    out = f"out_{version}.mp4"
    music = project.get("music")
    base = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", f"list_{version}.txt"]
    if music:
        base += ["-stream_loop", "-1", "-i", os.path.join("uploads", music)]
    fa = ["[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[dlg]"]
    if music:
        fa += [f"[1:a]atrim=0:{total},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
               f"volume={project.get('music_volume', 0.35)},afade=t=in:d=1,afade=t=out:st={max(0, total - 2)}:d=2[mus]",
               "[dlg]asplit=2[d1][d2]",
               # ducking: music automatically dips while someone is speaking
               "[mus][d1]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck]",
               "[d2][duck]amix=inputs=2:duration=first:normalize=0[aout]"]
    else:
        fa += ["[dlg]anull[aout]"]
    fs = 18 if W > H else 13
    burn = f"[0:v]subtitles={srt}:force_style='FontName=DejaVu Sans,FontSize={fs},Bold=1,Outline=2,Shadow=0,MarginV={22 if W > H else 45}'[vout]"
    tail = ["-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "160k", "-t", str(total), "-movflags", "+faststart", out]
    captions_burned = True
    try:
        run(base + ["-filter_complex", ";".join(fa + [burn])] + tail, cwd=workdir)
    except RuntimeError:
        # some FFmpeg builds have no font support: still deliver video + separate .srt
        captions_burned = False
        run(base + ["-filter_complex", ";".join(fa + ["[0:v]null[vout]"])] + tail, cwd=workdir)
    progress(100, "Done")
    return {"mp4": out, "srt": srt, "clips_rendered": made, "clips_reused": len(items) - made,
            "duration": total, "captions_burned": captions_burned}

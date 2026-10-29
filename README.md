# SmartCut v2: RYHATHON PS-05

AI video production web app. Neraiya recordings + oru brief kudutha, story-ku etha rough cut ready pannum. Ovvoru decision-um explain pannum, ellathaiyum edit pannalam. Oru project-la irundhu ella versions-um varum.

## Online deploy (Render.com, free)
1. GitHub-la pudhu repo create panni indha folder-la irukura ellathaiyum upload pannunga
2. render.com → GitHub login → **New → Blueprint** → repo select → **Apply** (`render.yaml` free plan-a set pannum)
3. (Optional) console.groq.com-la free API key create panni, Render-la `GROQ_API_KEY`-a set pannunga. Idhu irundha pudhu uploads-um transcribe aagum.
4. Link: `https://smartcut-xxxx.onrender.com`. **▶ Try the demo project** key illama-ve work aagum.

## Run (Windows)
1. Python 3.10+ ("Add Python to PATH" tick pannunga) and FFmpeg (PATH-la `C:\ffmpeg\bin`)
2. Indha folder-la oru thadava: `pip install -r requirements.txt`
3. **`run_windows.bat` double-click.** Free port-a automatic-a eduthu browser-a open pannum (port 8000 busy-na 8001 use pannum).

First time "Build rough cut" pannumbodhu speech model download aagum (~140 MB, internet venum).

## Demo
`demo_media` folder-la test files + `brief.txt` iruku:
1. 01–05 files-a **recordings** box-la drop pannunga, 06 music-a **music** box-la drop pannunga
2. `brief.txt`-la irundhu brief + caption fixes copy pannunga, Language: English
3. **✨ Build rough cut**, apram ovvoru clip-um yen keep/remove aachu-nu paarunga
4. Try pannunga: S9-a remove panni S10-la context warning varudha paarunga, caption fix, ⭐ Make intro, B-roll picture, ↶ Undo
5. **🎬 Render all versions** → play / download

## Features → problem statement
| Requirement | SmartCut |
|---|---|
| Identify meaningful segments | Whisper transcript + brief keyword relevance |
| Pauses / fillers | Silence-gap splitting, "um/uh" removal |
| Repetition without changing meaning | Similarity match, keeps the *more complete* take |
| False starts, mic checks | Prefix detection, production-chatter detection |
| Inconsistent audio levels | Per-recording loudness measure (EBU R128) + gain, final loudnorm |
| Music overpowering dialogue | Sidechain ducking (music dips while people speak) + volume slider |
| Incorrect names / technical terms | Low-confidence warnings, click-to-edit captions, caption-fix list |
| Changes propagate | One project.json → captions re-timed, music re-fitted, all versions marked out-of-date and rebuilt |
| Don't rebuild everything | Clip cache: only changed clips re-render |
| Multiple versions | 16:9 presentation, 9:16 reels, 15–60s teaser |
| Revised introduction | ⭐ Make intro (any clip from any recording) |
| Multi-camera / same moment | Cross-recording duplicate detection, keeps the clearer take |
| Missing visuals | Audio-only → honest black frame + warning; B-roll cover option |
| Framing / continuity | Portrait↔landscape blur-fit, framing & jump-cut notes |
| Order of events | Warning when a clip is moved before something recorded earlier |
| Integrity / inspect | ▶ Original shows source file + timestamp + recognised text + confidence; reasons for every cut; change history; undo |
| Editable project | Download project.json, not just an mp4 |

## Optional
- `GEMINI_API_KEY` set panna Gemini off-topic segments-a suggest pannum (drop mattum, edhuvum invent pannadhu).
- Accuracy "Better" = Whisper `small` model (slow, accurate).

## Code
- `app/pipeline.py`: transcription, analysis, warnings, captions, incremental FFmpeg render
- `app/main.py`: FastAPI API + background jobs with progress
- `app/static/index.html`: web UI
- `run.py`: launcher

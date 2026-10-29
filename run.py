"""SmartCut launcher: checks setup, picks a free port, opens the browser."""
import shutil, socket, sys, threading, time, webbrowser

def free_port(start=8000):
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port)); return port
            except OSError:
                continue
    return 0

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    print("\n[!] FFmpeg not found. Install FFmpeg and add its 'bin' folder to PATH (see README), then run again.\n")
    input("Press Enter to close..."); sys.exit(1)
try:
    import fastapi, uvicorn, multipart  # noqa
except ImportError as e:
    print(f"\n[!] Missing package ({e.name}). Run:  pip install -r requirements.txt\n")
    input("Press Enter to close..."); sys.exit(1)
import os
if not os.environ.get("GROQ_API_KEY"):
    try:
        import faster_whisper  # noqa
    except Exception as e:
        print(f"\n[!] Local speech engine not working ({type(e).__name__}).")
        print("    Fix: install 'VC++ Redistributable' (aka.ms/vs/17/release/vc_redist.x64.exe)")
        print("    or set a free Groq key:  set GROQ_API_KEY=your_key   then run again.")
        print("    The app still opens and the demo project works.\n")

port = free_port()
url = f"http://localhost:{port}"
print(f"\n  SmartCut is running at  {url}\n  Keep this window open. Press Ctrl+C to stop.\n")
threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")

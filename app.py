import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Set

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    send_from_directory,
    request,
    url_for,
)

import shutil
import shlex

BASE_DIR = Path(__file__).resolve().parent
PHOTO_DIR = BASE_DIR / "photos"
PREVIEW_LOG = BASE_DIR / "preview.log"

PHOTO_DIR.mkdir(exist_ok=True)
PREVIEW_LOG.touch(exist_ok=True)

COMMAND_PREFIX = os.environ.get("PHOTOBOOTH_COMMAND_PREFIX", "").strip()
COMMAND_PREFIX_PARTS = COMMAND_PREFIX.split() if COMMAND_PREFIX else []

GPHOTO_BINARY = os.environ.get("PHOTOBOOTH_GPHOTO_BINARY", "gphoto2")
USB_RESET_COMMAND = os.environ.get("PHOTOBOOTH_USB_RESET_COMMAND", "").strip() # usbreset ""Canon Digital Camera""
RESTART_COMMAND = os.environ.get("PHOTOBOOTH_RESTART_COMMAND", "").strip()
PREVIEW_BOUNDARY = b"frame"
SOI_MARKER = b"\xff\xd8"  # JPEG start of image
EOI_MARKER = b"\xff\xd9"  # JPEG end of image

app = Flask(__name__, static_folder="photos", static_url_path="/photos")


def build_command(*args: str) -> Iterable[str]:
    return [*COMMAND_PREFIX_PARTS, *args]


@dataclass
class PreviewSession:
    token: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    processes: Set[subprocess.Popen] = field(default_factory=set)


class PreviewManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, PreviewSession] = {}
        self._active_token: str | None = None

    def start_session(self) -> PreviewSession:
        with self._lock:
            self._stop_all_locked()
            token = uuid.uuid4().hex
            session = PreviewSession(token=token)
            self._sessions[token] = session
            self._active_token = token
            return session

    def stop_all(self) -> None:
        with self._lock:
            self._stop_all_locked()

    def stop_session(self, token: str) -> None:
        with self._lock:
            self._stop_session_locked(token)

    def get_session(self, token: str) -> PreviewSession | None:
        with self._lock:
            return self._sessions.get(token)

    def register_process(self, token: str, process: subprocess.Popen) -> None:
        with self._lock:
            session = self._sessions.get(token)
            if session is not None:
                session.processes.add(process)

    def release_process(self, token: str, process: subprocess.Popen) -> None:
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return
            session.processes.discard(process)
            if not session.processes and session.stop_event.is_set():
                self._cleanup_session_locked(token)

    def _stop_all_locked(self) -> None:
        for token in list(self._sessions.keys()):
            self._stop_session_locked(token)

    def _stop_session_locked(self, token: str) -> None:
        session = self._sessions.get(token)
        if session is None:
            return
        session.stop_event.set()
        for process in list(session.processes):
            terminate_process(process)
        self._cleanup_session_locked(token)

    def _cleanup_session_locked(self, token: str) -> None:
        self._sessions.pop(token, None)
        if self._active_token == token:
            self._active_token = None


preview_manager = PreviewManager()


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except OSError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        except OSError:
            pass
        process.wait(timeout=1)


def ensure_binaries_available() -> None:
    binaries = []
    if COMMAND_PREFIX_PARTS:
        binaries.append(COMMAND_PREFIX_PARTS[0])
    binaries.append(GPHOTO_BINARY)

    missing = [binary for binary in binaries if shutil.which(binary) is None]
    if missing:
        raise RuntimeError(f"Missing required binaries: {', '.join(missing)}")


def read_log_tail(limit_bytes: int = 4096) -> str:
    try:
        with PREVIEW_LOG.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - limit_bytes, 0))
            data = handle.read()
        return data.decode(errors="replace")
    except OSError:
        return ""


def attempt_usb_reset(log_handle) -> bool:
    if not USB_RESET_COMMAND:
        return False

    try:
        command = shlex.split(USB_RESET_COMMAND)
    except ValueError:
        log_handle.write(b"[photobooth] Invalid USB reset command configuration.\n")
        log_handle.flush()
        return False

    log_handle.write(b"[photobooth] Attempting USB reset...\n")
    log_handle.flush()

    try:
        result = subprocess.run(
            command,
            stdout=log_handle,
            stderr=log_handle,
        )
    except FileNotFoundError:
        log_handle.write(b"[photobooth] USB reset command not found.\n")
        log_handle.flush()
        return False

    log_handle.write(
        f"[photobooth] USB reset exit code: {result.returncode}\n".encode()
    )
    log_handle.flush()
    return result.returncode == 0


def attempt_restart(log_handle) -> None:
    if not RESTART_COMMAND:
        return

    try:
        command = shlex.split(RESTART_COMMAND)
    except ValueError:
        log_handle.write(b"[photobooth] Invalid restart command configuration.\n")
        log_handle.flush()
        return

    log_handle.write(b"[photobooth] Initiating system restart...\n")
    log_handle.flush()

    try:
        subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=(os.name != "nt"),
        )
    except FileNotFoundError:
        log_handle.write(b"[photobooth] Restart command not found.\n")
        log_handle.flush()


def generate_preview(session: PreviewSession):
    command = list(build_command(
        GPHOTO_BINARY,
        "--capture-preview",
        "--stdout",
        "-F",
        "1",
    ))

    log_handle = PREVIEW_LOG.open("ab", buffering=0)
    consecutive_failures = 0
    reset_attempted = False

    try:
        while not session.stop_event.is_set():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=log_handle,
                    bufsize=0,
                    start_new_session=(os.name != "nt"),
                )
            except FileNotFoundError:
                yield error_frame("Unable to start preview process (gphoto2 not found)")
                return

            preview_manager.register_process(session.token, process)

            try:
                stdout, _ = process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                terminate_process(process)
                stdout = b""
            finally:
                preview_manager.release_process(session.token, process)

            frame = None
            if stdout:
                start = stdout.find(SOI_MARKER)
                if start != -1:
                    end = stdout.find(EOI_MARKER, start + 2)
                    if end != -1:
                        frame = stdout[start:end + 2]

            if frame:
                consecutive_failures = 0
                reset_attempted = False
                yield (
                    b"--" + PREVIEW_BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                )
            else:
                consecutive_failures += 1

                if (
                    USB_RESET_COMMAND
                    and consecutive_failures >= 3
                    and not reset_attempted
                    and not session.stop_event.is_set()
                ):
                    success = attempt_usb_reset(log_handle)
                    reset_attempted = True
                    if success:
                        time.sleep(1.5)
                        continue

                if consecutive_failures >= 5 and not session.stop_event.is_set():
                    attempt_restart(log_handle)
                    yield error_frame(read_log_tail())
                    return

            if not session.stop_event.is_set():
                time.sleep(0.1)
    finally:
        log_handle.close()


def error_frame(message: str) -> bytes:
    text = message or "Preview pipeline did not produce output. Check camera connection."
    return (
        b"--" + PREVIEW_BOUNDARY + b"\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        + text.encode(errors="replace")
        + b"\r\n"
    )


@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/animate.min.css")
def animate():
    # Assuming animate.min.css is inside the "static" folder
    return send_from_directory("static", "animate.min.css", mimetype="text/css")


@app.post("/api/preview/start")
def api_preview_start():
    try:
        ensure_binaries_available()
    except RuntimeError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500

    session = preview_manager.start_session()
    stream_url = url_for("preview_stream", token=session.token)
    return jsonify({"status": "ok", "stream_url": stream_url})


@app.post("/api/preview/stop")
def api_preview_stop():
    preview_manager.stop_all()
    return jsonify({"status": "ok"})


@app.get("/preview/stream")
def preview_stream():
    token = request.args.get("token")
    if not token:
        abort(400, "Missing preview session token")

    session = preview_manager.get_session(token)
    if session is None:
        abort(404, "Preview session not found or stopped")

    return Response(
        generate_preview(session),
        mimetype=f"multipart/x-mixed-replace; boundary={PREVIEW_BOUNDARY.decode()}"
    )


@app.post("/api/capture")
def api_capture():
    preview_manager.stop_all()

    timestamp = uuid.uuid4().hex
    filename = f"photo_{timestamp}.jpg"
    filepath = PHOTO_DIR / filename

    command = list(build_command(
        GPHOTO_BINARY,
        "--capture-image-and-download",
        "--force-overwrite",
        "--filename",
        str(filepath),
    ))

    with PREVIEW_LOG.open("ab", buffering=0) as log_handle:
        result = subprocess.run(
            command,
            stdout=log_handle,
            stderr=log_handle,
        )

    if result.returncode != 0 or not filepath.exists():
        return (
            jsonify({
                "status": "error",
                "error": "Failed to capture image",
                "details": read_log_tail(),
            }),
            500,
        )

    return jsonify({
        "status": "ok",
        "message": "Photo captured",
        "photo_url": url_for("static", filename=filename),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), threaded=True)

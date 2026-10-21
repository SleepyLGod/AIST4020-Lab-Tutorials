"""Notebook plumbing, not an agent implementation. Uses the Python standard library."""

from __future__ import annotations

import codecs
import copy
from dataclasses import dataclass, field
import difflib
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from typing import Callable, Mapping, Sequence
import urllib.error
import urllib.request
import urllib.parse
import uuid
import zipfile


class Redactor:
    """Remove supplied credentials before displaying or saving subprocess output."""

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        self.secrets = tuple(sorted({s for s in secrets if s}, key=len, reverse=True))

    def __call__(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(json.dumps(secret)[1:-1], "[REDACTED]")
            text = text.replace(secret, "[REDACTED]")
        return text


def credential(name: str) -> str:
    """Use environment, then Colab Secrets, then hidden input; never print the key."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata
    except ImportError:
        userdata = None
    if userdata is not None:
        try:
            value = (userdata.get(name) or "").strip()
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError, TimeoutError):
            print(f"{name} is not available through Colab Secrets; use hidden input.")
    if not value:
        value = getpass.getpass(f"Enter {name}: ")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value.strip()


def isolated_environment(root: Path) -> dict[str, str]:
    """Create dedicated OpenCode state without inheriting other provider secrets."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
           "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "OPENCODE_DISABLE_AUTOUPDATE": "true"}
    for name, folder in {"HOME": "home", "XDG_CONFIG_HOME": "config",
                         "XDG_DATA_HOME": "data", "XDG_CACHE_HOME": "cache"}.items():
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def fresh_project(source: Path, parent: Path) -> Path:
    """Copy the fixture to a new directory; never reset or remove an old project."""
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="grade-report-", dir=parent))
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    return destination


def save_json(path: Path, value: object, redact: Redactor) -> None:
    """Save credential-redacted JSON; other personal content still needs review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(json.dumps(value, indent=2, ensure_ascii=False)) + "\n", encoding="utf-8")


def prepare_lab(lab_dir: Path, version: str) -> tuple[Path, dict[str, str], Path, Path, Path]:
    """Install the pinned CLI and make fresh state and project directories."""
    root = Path(tempfile.mkdtemp(prefix="opencode-course-"))
    env = isolated_environment(root)
    records = root / "records"
    records.mkdir()
    binary = install_opencode(root, version, env)
    project = fresh_project(lab_dir / "grade-report", root)
    print(f"Project: {project}\nRecords: {records}")
    return root, env, binary, project, records


def select_backend(backend: str, backends: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Validate one explicit endpoint without contacting it or choosing a fallback."""
    if backend not in backends:
        raise ValueError("Choose a backend listed in BACKENDS.")
    selected = {k: v.strip() for k, v in backends[backend].items()}
    if not selected.get("base_url") or not selected.get("model"):
        raise ValueError("Fill in base_url and model for the selected backend.")
    endpoint = urllib.parse.urlsplit(selected["base_url"])
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("base_url must be an HTTP(S) endpoint.")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Keep credentials and query parameters out of base_url; use key_env.")
    return selected


def start_ollama(root: Path, model: str, context_tokens: int, env: Mapping[str, str]) -> None:
    """Prepare a loopback Ollama server in a disposable Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("This installation path is for Colab. Use custom for an existing local server.") from exc
    for command in (["apt-get", "update"], ["apt-get", "install", "-y", "zstd", "curl", "ca-certificates"]):
        run_command(command, cwd=root, env=env, timeout=240).require_success()
    if shutil.which("ollama", path=env["PATH"]) is None:
        installer = root / "install-ollama.sh"
        with urllib.request.urlopen("https://ollama.com/install.sh", timeout=60) as response:
            installer.write_bytes(response.read())
        run_command(["sh", str(installer)], cwd=root, env=env, timeout=300).require_success()
    server_env = {**env, "OLLAMA_HOST": "127.0.0.1:11434",
                  "OLLAMA_CONTEXT_LENGTH": str(context_tokens), "OLLAMA_MODELS": str(root / "models"),
                  "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1"}
    for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"):
        if name in os.environ:
            server_env[name] = os.environ[name]
    try:
        wait_for_ollama("http://127.0.0.1:11434", None, timeout=2)
        print("Using the Ollama server already running in this runtime; its server settings remain unchanged.")
    except TimeoutError:
        log_path = root / "ollama.log"
        print(f"Working directory: {root}\n$ ollama serve\nServer log: {log_path}")
        with log_path.open("w") as log:
            process = subprocess.Popen(["ollama", "serve"], cwd=root, env=server_env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_for_ollama("http://127.0.0.1:11434", process)
    run_command(["ollama", "pull", model], cwd=root, env=server_env, timeout=1200).require_success()


def configure_backend(backend: str, backends: Mapping[str, Mapping[str, str]], *,
                      root: Path, env: dict[str, str], context_tokens: int,
                      enabled: bool) -> tuple[dict[str, str], Redactor]:
    """Load only the chosen credential; optionally prepare the local model server."""
    if not enabled:
        raise RuntimeError("Model calls are disabled. Set ENABLE_MODEL_CALLS = True when ready.")
    selected = select_backend(backend, backends)
    env.pop("COURSE_API_KEY", None)
    if backend == "ollama":
        if selected["base_url"].rstrip("/") != "http://127.0.0.1:11434/v1":
            raise ValueError("Use custom to connect to another Ollama address.")
        start_ollama(root, selected["model"], context_tokens, env)
        key = "ollama-local"
    else:
        key_name = selected.get("key_env", "")
        key = credential(key_name) if key_name else "local-no-key"
    env["COURSE_API_KEY"] = key
    print(f"Selected provider: {backend} | Model: {selected['model']}")
    return selected, Redactor([key])


def write_configuration(root: Path, env: dict[str, str], selected: Mapping[str, str], *,
                        provider: str, permissions: dict, context_tokens: int,
                        output_tokens: int, steps: int, redact: Redactor) -> tuple[str, dict[str, str]]:
    """Save repair and no-tools configurations with an environment key reference."""
    model_ref = f"course/{selected['model']}"
    request_timeout_ms = 180000 if provider == "ollama" else 60000
    config = {
        "$schema": "https://opencode.ai/config.json", "model": model_ref, "small_model": model_ref,
        "enabled_providers": ["course"], "share": "disabled", "snapshot": False, "autoupdate": False,
        "permission": permissions, "agent": {"build": {"steps": steps}},
        "provider": {"course": {
            "npm": "@ai-sdk/openai-compatible", "name": provider,
            "options": {"baseURL": selected["base_url"], "apiKey": "{env:COURSE_API_KEY}", "timeout": request_timeout_ms},
            "models": {selected["model"]: {"name": selected["model"], "tool_call": True,
                                          "limit": {"context": context_tokens, "output": output_tokens}}},
        }},
    }
    config_path = root / "opencode.json"
    save_json(config_path, config, redact)
    env["OPENCODE_CONFIG"] = str(config_path)
    chat = copy.deepcopy(config)
    chat["permission"] = {"*": "deny"}
    chat["agent"]["build"]["permission"] = {"*": "deny"}
    chat_path = root / "opencode-no-tools.json"
    save_json(chat_path, chat, redact)
    print("Repair permissions:", json.dumps(permissions, indent=2))
    return model_ref, {**env, "OPENCODE_CONFIG": str(chat_path)}


def package_records(records: Path, redact: Redactor) -> Path:
    """Package only named lab records, masking known keys again; never include config."""
    allowed = re.compile(r"(?:(?:hello|repair|same|fresh)-[0-9a-f]{32}\.(?:jsonl|process\.json)|"
                         r"(?:continued|fresh)-[0-9a-f]{32}\.session\.json|summary-[0-9a-f]{32}\.json)")
    files = [p for p in sorted(records.iterdir())
             if not p.is_symlink() and p.is_file() and allowed.fullmatch(p.name)]
    if not files:
        raise ValueError("No lab records found. Run a task before preparing a download.")
    destination = records.parent / f"lab-records-{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.writestr(path.name, redact(path.read_text(encoding="utf-8")))
    print("Packaged records:", ", ".join(path.name for path in files))
    return destination


def show_details(title: str, text: str, *, opened: bool = False) -> None:
    """Render escaped notebook output, with a plain-terminal fallback."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        print(f"{title}\n{text}")
        return
    if get_ipython() is None:
        print(f"{title}\n{text}")
        return
    flag = " open" if opened else ""
    display(HTML(f"<details{flag}><summary>{html.escape(title)}</summary>"
                 f"<pre style='white-space:pre-wrap;overflow-wrap:anywhere'>{html.escape(text)}</pre></details>"))


def event_summary(event: dict) -> str:
    """Describe only what the event says; a completed tool is not a passing test."""
    kind = event.get("type", "unknown")
    part = event.get("part") or {}
    if kind == "tool_use":
        state = part.get("state") or {}
        args = state.get("input") or {}
        target = args.get("filePath") or args.get("command") or args.get("pattern") or ""
        exit_code = (state.get("metadata") or {}).get("exit")
        suffix = f"; command exit={exit_code}" if exit_code is not None else ""
        return f"{part.get('tool', 'unknown tool')} [{state.get('status', 'unknown')}]{suffix}: {target}"
    if kind == "step_start":
        return "Model step started (no tool completion reported yet)."
    if kind == "step_finish":
        return f"Model step ended: {part.get('reason', 'reason not reported')}"
    if kind == "text":
        return "Assistant response"
    if kind == "error":
        return "OpenCode reported an error"
    return f"Event: {kind}"


def _table_html(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    style = "text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #bbb;white-space:pre-wrap;overflow-wrap:anywhere"
    headings = "".join(f'<th scope="col" style="{style}">{html.escape(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="{style}">{html.escape(v)}</td>' for v in row)
                   + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;table-layout:fixed">'
            f'<caption style="text-align:left;font-weight:bold">{html.escape(title)}</caption>'
            f'<thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>')


def show_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]], *,
               redact: Redactor | None = None) -> None:
    """Display escaped observations side by side, with a terminal fallback."""
    mask = redact or Redactor()
    title = mask(title)
    headers = [mask(h) for h in headers]
    rows = [[mask(v) for v in row] for row in rows]
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        get_ipython = None
    if get_ipython is not None and get_ipython() is not None:
        display(HTML(_table_html(title, headers, rows)))
    else:
        print(title)
        for row in rows:
            print(" | ".join(f"{heading}: {value}" for heading, value in zip(headers, row)))


def show_project(project: Path) -> None:
    """List the actual project entries and explain the four supplied files."""
    roles = {"README.md": "requirements", "grades.py": "program to repair",
             "scores.csv": "input grades", "test_grades.py": "test of the requirement"}
    lines = [f"{project.name}/"]
    for path in sorted(project.iterdir()):
        suffix = "/" if path.is_dir() else ""
        lines.append(f"  {path.name}{suffix}  {roles.get(path.name, '')}".rstrip())
    for name in roles:
        if not (project / name).is_file():
            lines.append(f"  [Missing file: {name}]")
    show_details("Project files", "\n".join(lines), opened=True)


@dataclass
class RunResult:
    """A process outcome and its reported events, not a task correctness verdict."""

    returncode: int
    timed_out: bool
    seconds: float
    stdout: str
    stderr: str
    events: list[dict] = field(default_factory=list)

    @property
    def session_id(self) -> str | None:
        ids = {e["sessionID"] for e in self.events if e.get("sessionID")}
        if len(ids) > 1:
            raise ValueError("More than one session ID in this run; inspect the saved events.")
        return next(iter(ids), None)

    def require_success(self) -> None:
        """Reject process/API errors; this does not certify the agent's answer."""
        if self.timed_out or self.returncode != 0 or any(e.get("type") == "error" for e in self.events):
            raise RuntimeError("The command did not complete successfully. Read the output above.")


def run_command(args: Sequence[str], *, cwd: Path, env: Mapping[str, str],
                timeout: float = 300, redact: Redactor | None = None,
                event_log: Path | None = None, show: bool = True,
                on_event: Callable[[dict], None] | None = None) -> RunResult:
    """Run one command, stream real CLI events, and stop its process group on timeout."""
    mask = redact or Redactor()
    args = [str(arg) for arg in args]
    if show:
        print(mask(f"Working directory: {cwd}\n$ {shlex.join(args)}"), flush=True)
    if event_log:
        event_log.parent.mkdir(parents=True, exist_ok=True)
        if event_log.exists():
            raise FileExistsError(f"Choose a new record name: {event_log}")
    events: list[dict] = []
    output: dict[str, list[str]] = {"stdout": [], "stderr": []}
    pending = {"stdout": "", "stderr": ""}
    decoders = {k: codecs.getincrementaldecoder("utf-8")("replace") for k in pending}
    log = event_log.open("x", encoding="utf-8") if event_log else None

    def consume(channel: str, line: str) -> None:
        clean = mask(line)
        output[channel].append(clean)
        if channel == "stdout" and event_log:
            try:
                event = json.loads(clean)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and isinstance(event.get("type"), str):
                events.append(event)
                assert log is not None
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                log.flush()
                if on_event:
                    on_event(event)
                if show:
                    print(f"{len(events):02d}. {event_summary(event)}", flush=True)
                    part = event.get("part") or {}
                    if event["type"] == "text":
                        print(part.get("text", ""), flush=True)
                    elif event["type"] in {"tool_use", "error"}:
                        state = part.get("state") or {}
                        failed = event["type"] == "error" or state.get("status") == "error"
                        if part.get("tool") == "bash":
                            # Show test output even when no exit metadata was supplied.
                            failed = True
                        if event["type"] == "tool_use":
                            show_tool_event(event, opened=failed)
                        else:
                            show_details("OpenCode error", json.dumps(event, indent=2), opened=True)
                return
            if show and clean.strip():
                print(f"[CLI output, not a JSON event] {clean}", end="", flush=True)
        elif show and clean.strip():
            print(clean, end="", flush=True)

    started = time.monotonic()
    proc = None
    timed_out = False
    try:
        proc = subprocess.Popen(args, cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        assert proc.stdout is not None and proc.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                elapsed = time.monotonic() - started
                if elapsed >= timeout and not timed_out:
                    timed_out = True
                    _signal_group(proc.pid, signal.SIGTERM)
                if timed_out and elapsed >= timeout + 2:
                    _signal_group(proc.pid, signal.SIGKILL)
                for key, _ in selector.select(timeout=0.1):
                    channel = key.data
                    chunk = os.read(key.fd, 65536)
                    pending[channel] += decoders[channel].decode(chunk, final=not chunk)
                    while "\n" in pending[channel]:
                        line, pending[channel] = pending[channel].split("\n", 1)
                        consume(channel, line + "\n")
                    if not chunk:
                        if pending[channel]:
                            consume(channel, pending[channel])
                        selector.unregister(key.fileobj)
            remaining = max(0.1, timeout - (time.monotonic() - started))
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
    finally:
        if proc is not None:
            if proc.poll() is None:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            proc.stdout.close()
            proc.stderr.close()
        if log:
            log.close()
    assert proc is not None
    result = RunResult(proc.returncode, timed_out, round(time.monotonic() - started, 2),
                       "".join(output["stdout"]), "".join(output["stderr"]), events)
    if event_log:
        save_json(event_log.with_suffix(".process.json"), {
            "returncode": result.returncode, "timed_out": timed_out, "seconds": result.seconds,
            "stdout": result.stdout, "stderr": result.stderr,
        }, mask)
    if show:
        print(f"\nProcess exit={result.returncode}; timeout={timed_out}; elapsed={result.seconds}s")
    return result


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return  # The process group already exited.


def install_opencode(root: Path, version: str, env: Mapping[str, str]) -> Path:
    """Install the pinned Linux x86-64 release in the lab's own directory."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("This installation cell targets Colab Linux x86-64.")
    directory = root / f"opencode-{version}"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "opencode"
    if not binary.exists():
        url = f"https://github.com/anomalyco/opencode/releases/download/v{version}/opencode-linux-x64.tar.gz"
        print(f"Downloading the pinned release: {url}")
        archive_path = directory / "release.tar.gz"
        with urllib.request.urlopen(url, timeout=90) as response, archive_path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        with tarfile.open(archive_path, "r:gz") as archive:
            matches = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == "opencode"]
            if len(matches) != 1:
                raise ValueError("Release archive does not contain one unambiguous OpenCode binary.")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError("Could not read the release binary.")
            with stream, binary.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
        binary.chmod(0o755)
    result = run_command([str(binary), "--version"], cwd=root, env=env, timeout=30)
    result.require_success()
    if result.stdout.strip() != version:
        raise RuntimeError(f"Expected OpenCode {version}; the reported version differs.")
    return binary


def wait_for_ollama(url: str, process: subprocess.Popen | None, timeout: float = 60) -> None:
    """Wait for the local server, reporting an unavailable server rather than proceeding."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("Ollama exited. Read the Ollama log printed by the setup cell.")
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=2) as response:
                json.load(response)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Ollama did not become ready; inspect its log before continuing.")


def snapshot(folder: Path) -> dict[str, dict[str, str]]:
    """Capture all project entries except Python bytecode, without following symlinks."""
    entries = {}
    for path in sorted(folder.rglob("*")):
        if "__pycache__" in path.relative_to(folder).parts:
            continue
        name = str(path.relative_to(folder))
        if path.is_symlink():
            entries[name] = {"kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            data = path.read_bytes()
            entries[name] = {"kind": "file", "sha256": hashlib.sha256(data).hexdigest(),
                             "text": data.decode("utf-8", errors="replace")}
        elif path.is_dir():
            entries[name] = {"kind": "directory"}
    return entries


def show_changes(before: dict, after: dict, redact: Redactor) -> list[str]:
    """Display actual changes; do not label an edit as correct merely because it exists."""
    changed = [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
    for name in changed:
        old, new = before.get(name, {}), after.get(name, {})
        diff = "".join(difflib.unified_diff(old.get("text", "").splitlines(True),
                                          new.get("text", "").splitlines(True),
                                          fromfile=f"before/{name}", tofile=f"after/{name}"))
        show_details(redact(f"Changed: {name}"), redact(diff or f"{old.get('kind', 'absent')} -> {new.get('kind', 'absent')}"), opened=True)
    if not changed:
        print("No project file changes found.")
    return changed


def show_messages(transcript: dict, redact: Redactor) -> None:
    """Show saved user/assistant text and tool records, not hidden model reasoning."""
    if not isinstance(transcript.get("messages"), list):
        raise ValueError("The export has no messages list; inspect the export format.")
    for message in transcript["messages"]:
        info = message.get("info", {})
        for part in message.get("parts", []):
            if part.get("type") == "text":
                label = f"{info.get('role', 'unknown')} ({info.get('id', 'unknown')})"
                show_details(redact(label), redact(part.get("text", "")), opened=info.get("role") == "user")
            elif part.get("type") == "tool":
                show_details(redact(f"Saved tool: {part.get('tool', 'unknown')}"), redact(json.dumps(part, indent=2)))


@dataclass
class Lab:
    """Hold notebook setup and records; each run delegates to the real OpenCode CLI."""

    root: Path
    env: dict[str, str]
    binary: Path
    project: Path
    records: Path
    version: str
    redact: Redactor = field(default_factory=Redactor)
    model: str = ""
    chat_env: dict[str, str] = field(default_factory=dict)
    enabled: bool = False

    def configure(self, backend: str, backends: Mapping[str, Mapping[str, str]], *,
                  permissions: dict, enabled: bool) -> None:
        """Configure only the selected provider, using the course's existing limits."""
        self.enabled = False
        context_tokens = 16384 if backend == "ollama" else 32768
        selected, self.redact = configure_backend(
            backend, backends, root=self.root, env=self.env,
            context_tokens=context_tokens, enabled=enabled,
        )
        self.model, self.chat_env = write_configuration(
            self.root, self.env, selected, provider=backend, permissions=permissions,
            context_tokens=context_tokens, output_tokens=2048, steps=12, redact=self.redact,
        )
        self.enabled = enabled
        self.save_record({"opencode_version": self.version, "provider": backend,
                          "model": selected["model"], "permissions": permissions})
        print(f"Model: {self.model}")

    def run(self, prompt: str, *, session_id: str | None = None,
            tools: bool = True, label: str = "repair") -> RunResult:
        """Run one visible CLI command, without retrying or completing missing actions."""
        if not self.enabled:
            raise RuntimeError("Model calls are disabled. Configure the selected provider first.")
        if label not in {"hello", "repair", "same", "fresh"}:
            raise ValueError("Choose a known lab record label.")
        args = [str(self.binary), "run", "--pure", "--format", "json",
                "--agent", "build", "--model", self.model]
        if session_id is not None:
            if not session_id.strip():
                raise ValueError("A continued conversation needs a nonempty session ID.")
            args += ["--session", session_id]
        result = run_command(
            [*args, prompt], cwd=self.project, env=self.env if tools else self.chat_env,
            timeout=300, redact=self.redact,
            event_log=self.records / f"{label}-{uuid.uuid4().hex}.jsonl",
        )
        if not tools:
            result.require_success()
            if any(e.get("type") == "tool_use" for e in result.events):
                raise RuntimeError("A no-tools request reported a tool action. Inspect its output.")
            if not any(e.get("type") == "text" and (e.get("part") or {}).get("text", "").strip()
                       for e in result.events):
                raise RuntimeError("No text reply was reported. Inspect the output.")
        return result

    def check(self, command: Sequence[str]) -> RunResult:
        """Run a student's explicit check without the model credential or configuration."""
        env = {k: v for k, v in self.env.items() if k not in {"COURSE_API_KEY", "OPENCODE_CONFIG"}}
        return run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)

    def export(self, session_id: str, *, label: str, show: bool = True) -> dict:
        """Export and display saved messages; do not infer what the model received."""
        if not session_id or label not in {"continued", "fresh"}:
            raise ValueError("Provide a session ID and a continued/fresh label.")
        command = [str(self.binary), "export", session_id]
        print(self.redact(f"Working directory: {self.project}\n$ {shlex.join(command)}"))
        result = run_command(command, cwd=self.project,
                             env=self.chat_env, redact=self.redact, timeout=60, show=False)
        if result.stderr or result.returncode != 0 or result.timed_out:
            show_details("Export output", result.stdout + result.stderr, opened=True)
        result.require_success()
        transcript = json.loads(result.stdout)
        if not isinstance(transcript, dict) or not isinstance(transcript.get("messages"), list):
            raise ValueError("The session export has no messages list.")
        save_json(self.records / f"{label}-{uuid.uuid4().hex}.session.json", transcript, self.redact)
        if show:
            print(f"Conversation: {label}")
            show_messages(transcript, self.redact)
        return transcript

    def save_record(self, values: Mapping[str, object]) -> Path:
        """Save a small redacted record alongside the actual CLI output."""
        path = self.records / f"summary-{uuid.uuid4().hex}.json"
        save_json(path, dict(values), self.redact)
        return path

    def download_records(self) -> None:
        """Download the existing allowlisted records, or print the archive path locally."""
        bundle = package_records(self.records, self.redact)
        try:
            from google.colab import files
        except ImportError:
            print(f"Download file: {bundle}")
        else:
            files.download(str(bundle))


def open_lab(lab_dir: Path, version: str) -> Lab:
    """Prepare a fresh project and pinned CLI without contacting a model provider."""
    return Lab(*prepare_lab(lab_dir, version), version=version)


def has_user_message(transcript: dict, text: str) -> bool:
    """Check saved user text, excluding assistant guesses and tool outputs."""
    return any(
        text in part.get("text", "")
        for message in transcript["messages"]
        if message.get("info", {}).get("role") == "user"
        for part in message.get("parts", [])
        if part.get("type") == "text"
    )


def show_tool_actions(result: RunResult) -> None:
    """Summarize reported actions in order, without inventing missing steps."""
    actions = [e for e in result.events if e.get("type") == "tool_use"]
    if result.timed_out or result.returncode != 0:
        print(f"Run stopped: {'timed out' if result.timed_out else f'exit {result.returncode}'}.")
    if not actions:
        print("No tool actions were reported. Inspect the run's output for errors.")
    rows = []
    for index, event in enumerate(actions, 1):
        part = event.get("part") or {}
        state = part.get("state") or {}
        args = state.get("input") or {}
        target = args.get("filePath") or args.get("command") or args.get("pattern")
        target = target or json.dumps(args, ensure_ascii=False)
        if len(target) > 180:
            target = target[:180] + "\n[excerpt; expand below]"
        status = state.get("status", "Not reported")
        exit_code = (state.get("metadata") or {}).get("exit")
        if exit_code is not None:
            status += f"; command exit {exit_code}"
        output = state.get("error", state.get("output"))
        if output is None:
            preview = "No returned output reported"
        else:
            preview = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            preview = preview.strip() or "Empty returned output"
            if len(preview) > 180:
                preview = preview[:180] + "\n[excerpt; expand below]"
        rows.append([str(index), part.get("tool", "Not reported"), target, status, preview])
    if rows:
        show_table("Actions reported by OpenCode", ["Order", "Tool", "Requested", "Status", "Returned output"], rows)
    for index, event in enumerate(actions, 1):
        state = (event.get("part") or {}).get("state") or {}
        failed = state.get("status") == "error" or "error" in state
        exit_code = (state.get("metadata") or {}).get("exit")
        show_tool_event(event, title=f"Action {index}: {event_summary(event)}",
                        opened=failed or exit_code not in (None, 0))
    for event in result.events:
        if event.get("type") == "error":
            show_details("OpenCode error", json.dumps(event, indent=2), opened=True)


def show_repair_comparison(initial_average: RunResult, initial_test: RunResult,
                           final_average: RunResult, final_test: RunResult,
                           changed_files: Sequence[str], redact: Redactor) -> None:
    """Compare the student's actual checks, not the agent's claimed outcome."""
    def outcome(result: RunResult) -> str:
        return "Timed out" if result.timed_out else f"Exit {result.returncode}"

    def average_output(result: RunResult) -> str:
        return f"{result.stdout.strip() or 'No output reported'}\n({outcome(result)})"

    show_table("Your checks before and after the repair", ["Check", "Before", "After"], [
        ["Program output", average_output(initial_average), average_output(final_average)],
        ["Test command (0 = passed)", outcome(initial_test), outcome(final_test)],
        ["Files changed since the first snapshot", "Starting snapshot",
         "\n".join(changed_files) or "No changes"],
    ], redact=redact)


def show_session_comparison(same: RunResult, fresh: RunResult, *,
                            continued_history: dict | None, fresh_history: dict | None,
                            instruction: str, redact: Redactor) -> None:
    """Compare actual replies and saved user messages without grading the answers."""
    def reply(result: RunResult) -> str:
        parts = [(event.get("part") or {}).get("text", "") for event in result.events
                 if event.get("type") == "text"]
        return "\n".join(parts).strip() or "No reply reported"

    def saved(history: dict | None) -> str:
        if history is None or not isinstance(history.get("messages"), list):
            return "No saved messages available"
        return "Found" if has_user_message(history, instruction) else "Not found"

    show_table("Compare the two conversations", ["Observation", "Same session", "Fresh session"], [
        ["Session ID", same.session_id or "Not reported", fresh.session_id or "Not reported"],
        ["Model reply", reply(same), reply(fresh)],
        ["Original report-order instruction in saved user messages",
         saved(continued_history), saved(fresh_history)],
    ], redact=redact)
    for label, history in (("Same session", continued_history), ("Fresh session", fresh_history)):
        if history is not None:
            show_details(f"{label}: saved messages", redact(json.dumps(history, indent=2, ensure_ascii=False)))


def show_tool_event(event: dict, *, title: str = "Tool request and result", opened: bool = False) -> None:
    """Display task-relevant output first, retaining the complete event for inspection."""
    part = event.get("part") or {}
    state = part.get("state") or {}
    lines = ["Requested:", json.dumps(state.get("input", {}), indent=2, ensure_ascii=False)]
    for key in ("output", "error"):
        if key in state:
            value = state[key]
            lines += [f"{key.capitalize()}:", value if isinstance(value, str) else json.dumps(value)]
    if "output" not in state and "error" not in state:
        lines.append("No returned output was included in this event.")
    show_details("Full event record", json.dumps(event, indent=2, ensure_ascii=False))
    show_details(title, "\n".join(lines), opened=opened)

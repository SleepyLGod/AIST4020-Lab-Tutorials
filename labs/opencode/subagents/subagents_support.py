"""Prepare Chapter 3 cases and display saved runs; OpenCode performs delegation."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Mapping, Sequence
import uuid
import zipfile


_CHAPTER1 = Path(__file__).resolve().parent.parent / "overall-harness"
_SHARED = _CHAPTER1 / "lab_support.py"
_MODULE = "_aist4020_overall_support"
if _MODULE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE, _SHARED)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load the Chapter 1 helper at {_SHARED}")
    _loaded = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE] = _loaded
    _spec.loader.exec_module(_loaded)
base = sys.modules[_MODULE]
if Path(base.__file__).resolve() != _SHARED.resolve():
    raise ImportError("Restart the kernel to load the helpers from this course checkout.")

RunResult = base.RunResult
Snapshot = dict[str, dict[str, str]]
_TIMEOUT = 300


def tool_parts(transcript: dict) -> list[dict]:
    """Read stored tool parts, not assistant text that resembles a tool call."""
    return [part for message in transcript["messages"] for part in message.get("parts", [])
            if part.get("type") == "tool"]


def recorded_reply(transcript: dict) -> tuple[str, bool]:
    """Return the latest assistant text and whether its turn finished normally."""
    messages = transcript["messages"]
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        info = message.get("info", {})
        if info.get("role") != "assistant":
            continue
        texts = [part.get("text", "") for part in message.get("parts", []) if part.get("type") == "text"]
        completed = bool(texts and texts[-1].strip() and info.get("finish") == "stop"
                         and info.get("time", {}).get("completed") and not info.get("error")
                         and not any(m.get("info", {}).get("role") == "user" for m in messages[index + 1:])
                         and all((p.get("state") or {}).get("status") not in {"pending", "running"}
                                 for p in message.get("parts", []) if p.get("type") == "tool"))
        return (texts[-1] if texts else "No text recorded in the latest assistant message."), completed
    return "No assistant text recorded.", False


def changed_files(before: Snapshot, after: Snapshot) -> list[str]:
    """Retain every changed entry except bytecode already excluded by snapshot()."""
    return [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]


def _action_rows(parts: list[dict], project: Path) -> list[list[str]]:
    rows = []
    for number, part in enumerate(parts, 1):
        state = part.get("state") or {}
        args = state.get("input") or {}
        target = args.get("filePath") or args.get("command") or args.get("subagent_type") or args.get("pattern", "")
        target = str(target).replace(str(project) + "/", "")
        rows.append([str(number), part.get("tool", "Unknown"), target, state.get("status", "Not reported")])
    return rows


def _show_actions(title: str, parts: list[dict], project: Path, redact: base.Redactor,
                  *, complete: bool = True) -> None:
    base.show_table(title, ["#", "Tool", "File, command, or recipient", "Result"],
                    _action_rows(parts, project), redact=redact)
    if not parts:
        print("No tool actions recorded in this conversation." if complete
              else "No tool actions found in the available CLI output.")
    for number, part in enumerate(parts, 1):
        base.show_tool_event({"part": part}, title=f"{title}: action {number}",
                             opened=(part.get("state") or {}).get("status") == "error")


@dataclass
class Case:
    """One fresh project, its real run, and separately exported conversations."""

    name: str
    root: Path
    project: Path
    binary: Path
    env: dict[str, str]
    model: str
    records: Path
    redact: base.Redactor
    before: Snapshot
    result: RunResult | None = None
    after: Snapshot | None = None
    parent: dict | None = None
    children: dict[str, dict] = field(default_factory=dict)
    session_errors: dict[str, str] = field(default_factory=dict)
    _started: bool = False

    def run(self, command: Sequence[str]) -> RunResult:
        """Invoke the main agent once; do not invoke an investigator agent from Python."""
        if self._started:
            raise RuntimeError("Create a new case before rerunning; this project has already been used.")
        if isinstance(command, str) or not command:
            raise ValueError("Provide the complete command as a list of arguments.")
        self._started = True
        print(self.redact(f"Working directory: {self.project}\n$ {shlex.join(command)}"), flush=True)

        def progress(event: dict) -> None:
            part = event.get("part") or {}
            if event.get("type") == "tool_use":
                state = part.get("state") or {}
                print(self.redact(f"Main agent: {part.get('tool', 'tool')} - {state.get('status', 'unknown')}"), flush=True)
                if state.get("error"):
                    print(self.redact(str(state["error"])), flush=True)
            elif event.get("type") == "error":
                print(self.redact(json.dumps(event)), flush=True)

        self.result = base.run_command(command, cwd=self.project, env=self.env, timeout=_TIMEOUT,
            redact=self.redact, show=False, on_event=progress, event_log=self.records / "agent.jsonl")
        self.after = base.snapshot(self.project)
        base.save_json(self.records / "files.json", {"before": self.before, "after": self.after}, self.redact)
        base.save_json(self.records / "task.json", {"command": list(command), "model": self.model}, self.redact)
        if self.result.stderr:
            base.show_details("CLI messages", self.redact(self.result.stderr), opened=self.result.returncode != 0)
        print(f"Exit {self.result.returncode}; timed out: {self.result.timed_out}; {self.result.seconds}s")
        return self.result

    def check(self, command: Sequence[str]) -> RunResult:
        """Run the student's command without passing the provider credential."""
        env = {k: v for k, v in self.env.items() if k not in {"COURSE_API_KEY", "OPENCODE_CONFIG"}}
        result = base.run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)
        base.save_json(self.records / f"check-{uuid.uuid4().hex}.json", {
            "command": list(command), "returncode": result.returncode, "timed_out": result.timed_out,
            "stdout": result.stdout, "stderr": result.stderr,
        }, self.redact)
        return result

    def export(self, session_id: str) -> dict:
        """Export a recorded session; this command does not call the model."""
        command = [str(self.binary), "export", session_id]
        print(self.redact(f"$ {shlex.join(command)}"))
        result = base.run_command(command, cwd=self.project, env=self.env, timeout=60,
                                  redact=self.redact, show=False)
        if result.timed_out or result.returncode != 0:
            base.show_details("Session export error", self.redact(result.stdout + result.stderr), opened=True)
        result.require_success()
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError("Session export has no messages list.")
        if data.get("info", {}).get("id") != session_id:
            raise ValueError("Exported session ID does not match the requested session.")
        base.save_json(self.records / f"session-{uuid.uuid4().hex}.json", data, self.redact)
        return data

    def load_sessions(self) -> bool:
        """Load available conversations, reporting each missing export explicitly."""
        if self.result is None:
            raise RuntimeError("Run the case before loading its sessions.")
        self.parent = None
        self.children = {}
        self.session_errors = {}

        def unavailable(label: str, error: str) -> None:
            self.session_errors[label] = self.redact(error)
            print(self.redact(f"{label}: {error}"))

        try:
            sid = self.result.session_id
            if not sid:
                raise ValueError("No main session ID was reported.")
            self.parent = self.export(sid)
        except (RuntimeError, ValueError, OSError) as error:
            unavailable("Main conversation unavailable", str(error))

        attempted = set()
        for part in self.main_actions():
            if part.get("tool") != "task":
                continue
            sid = ((part.get("state") or {}).get("metadata") or {}).get("sessionId")
            if not isinstance(sid, str) or not sid:
                print("A task was recorded without a child session ID; inspect its result below.")
                continue
            if sid not in attempted:
                attempted.add(sid)
                try:
                    self.children[sid] = self.export(sid)
                except (RuntimeError, ValueError, OSError) as error:
                    unavailable(f"Child conversation {sid} unavailable", str(error))
        return self.parent is not None and not self.session_errors

    def main_actions(self) -> list[dict]:
        """Use the transcript when available, otherwise retain actual CLI tool events."""
        if self.parent is not None:
            return tool_parts(self.parent)
        if self.result is None:
            return []
        return [event["part"] for event in self.result.events
                if event.get("type") == "tool_use" and isinstance(event.get("part"), dict)]

    def inspect_agent(self, name: str) -> dict:
        """Display OpenCode's resolved tool availability without running a model."""
        command = [str(self.binary), "debug", "agent", name]
        print(self.redact(f"$ {shlex.join(command)}"))
        result = base.run_command(command, cwd=self.project, env=self.env, timeout=60,
                                  redact=self.redact, show=False)
        if result.timed_out or result.returncode != 0:
            base.show_details("Agent configuration error", self.redact(result.stdout + result.stderr), opened=True)
        result.require_success()
        data = json.loads(result.stdout)
        if not isinstance(data.get("tools"), dict):
            raise ValueError("Agent inspection did not report tool availability.")
        base.save_json(self.records / f"agent-{uuid.uuid4().hex}.json", data, self.redact)
        base.show_table(f"{name}: tools available", ["Tool", "Available"],
            [[tool, "Yes" if enabled else "No"] for tool, enabled in data["tools"].items()], redact=self.redact)
        return data


@dataclass
class Experiment:
    """Reuse the course provider setup, with fresh state for each comparison."""

    root: Path
    env: dict[str, str]
    binary: Path
    project: Path
    records: Path
    version: str
    selected: dict[str, str] = field(default_factory=dict)
    backend: str = ""
    model: str = ""
    redact: base.Redactor = field(default_factory=base.Redactor)
    cases: list[Case] = field(default_factory=list)

    def configure(self, backend: str, backends: Mapping[str, Mapping[str, str]], *, enabled: bool) -> None:
        """Configure the chosen provider; do not make a model request."""
        if self.cases:
            raise RuntimeError("Start a new experiment before changing the comparison model.")
        self.selected = {}
        selected, redact = base.configure_backend(backend, backends, root=self.root, env=self.env,
            context_tokens=16384 if backend == "ollama" else 32768, enabled=enabled)
        self.selected, self.redact, self.backend = selected, redact, backend
        self.model = f"course/{selected['model']}"

    def new_case(self, name: str, settings: dict) -> Case:
        """Save the visible role definitions and copy the untouched four-file example."""
        if not self.selected:
            raise RuntimeError("Choose and configure a provider before creating a case.")
        root = Path(tempfile.mkdtemp(prefix="case-", dir=self.root))
        project = base.fresh_project(_CHAPTER1 / "grade-report", root)
        env = base.isolated_environment(root)
        env["COURSE_API_KEY"] = self.env["COURSE_API_KEY"]
        model, _ = base.write_configuration(root, env, self.selected, provider=self.backend,
            settings=settings, context_tokens=16384 if self.backend == "ollama" else 32768,
            output_tokens=2048, redact=self.redact)
        config_path = Path(env["OPENCODE_CONFIG"])
        records = root / "records"
        records.mkdir()
        before = base.snapshot(project)
        base.save_json(records / "setup.json", {"name": name, "version": self.version,
            "model": model, "agents": settings["agent"]}, self.redact)
        case = Case(name, root, project, self.binary, env, model, records, self.redact, before)
        self.cases.append(case)
        print(f"{name}: {project}\nConfiguration: {config_path}")
        return case

    def check(self, command: Sequence[str]) -> RunResult:
        """Check the unmodified example without a model credential."""
        env = {k: v for k, v in self.env.items() if k not in {"COURSE_API_KEY", "OPENCODE_CONFIG"}}
        return base.run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)

    def download_records(self) -> None:
        """Package only redacted records, excluding credential/config directories."""
        if not any(c.result for c in self.cases):
            raise RuntimeError("Run a task before downloading records.")
        path = self.root / f"subagents-records-{uuid.uuid4().hex}.zip"
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as bundle:
            for case in self.cases:
                for record in sorted(case.records.iterdir()):
                    if record.is_file() and not record.is_symlink() and record.suffix in {".json", ".jsonl"}:
                        bundle.writestr(f"{case.root.name}/{record.name}", self.redact(record.read_text()))
        try:
            from google.colab import files
        except ImportError:
            print(f"Records: {path}")
        else:
            files.download(str(path))


def open_lab(version: str) -> Experiment:
    """Install the pinned CLI and copy Chapter 1's example into a fresh directory."""
    return Experiment(*base.prepare_lab(_CHAPTER1, version), version=version)


def show_project(project: Path) -> None:
    """Show the example directory and its actual four files."""
    base.show_project(project)
    for name in ("README.md", "grades.py", "scores.csv", "test_grades.py"):
        base.show_details(name, (project / name).read_text(), opened=True)


def show_case(case: Case, test: RunResult) -> None:
    """Show observed actions, final answer, independent check and all file changes."""
    if case.after is None or case.result is None:
        raise RuntimeError("Run the case first.")
    if case.parent is None:
        print("Main conversation unavailable. Showing recorded CLI actions; the list may be incomplete.")
        texts = [(e.get("part") or {}).get("text", "") for e in case.result.events if e.get("type") == "text"]
        reply, completed = (texts[-1] if texts else "No assistant text recorded."), False
    else:
        reply, completed = recorded_reply(case.parent)
    _show_actions(f"{case.name}: main agent", case.main_actions(), case.project, case.redact,
                  complete=case.parent is not None)
    completed = (completed and not case.result.timed_out and case.result.returncode == 0
                 and not any(e.get("type") == "error" for e in case.result.events))
    title = "Main agent's final answer" if completed else "Main agent's last recorded text (completion not confirmed)"
    base.show_details(title, case.redact(reply), opened=True)
    if case.parent is None and case.result.stdout:
        base.show_details("Recorded CLI output", case.redact(case.result.stdout), opened=False)
    if case.result.stderr:
        base.show_details("CLI messages", case.redact(case.result.stderr), opened=True)
    for event in case.result.events:
        if event.get("type") == "error":
            base.show_details("Run error", case.redact(json.dumps(event)), opened=True)
    base.show_table("Your independent check", ["Check", "Result"], [
        ["Test", base.test_outcome(test)],
        ["Changed files", ", ".join(changed_files(case.before, case.after)) or "None"],
    ], redact=case.redact)
    base.show_changes(case.before, case.after, case.redact)


def show_handoffs(case: Case) -> None:
    """Display each actual task, child conversation and returned text separately."""
    if case.result is None:
        raise RuntimeError("Run the case first.")
    if case.parent is None:
        print("Main conversation unavailable. Showing task calls found in CLI output; others may be missing.")
    tasks = [p for p in case.main_actions() if p.get("tool") == "task"]
    if not tasks:
        print("No task call recorded. This run does not demonstrate delegation." if case.parent is not None
              else "Cannot determine whether delegation occurred from the available records.")
    for number, part in enumerate(tasks, 1):
        state = part.get("state") or {}
        sid = (state.get("metadata") or {}).get("sessionId")
        args = state.get("input") or {}
        base.show_details(f"Handoff {number}: main agent's request to {args.get('subagent_type', 'unknown')}",
                          case.redact(args.get("prompt", "No prompt recorded.")), opened=True)
        base.show_tool_event({"part": part}, title="What the main agent received", opened=True)
        if state.get("status") != "completed":
            print("This task did not finish successfully. Check its status and error above.")
        child = case.children.get(sid)
        if child is None:
            print("No child export available for this task. Read its saved result above.")
            continue
        parent_id = case.parent.get("info", {}).get("id") if case.parent else None
        linked = bool(parent_id) and child.get("info", {}).get("parentID") == parent_id
        base.show_table("Two conversations", ["Record", "Observed"], [
            ["Main session", parent_id or "Not reported"], ["Child session", sid],
            ["Child parentID matches main session", "Unknown: main conversation unavailable" if not parent_id
             else "Yes" if linked else "No: inspect this mismatch"],
        ], redact=case.redact)
        _show_actions("Investigator agent actions", tool_parts(child), case.project, case.redact)
        reply, completed = recorded_reply(child)
        completed = completed and state.get("status") == "completed"
        title = "Investigator agent's final reply" if completed else "Investigator agent's last recorded text (completion not confirmed)"
        base.show_details(title, case.redact(reply), opened=True)
        if completed:
            returned = bool(reply.strip()) and reply.strip() in state.get("output", "")
            print(f"Final reply found in the task result: {'Yes' if returned else 'No: compare the two texts above'}")
        else:
            print("No completed reply confirmed. The text above may only describe work in progress.")


def compare_cases(single: Case, delegated: Case, *, single_test: RunResult, delegated_test: RunResult) -> None:
    """Compare actual outcomes, including all exported child actions in the count."""
    columns = []
    for case, test in [(single, single_test), (delegated, delegated_test)]:
        if case.after is None or case.result is None:
            raise RuntimeError("Run both cases before comparing.")
        actions = case.main_actions()
        tasks = [p for p in actions if p.get("tool") == "task"]
        missing = case.parent is None or any(((p.get("state") or {}).get("metadata") or {}).get("sessionId") not in case.children for p in tasks)
        child_count = "Incomplete: inspect task results" if missing else str(sum(len(tool_parts(c)) for c in case.children.values()))
        columns.append([
            base.test_outcome(test),
            ", ".join(changed_files(case.before, case.after)) or "None",
            str(len(tasks)) if case.parent is not None else f"Unknown ({len(tasks)} recorded in CLI)",
            str(len(case.children)),
            str(len(actions)) if case.parent is not None else f"Unknown ({len(actions)} recorded in CLI)", child_count,
            f"{case.result.seconds}s" + (" (timed out)" if case.result.timed_out else ""),
        ])
    labels = ["Your test", "Changed files", "Task calls", "Exported child sessions",
              "Main agent tool actions", "Child tool actions", "Task time"]
    base.show_table("Same repair, different division of work", ["Observation", single.name, delegated.name],
        [[label, columns[0][i], columns[1][i]] for i, label in enumerate(labels)], redact=single.redact)

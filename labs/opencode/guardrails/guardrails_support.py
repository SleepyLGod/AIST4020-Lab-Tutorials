"""Prepare Chapter 2 cases and display real records; OpenCode runs the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from typing import Mapping, Sequence
import uuid
import zipfile


_SHARED = Path(__file__).resolve().parent.parent / "overall-harness" / "lab_support.py"
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

show_details = base.show_details
RunResult = base.RunResult
Snapshot = dict[str, dict[str, str]]
_GENERATED_IGNORE = "node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore"
_DETAIL_FIELDS = {"Session", "OpenCode setup file"}
_RUN_TIMEOUT_SECONDS = 300


def _target(part: dict, project: Path, filename: str) -> bool:
    value = (part.get("state") or {}).get("input", {}).get("filePath")
    if not isinstance(value, str):
        return False
    path = Path(value)
    if not path.is_absolute():
        path = project / path
    return path == project / filename


def _denied(state: dict) -> bool:
    return state.get("status") == "error" and "specified a rule which prevents" in str(state.get("error", ""))


def tool_parts(result: RunResult) -> list[dict]:
    """Return recorded tool actions, without filling in missing events."""
    return [event["part"] for event in result.events
            if event.get("type") == "tool_use" and isinstance(event.get("part"), dict)]


def _is_dependency_path(name: str) -> bool:
    return (name == ".opencode/node_modules" or name.startswith(".opencode/node_modules/")
            or name in {".opencode/package.json", ".opencode/package-lock.json",
                        ".opencode/bun.lock", ".opencode/bun.lockb"})


def file_changes(before: Snapshot, after: Snapshot, result: RunResult, project: Path) -> tuple[list[str], list[str]]:
    """Separate the verified generated ignore file, retaining every changed path."""
    changed = [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
    setup = []
    name = ".opencode/.gitignore"
    requested = any(_target(part, project, name) for part in tool_parts(result))
    if name not in before and after.get(name, {}).get("text") == _GENERATED_IGNORE and not requested:
        setup.append(name)
    return [name for name in changed if name not in setup], setup


@dataclass
class Case:
    """One fresh project and one agent request, with separate independent checks."""

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
    _started: bool = False

    def run(self, command: Sequence[str]) -> RunResult:
        """Run a fresh session once; do not retry or complete missing agent actions."""
        if self._started:
            raise RuntimeError("Create a new case before rerunning this task; this project has already been used.")
        if isinstance(command, str) or not command:
            raise ValueError("Provide the complete command as a list of arguments.")
        self._started = True
        print(self.redact(f"Working directory: {self.project}\n$ {shlex.join(command)}"), flush=True)
        print(f"The helper stops after {_RUN_TIMEOUT_SECONDS // 60} minutes.", flush=True)

        def progress(event: dict) -> None:
            if event.get("type") == "tool_use":
                part = event.get("part") or {}
                state = part.get("state") or {}
                args = state.get("input") or {}
                target = args.get("filePath") or args.get("command") or args.get("name") or args.get("pattern", "")
                print(self.redact(f"{part.get('tool', 'tool')}: {state.get('status', 'unknown')} {target}"), flush=True)
                if state.get("error"):
                    print(self.redact(str(state["error"])), flush=True)
            elif event.get("type") == "text":
                print(self.redact((event.get("part") or {}).get("text", "")), flush=True)
            elif event.get("type") == "error":
                print(self.redact(json.dumps(event)), flush=True)

        self.result = base.run_command(command, cwd=self.project, env=self.env, timeout=_RUN_TIMEOUT_SECONDS,
                                       redact=self.redact, show=False, on_event=progress,
                                       event_log=self.records / "agent.jsonl")
        self.after = base.snapshot(self.project)
        base.save_json(self.records / "files.json", {"before": self.before, "after": self.after}, self.redact)
        base.save_json(self.records / "task.json", {"command": list(command), "session_id": self.result.session_id}, self.redact)
        if self.result.stderr:
            show_details("CLI messages", self.redact(self.result.stderr), opened=self.result.returncode != 0)
        print(f"Exit {self.result.returncode}; timed out: {self.result.timed_out}; {self.result.seconds}s")
        return self.result

    def check(self, command: Sequence[str]) -> RunResult:
        """Run the student's visible command without the model credential."""
        env = {k: v for k, v in self.env.items() if k not in {"COURSE_API_KEY", "OPENCODE_CONFIG"}}
        result = base.run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)
        base.save_json(self.records / f"check-{uuid.uuid4().hex}.json", {
            "command": list(command), "returncode": result.returncode, "timed_out": result.timed_out,
            "stdout": result.stdout, "stderr": result.stderr,
        }, self.redact)
        return result


@dataclass
class Experiment:
    """Reuse the Chapter 1 installer and provider setup for independent cases."""

    lab_dir: Path
    root: Path
    binary: Path
    project: Path
    env: dict[str, str]
    version: str
    selected: dict[str, str] = field(default_factory=dict)
    backend: str = ""
    redact: base.Redactor = field(default_factory=base.Redactor)
    cases: list[Case] = field(default_factory=list)

    def configure(self, backend: str, backends: Mapping[str, Mapping[str, str]], *, enabled: bool) -> None:
        """Read only the selected credential; this does not make a model call."""
        if self.cases:
            raise RuntimeError("Start a new experiment before changing the model used for a comparison.")
        self.selected = {}
        selected, redact = base.configure_backend(
            backend, backends, root=self.root, env=self.env,
            context_tokens=16384 if backend == "ollama" else 32768, enabled=enabled,
        )
        self.selected, self.redact, self.backend = selected, redact, backend

    def new_case(self, name: str, settings: dict, *, with_skill: bool = False) -> Case:
        """Copy unchanged starting files and apply exactly the supplied permissions."""
        if not self.selected:
            raise RuntimeError("Choose and configure a model before creating a case.")
        root = Path(tempfile.mkdtemp(prefix="case-", dir=self.root))
        project = base.fresh_project(self.lab_dir / "grade-report", root)
        if with_skill:
            target = project / ".opencode" / "skills" / "repair-check" / "SKILL.md"
            target.parent.mkdir(parents=True)
            shutil.copy2(self.lab_dir / "skills" / "repair-check" / "SKILL.md", target)
        env = base.isolated_environment(root)
        env["COURSE_API_KEY"] = self.env["COURSE_API_KEY"]
        model, _ = base.write_configuration(
            root, env, self.selected, provider=self.backend, settings=settings,
            context_tokens=16384 if self.backend == "ollama" else 32768,
            output_tokens=2048, redact=self.redact,
        )
        records = root / "records"
        records.mkdir()
        before = base.snapshot(project)
        base.save_json(records / "setup.json", {"name": name, "version": self.version,
            "model": model, "permissions": settings["permission"], "with_skill": with_skill}, self.redact)
        case = Case(name, root, project, self.binary, env, model, records, self.redact, before)
        self.cases.append(case)
        print(f"{name}: fresh project at {project}; the next request starts a new session.")
        return case

    def check(self, command: Sequence[str]) -> RunResult:
        """Inspect the starting example before asking an agent to change anything."""
        env = {k: v for k, v in self.env.items() if k not in {"COURSE_API_KEY", "OPENCODE_CONFIG"}}
        return base.run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)

    def download_records(self) -> None:
        """Download only redacted task evidence, excluding credentials and configuration files."""
        if not any(case.result for case in self.cases):
            raise RuntimeError("Run a task before downloading records.")
        path = self.root / f"guardrails-records-{uuid.uuid4().hex}.zip"
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


def open_lab(lab_dir: Path, version: str) -> Experiment:
    """Install the pinned CLI and prepare a disposable preview of the project."""
    root, env, binary, project, _ = base.prepare_lab(lab_dir, version)
    return Experiment(lab_dir, root, binary, project, env, version)


def show_project(project: Path) -> None:
    """Show actual files and their roles, then the full text students will use."""
    roles = {"README.md": "requirements", "AGENTS.md": "project instructions",
             "grades.py": "implementation", "scores.csv": "input data",
             "test_grades.py": "test", "review.md": "report to write"}
    base.show_table("Project files", ["File", "Purpose"], [
        [p.name, roles.get(p.name, "Additional entry")] for p in sorted(project.iterdir())])
    for name in roles:
        show_details(name, (project / name).read_text(), opened=True)


def observations(case: Case, test: RunResult, average: RunResult) -> dict[str, str]:
    """Derive comparison cells from actual requests, file snapshots, and checks."""
    if case.result is None or case.after is None:
        raise RuntimeError("Run the case before comparing its results.")
    edits = [p for p in tool_parts(case.result)
             if p.get("tool") in {"edit", "write"} and _target(p, case.project, "grades.py")]
    statuses = ["Permission denied" if _denied(p.get("state", {}))
                else p.get("state", {}).get("status", "Not reported") for p in edits]
    changed, setup = file_changes(case.before, case.after, case.result, case.project)
    dependencies = [name for name in changed if _is_dependency_path(name)]
    report_changed = case.before.get("review.md") != case.after.get("review.md")
    return {
        "Session": case.result.session_id or "Not reported",
        "Edit/write requests for grades.py": "; ".join(statuses) if statuses else "No edit/write request recorded",
        "grades.py changed": "Yes" if case.before.get("grades.py") != case.after.get("grades.py") else "No",
        "Your test": base.test_outcome(test),
        "Your average check": (average.stdout.strip() or "No output") +
            (" (timed out)" if average.timed_out else f" (exit {average.returncode})"),
        "review.md changed": "Yes" if report_changed else "No",
        "Other file changes": ", ".join(p for p in changed
                                         if p not in {"grades.py", "review.md"} and not _is_dependency_path(p)) or "None",
        "Dependency changes": f"{len(dependencies)} paths (details folded below each run)" if dependencies else "None",
        "OpenCode setup file": ", ".join(setup) or "No matching generated file",
    }


def _show_run_details(case: Case, values: dict[str, str]) -> None:
    text = f"Session: {values['Session']}"
    if values["OpenCode setup file"] != "No matching generated file":
        text += ("\nThe new .opencode/.gitignore matches OpenCode's generated setup file. "
                 "It is listed separately from the repair; its diff is retained.")
    show_details(f"{case.name}: run details", case.redact(text))


def show_case(case: Case, test: RunResult, average: RunResult) -> None:
    """Show edits and denials distinctly, retaining every request and returned result."""
    values = observations(case, test, average)
    base.show_table(case.name, ["Check", "Observed"],
                    [[k, v] for k, v in values.items() if k not in _DETAIL_FIELDS], redact=case.redact)
    _show_run_details(case, values)
    assert case.result is not None and case.after is not None
    if case.result.timed_out or case.result.returncode != 0:
        print("The agent run stopped early. Inspect its output before drawing a conclusion.")
    for i, event in enumerate((e for e in case.result.events if e.get("type") == "tool_use"), 1):
        part = event.get("part") or {}
        state = part.get("state") or {}
        target_edit = part.get("tool") in {"edit", "write"} and _target(part, case.project, "grades.py")
        report_written = (part.get("tool") in {"edit", "write"}
                          and _target(part, case.project, "review.md") and state.get("status") == "completed")
        title = f"Action {i}: {part.get('tool', 'tool')} - {state.get('status', 'Not reported')}"
        if _denied(state):
            title += " (grades.py edit denied)" if target_edit else " (another action denied)"
        base.show_tool_event(event, title=title, opened=target_edit or report_written or state.get("status") == "error")
    for event in case.result.events:
        if event.get("type") == "error":
            show_details("OpenCode error", case.redact(json.dumps(event)), opened=True)
    base.show_changes(
        {name: value for name, value in case.before.items() if not _is_dependency_path(name)},
        {name: value for name, value in case.after.items() if not _is_dependency_path(name)},
        case.redact,
    )
    dependencies = [name for name in sorted(case.before.keys() | case.after.keys())
                    if _is_dependency_path(name) and case.before.get(name) != case.after.get(name)]
    if dependencies:
        show_details(
            f"Dependency changes: {len(dependencies)} paths",
            case.redact("Full before-and-after records are kept in files.json in the download.\n\n"
                        + "\n".join(dependencies)),
            opened=False,
        )
    show_details("review.md after this run", case.redact(case.after.get("review.md", {}).get("text", "File missing")), opened=True)
    base.save_json(case.records / "observations.json", values, case.redact)


def compare_cases(allowed_case: Case, denied_case: Case, *, allowed_test: RunResult, denied_test: RunResult,
                  allowed_average: RunResult, denied_average: RunResult) -> None:
    """Compare observations without substituting the expected outcome."""
    left = observations(allowed_case, allowed_test, allowed_average)
    right = observations(denied_case, denied_test, denied_average)
    base.show_table("Same task, different edit permission", ["Observation", allowed_case.name, denied_case.name],
                    [[key, left[key], right[key]] for key in left if key not in _DETAIL_FIELDS], redact=allowed_case.redact)
    _show_run_details(allowed_case, left)
    _show_run_details(denied_case, right)


def show_skill(case: Case) -> None:
    """Show actual skill-tool results, not the fixture text as proof of loading."""
    if case.result is None:
        raise RuntimeError("Run the Skill case first.")
    events = [e for e in case.result.events if e.get("type") == "tool_use"
              and (e.get("part") or {}).get("tool") == "skill"]
    if not events:
        print("No skill-tool call was reported in this run.")
    for event in events:
        base.show_tool_event(event, title="Skill request and returned content", opened=True)

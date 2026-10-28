"""Restore real deliveries and display checks; OpenCode performs any remediation."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Mapping, Sequence
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'overall-harness/lab_support.py'
MODULE = '_aist4020_overall_support'
if MODULE not in sys.modules:
    spec = importlib.util.spec_from_file_location(MODULE, SHARED)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SHARED}')
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = loaded
    spec.loader.exec_module(loaded)
base = sys.modules[MODULE]
if Path(base.__file__).resolve() != SHARED.resolve():
    raise ImportError('Restart the kernel to use this course checkout.')

Snapshot = dict[str, dict[str, str]]


def report_sections(text: str) -> dict[str, str]:
    """Read section bodies, including nested content but not empty subheadings."""
    sections: dict[str, list[str]] = {}
    parents: list[tuple[int, str]] = []
    fence = ''
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if not fence:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = ''
            for _, name in parents:
                sections[name].append(line)
            continue
        heading = None if fence else re.match(r'^\s*(#{1,6})\s+(.+?)\s*#*\s*$', line)
        if heading:
            depth, name = len(heading.group(1)), heading.group(2).strip().casefold()
            while parents and parents[-1][0] >= depth:
                parents.pop()
            parents.append((depth, name))
            sections.setdefault(name, [])
        else:
            for _, name in parents:
                sections[name].append(line)
    return {name: '\n'.join(body).strip() for name, body in sections.items()}


def changed_files(before: Snapshot, after: Snapshot) -> list[str]:
    """Include added, removed, or changed entries in the project comparison."""
    return [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]


def tools(transcript: dict) -> list[dict]:
    """Read recorded tools rather than tool-like text in a model answer."""
    return [part for message in transcript.get('messages', []) for part in message.get('parts', [])
            if part.get('type') == 'tool']


def usage(transcript: dict | None) -> dict:
    """Sum unique assistant-message usage; missing records remain unknown."""
    if transcript is None:
        return {'responses': None, 'tokens': None, 'cache_read': None}
    messages = {m['info']['id']: m['info'] for m in transcript['messages']
                if m.get('info', {}).get('role') == 'assistant'}
    recorded = list(messages.values())
    available = bool(recorded) and all('tokens' in m and 'total' in m['tokens'] for m in recorded)
    return {'responses': len(recorded),
            'tokens': sum(m['tokens']['total'] for m in recorded) if available else None,
            'cache_read': sum(m['tokens'].get('cache', {}).get('read', 0) for m in recorded) if available else None}


@dataclass
class Delivery:
    """A fresh working copy plus the run evidence associated with that delivery."""

    label: str
    project: Path
    before: Snapshot
    env: dict[str, str]
    records: Path
    record: dict
    transcript: dict | None
    redact: base.Redactor = field(default_factory=base.Redactor)
    result: base.RunResult | None = None
    export_error: str = ''
    is_live: bool = False
    comparison_start: str = 'Before this task'

    def read(self, name: str) -> str:
        """Read a regular project file; do not follow an agent-created symlink."""
        entry = base.snapshot(self.project).get(name, {})
        return entry.get('text', '') if entry.get('kind') == 'file' else ''

    def check(self, command: Sequence[str]) -> base.RunResult:
        """Execute the student's independent check without provider credentials."""
        env = {k: v for k, v in self.env.items() if k not in {'COURSE_API_KEY', 'OPENCODE_CONFIG'}}
        result = base.run_command(command, cwd=self.project, env=env, timeout=20, redact=self.redact)
        base.save_json(self.records / f'check-{uuid.uuid4().hex}.json', {
            'command': list(command), 'returncode': result.returncode, 'timed_out': result.timed_out,
            'stdout': result.stdout, 'stderr': result.stderr,
        }, self.redact)
        return result


def _restore(snapshot: Snapshot, folder: Path) -> None:
    for name, entry in snapshot.items():
        if name in ('.', '..') or Path(name).name != name or entry.get('kind') != 'file':
            raise ValueError('The teaching record must contain top-level regular files only.')
        data = entry['text'].encode('utf-8')
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Recorded bytes do not match the hash for {name}.')
        (folder / name).write_bytes(data)


@dataclass
class Lab:
    """Keep historical analysis offline and permit one explicit live remediation."""

    root: Path
    version: str
    attempted: bool = False
    cases: list[Delivery] = field(default_factory=list)
    redact: base.Redactor = field(default_factory=base.Redactor)

    def historical(self, name: str) -> Delivery:
        """Restore saved after-files without installing OpenCode or reading a key."""
        if name not in ('incomplete', 'complete', 'remediated'):
            raise ValueError('Choose a supplied historical record.')
        record = json.loads((HERE / 'records' / f'{name}.json').read_text())
        if record['version'] != self.version:
            raise ValueError('Historical run and lab versions differ.')
        root = Path(tempfile.mkdtemp(prefix=f'{name}-', dir=self.root))
        project, records = root / 'project', root / 'records'
        project.mkdir()
        records.mkdir()
        _restore(record['files']['after'], project)
        delivery = Delivery(record['label'], project, record['files']['before'],
            base.isolated_environment(root), records, record, record['transcript'],
            comparison_start='Before completing the report (program already repaired)'
                if name == 'remediated' else 'Before the original code repair')
        self.cases.append(delivery)
        print(f'{delivery.label} | {record["model"]}\nRestored files: {project}')
        return delivery

    def prepare_live(self, failed: Delivery, backend: str, backends: Mapping[str, Mapping[str, str]], *,
                     settings: dict, enabled: bool) -> tuple[Delivery, Path, str]:
        """Install/configure the chosen provider, restoring the untouched historical after-state."""
        if not enabled:
            raise RuntimeError('Live calls are disabled.')
        if self.attempted:
            raise RuntimeError('The one live follow-up has already been attempted.')
        root = Path(tempfile.mkdtemp(prefix='live-', dir=self.root))
        env = base.isolated_environment(root)
        selected, redact = base.configure_backend(backend, backends, root=root, env=env,
            context_tokens=16384 if backend == 'ollama' else 32768, enabled=enabled)
        self.redact = redact
        binary = base.install_opencode(root, self.version, env)
        project, records = root / 'project', root / 'records'
        project.mkdir()
        records.mkdir()
        starting = failed.record['files']['after']
        _restore(starting, project)
        model, _ = base.write_configuration(root, env, selected, provider=backend,
            settings=settings, context_tokens=16384 if backend == 'ollama' else 32768,
            output_tokens=2048, redact=redact)
        path = Path(env['OPENCODE_CONFIG'])
        record = {'model': model, 'seconds': None, 'version': self.version,
                  'original_model': failed.record['model']}
        delivery = Delivery('Your live follow-up', project, starting, env, records, record, None,
                            redact=redact, is_live=True,
                            comparison_start='Before completing the report (program already repaired)')
        self.cases.append(delivery)
        print(f'New session | {model}\nHistorical files restored from {failed.record["model"]}.')
        print(f'Permissions saved in {path}; OpenCode reads OPENCODE_CONFIG.')
        return delivery, binary, model

    def run_once(self, delivery: Delivery, command: Sequence[str], *, timeout: int) -> base.RunResult:
        """Run the visible command once, keep errors, and never retry or repair files here."""
        if self.attempted:
            raise RuntimeError('One follow-up was already attempted; inspect its result.')
        self.attempted = True
        print(delivery.redact(f'Working directory: {delivery.project}\n$ {shlex.join(command)}'), flush=True)

        def progress(event: dict) -> None:
            if event.get('type') == 'tool_use':
                part = event.get('part') or {}
                state = part.get('state') or {}
                print(delivery.redact(f'{part.get("tool")}: {state.get("status", "unknown")}'), flush=True)
                if state.get('error'):
                    print(delivery.redact(state['error']), flush=True)
            elif event.get('type') == 'error':
                print(delivery.redact(json.dumps(event)), flush=True)

        try:
            result = base.run_command(command, cwd=delivery.project, env=delivery.env, timeout=timeout,
                redact=delivery.redact, show=False, on_event=progress,
                event_log=delivery.records / 'agent.jsonl')
        except OSError as exc:
            result = base.RunResult(127, False, 0, '', f'Could not start OpenCode: {exc}')
        delivery.result = result
        delivery.record['seconds'] = result.seconds
        base.save_json(delivery.records / 'run.json', {
            'command': list(command), 'model': delivery.record['model'], 'seconds': result.seconds,
            'returncode': result.returncode, 'timed_out': result.timed_out,
            'stdout': result.stdout, 'stderr': result.stderr, 'events': result.events,
            'before': delivery.before, 'after': base.snapshot(delivery.project),
        }, delivery.redact)
        if result.stderr:
            base.show_details('CLI messages', delivery.redact(result.stderr), opened=True)
        print(f'Exit {result.returncode}; timed out: {result.timed_out}; {result.seconds}s')
        return result

    def collect(self, delivery: Delivery, binary: Path) -> None:
        """Export the actual session when available; missing evidence stays missing."""
        if delivery.result is None:
            raise RuntimeError('Run the live task before collecting its conversation.')
        sid = delivery.result.session_id
        if sid is None:
            delivery.export_error = 'No session ID reported. No saved conversation is available.'
        else:
            exported = base.run_command([str(binary), 'export', sid], cwd=delivery.project,
                env=delivery.env, timeout=60, redact=delivery.redact, show=False)
            if exported.timed_out or exported.returncode != 0:
                delivery.export_error = 'Session export failed: ' + exported.stdout + exported.stderr
            else:
                try:
                    data = json.loads(exported.stdout)
                    if data.get('info', {}).get('id') != sid or not isinstance(data.get('messages'), list):
                        raise ValueError('Exported session does not match this run.')
                    delivery.transcript = data
                    base.save_json(delivery.records / 'session.json', data, delivery.redact)
                except (ValueError, AttributeError) as exc:
                    delivery.export_error = f'Cannot read session export: {exc}'
        if delivery.export_error:
            print(delivery.redact(delivery.export_error))

    def download_records(self) -> None:
        """Download redacted checks and runs, not provider configuration or keys."""
        destination = self.root / f'evaluation-records-{uuid.uuid4().hex}.zip'
        with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
            for i, case in enumerate(self.cases):
                for path in sorted(case.records.iterdir()):
                    if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.jsonl'):
                        archive.writestr(f'{i}/{path.name}', case.redact(path.read_text()))
        try:
            from google.colab import files
        except ImportError:
            print(f'Records: {destination}')
        else:
            files.download(str(destination))


def open_lab(version: str = '1.18.34') -> Lab:
    """Start local analysis only; no installation, provider lookup, or model calls."""
    return Lab(Path(tempfile.mkdtemp(prefix='opencode-evaluation-')), version)


def show_project(delivery: Delivery) -> None:
    """Show the restored folder and files used to judge the delivery."""
    base.show_project(delivery.project)
    for name in ('README.md', 'AGENTS.md', 'grades.py', 'scores.csv', 'test_grades.py'):
        base.show_details(name, delivery.read(name), opened=name in ('README.md', 'AGENTS.md'))
    base.show_details('Actual review.md', delivery.read('review.md'), opened=True)


def show_answer(delivery: Delivery) -> None:
    """Show the last recorded reply without claiming that the run finished."""
    texts = [p['text'] for m in (delivery.transcript or {}).get('messages', [])
             if m.get('info', {}).get('role') == 'assistant'
             for p in m.get('parts', []) if p.get('type') == 'text']
    base.show_details(f'{delivery.label}: Last recorded reply', delivery.redact(texts[-1] if texts else
                      'No saved reply available.'), opened=True)


def show_checks(label: str, checks: dict) -> None:
    """Show separate checks rather than combining them into a truth score."""
    base.show_table(label, ['Check', 'Observed'], [
        ['Program test', checks['test_status']],
        ['Missing or empty report sections', ', '.join(checks['report_gaps']) or 'None'],
        ['File comparison starts at', checks['comparison_start']],
        ['Changed files', ', '.join(checks['changed_files']) or 'None'],
        ['Changes outside the requested work', ', '.join(checks['unexpected_changes']) or 'None'],
        ['Report claims', 'Compare with the evidence below; not automatically judged.'],
    ])


def show_evidence(delivery: Delivery) -> None:
    """Put report text beside actual test records and file changes for human review."""
    actions = tools(delivery.transcript or {})
    def request(part: dict) -> str:
        args = part['state'].get('input', {})
        value = args.get('filePath') or args.get('command') or args.get('pattern') or str(args)
        return delivery.redact(str(value).replace(str(delivery.project), '<project>'))

    base.show_table(delivery.label + ': recorded actions', ['#', 'Tool', 'Request', 'Status'], [
        [str(i), p['tool'], request(p),
         p['state'].get('status', 'Unknown')] for i, p in enumerate(actions, 1)], redact=delivery.redact)
    if delivery.transcript is None:
        print('Conversation unavailable. File changes can still be checked; agent actions are unknown.')
    tests = [p for p in actions if p['tool'] == 'bash']
    test_evidence = '\n\n'.join(
        f"{p['state'].get('input', {}).get('command', '')}\nStatus: {p['state'].get('status')}\n"
        f"Exit: {p['state'].get('metadata', {}).get('exit', 'not reported')}\n"
        + p['state'].get('output', p['state'].get('error', 'No output recorded.')) for p in tests
    ) or 'No command results available.'
    report = report_sections(delivery.read('review.md'))
    changed = changed_files(delivery.before, base.snapshot(delivery.project))
    errors = [f'Action {i}: {p["tool"]}' for i, p in enumerate(actions, 1)
              if p['state'].get('status') == 'error']
    if delivery.transcript is None:
        issue_evidence = 'Tool history unavailable. Whether an earlier error was resolved is unknown.'
    elif errors:
        issue_evidence = ('Tool errors to check: ' + ', '.join(errors)
                          + '. Read the later actions to see whether each problem was resolved.')
    else:
        issue_evidence = 'No tool error recorded. Check whether any task requirement remains unfinished.'
    print(f'File comparison starts at: {delivery.comparison_start}.')
    base.show_table('Read the claim and check its evidence', ['Report section', 'Actual report text', 'Evidence to inspect'], [
        ['Test results', report.get('test results', '(missing)'), test_evidence],
        ['Changes', report.get('changes', '(missing)'), 'Files changed since that starting point: ' + (', '.join(changed) or 'None')],
        ['Unresolved issues', report.get('unresolved issues', '(missing)'),
         issue_evidence],
    ], redact=delivery.redact)
    after = base.snapshot(delivery.project)
    for name in changed:
        old, new = delivery.before.get(name, {}), after.get(name, {})
        diff = ''.join(difflib.unified_diff(old.get('text', '').splitlines(True),
            new.get('text', '').splitlines(True), fromfile=f'before/{name}', tofile=f'after/{name}'))
        base.show_details(delivery.redact(f'File change: {name}'),
            delivery.redact(diff or f"{old.get('kind', 'absent')} -> {new.get('kind', 'absent')}"),
            opened=name != 'review.md')
    for i, part in enumerate(actions, 1):
        base.show_tool_event({'part': part}, title=f'Recorded action {i}: {part["tool"]}',
                             opened=part['state'].get('status') == 'error')


def compare(before_checks: dict, after_checks: dict, delivery: Delivery) -> None:
    """Compare delivery checks and show the additional work separately."""
    base.show_table('Before and after the report follow-up', ['Check', 'Repaired program, unfinished report', delivery.label], [
        ['Program test', before_checks['test_status'], after_checks['test_status']],
        ['Report sections still missing', ', '.join(before_checks['report_gaps']) or 'None',
         ', '.join(after_checks['report_gaps']) or 'None'],
        ['File comparison starts at', before_checks['comparison_start'], after_checks['comparison_start']],
        ['Changed files', ', '.join(before_checks['changed_files']) or 'None',
         ', '.join(after_checks['changed_files']) or 'None'],
        ['Changes outside the requested work', ', '.join(before_checks['unexpected_changes']) or 'None',
         ', '.join(after_checks['unexpected_changes']) or 'None'],
        ['Report truth', 'Original claim contradicted by title-only file', 'Use the report/evidence comparison above'],
    ])
    values = usage(delivery.transcript)
    base.show_table(delivery.label + ': additional work', ['Measure', 'Recorded'], [
        ['Provider/model', delivery.record['model']],
        ['Task wall time', str(delivery.record.get('seconds')) + ' seconds'],
        ['Model responses (not HTTP request count)', str(values['responses']) if values['responses'] is not None else 'Unknown'],
        ['Tool requests', str(len(tools(delivery.transcript))) if delivery.transcript is not None else 'Unknown'],
        ['Tokens, including cached input', str(values['tokens']) if values['tokens'] is not None else 'Unknown'],
        ['Of these, cached input', str(values['cache_read']) if values['cache_read'] is not None else 'Unknown'],
        ['Billed amount', 'Unknown; a zero OpenCode cost field is not a billing receipt.'],
    ])

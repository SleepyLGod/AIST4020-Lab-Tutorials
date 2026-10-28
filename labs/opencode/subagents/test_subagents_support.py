"""Offline helper checks using synthetic records, not live model runs."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import subagents_support as support


HERE = Path(__file__).resolve().parent
AGENTS = {
    "build": {"steps": 12, "permission": {"*": "deny", "read": "allow", "task": {"*": "deny", "investigator": "allow"}}},
    "investigator": {"mode": "subagent", "steps": 6, "prompt": "Synthetic investigator instructions",
                 "permission": {"*": "deny", "read": "allow", "edit": "deny", "task": "deny"}},
}


def tool(name: str, status: str = "completed", *, child_id: str | None = None) -> dict:
    """Build an explicitly synthetic saved tool part."""
    return {"type": "tool", "tool": name, "state": {"status": status, "input": {},
        "metadata": {"sessionId": child_id} if child_id else {}, "output": "Synthetic result"}}


def transcript(sid: str, parts: list[dict], parent: str | None = None) -> dict:
    """Build a synthetic session export for parser tests."""
    return {"info": {"id": sid, "parentID": parent}, "messages": [
        {"info": {"role": "assistant", "finish": "stop", "time": {"completed": 1}}, "parts": parts}]}


class SubagentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="subagents-offline-"))
        env = support.base.isolated_environment(self.root)
        self.lab = support.Experiment(self.root, env, Path('/not-executed/opencode'),
            self.root, self.root / 'records', '1.18.34')
        self.lab.selected = {"base_url": "https://example.invalid/v1", "model": "synthetic-model"}
        self.lab.env['COURSE_API_KEY'] = 'synthetic-test-key'
        self.lab.backend = 'deepseek'
        self.lab.model = 'course/synthetic-model'
        self.lab.redact = support.base.Redactor(['synthetic-test-key'])

    def case(self, name: str = 'Synthetic case') -> support.Case:
        with contextlib.redirect_stdout(io.StringIO()):
            return self.lab.new_case(name, {"permission": {"*": "deny"}, "agent": copy.deepcopy(AGENTS)})

    def completed(self, case: support.Case, parent: dict) -> None:
        case.parent = parent
        case.result = support.RunResult(0, False, 0.5, '', '', [
            {'type': 'text', 'sessionID': parent['info']['id'], 'part': {'text': 'Synthetic answer'}}])
        case.after = support.base.snapshot(case.project)

    def test_case_configuration_preserves_visible_role_definitions(self) -> None:
        case = self.case()
        text = Path(case.env['OPENCODE_CONFIG']).read_text()
        cfg = json.loads(text)
        self.assertEqual(cfg['agent'], AGENTS)
        self.assertEqual(cfg['permission'], {'*': 'deny'})
        self.assertEqual(cfg['provider']['course']['options']['apiKey'], '{env:COURSE_API_KEY}')
        self.assertNotIn('synthetic-test-key', text)

    def test_cases_have_same_starting_files_and_distinct_state(self) -> None:
        a = self.case('A')
        (a.project / 'grades.py').write_text('# Synthetic changed file\n')
        b = self.case('B')
        self.assertEqual(a.before, b.before)
        self.assertNotEqual(a.env['HOME'], b.env['HOME'])
        self.assertNotEqual(a.project, b.project)
        self.assertIn('len(rows)', (b.project / 'grades.py').read_text())

    def test_disabled_configure_never_reads_key_or_starts_server(self) -> None:
        with patch.object(support.base, 'credential') as credential, patch.object(support.base, 'start_ollama') as ollama:
            with self.assertRaises(RuntimeError):
                self.lab.configure('custom', {}, enabled=False)
        credential.assert_not_called()
        ollama.assert_not_called()
        self.assertEqual(self.lab.selected, {})

    def test_provider_cannot_change_mid_comparison(self) -> None:
        self.case()
        with self.assertRaisesRegex(RuntimeError, 'Start a new experiment'):
            self.lab.configure('custom', {}, enabled=True)

    def test_run_only_calls_main_agent_once_and_keeps_timeout(self) -> None:
        case = self.case()
        result = support.RunResult(-15, True, 300, '', 'Synthetic timeout')
        command = [str(case.binary), 'run', '--pure', '--format', 'json',
                   '--agent', 'build', '--model', case.model, 'Synthetic user request']
        with patch.object(support.base, 'run_command', return_value=result) as run, \
                patch.object(support.base, 'show_details'), contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(case.run(command), result)
            with self.assertRaises(RuntimeError):
                case.run(command)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command, [str(case.binary), 'run', '--pure', '--format', 'json',
                                  '--agent', 'build', '--model', case.model, 'Synthetic user request'])
        self.assertEqual(run.call_args.kwargs['timeout'], 300)
        self.assertIsNotNone(case.after)

    def test_student_check_keeps_failure_and_omits_model_key(self) -> None:
        case = self.case()
        result = support.RunResult(1, False, 0.1, '', 'Synthetic failure')
        with patch.object(support.base, 'run_command', return_value=result) as run:
            self.assertIs(case.check(['python3', '-m', 'unittest', '-v']), result)
        self.assertNotIn('COURSE_API_KEY', run.call_args.kwargs['env'])
        self.assertNotIn('OPENCODE_CONFIG', run.call_args.kwargs['env'])
        self.assertIn('COURSE_API_KEY', case.env)

    def test_load_sessions_uses_recorded_child_id_and_exports_once(self) -> None:
        case = self.case()
        parent = transcript('parent', [tool('task', child_id='child'), tool('task', child_id='child')])
        child = transcript('child', [tool('read')], parent='parent')
        self.completed(case, parent)
        with patch.object(case, 'export', side_effect=[parent, child]) as export:
            case.load_sessions()
        self.assertEqual([c.args[0] for c in export.call_args_list], ['parent', 'child'])
        self.assertEqual(case.children, {'child': child})

    def test_missing_task_metadata_does_not_invent_child(self) -> None:
        case = self.case()
        failed = tool('task', 'error')
        failed['state']['metadata'] = None
        parent = transcript('parent', [failed])
        self.completed(case, parent)
        with patch.object(case, 'export', return_value=parent) as export, contextlib.redirect_stdout(io.StringIO()) as out:
            case.load_sessions()
        export.assert_called_once_with('parent')
        self.assertFalse(case.children)
        self.assertIn('without a child session ID', out.getvalue())

    def test_missing_session_is_reported_without_blocking_checks(self) -> None:
        case = self.case()
        case.result = support.RunResult(1, False, 0.1, '', 'Synthetic failure')
        case.after = support.base.snapshot(case.project)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(case.load_sessions())
        self.assertIn('No main session ID', out.getvalue())
        self.assertTrue(case.session_errors)
        with patch.object(support.base, 'show_table') as table, patch.object(support.base, 'show_details'), \
                patch.object(support.base, 'show_changes') as changes, contextlib.redirect_stdout(io.StringIO()):
            support.show_case(case, support.RunResult(1, False, 0.1, '', ''))
        self.assertIn(['Test', 'Failed'], table.call_args.args[2])
        changes.assert_called_once()

    def test_failed_child_export_keeps_other_children_and_task_result(self) -> None:
        case = self.case()
        parent = transcript('parent', [tool('task', child_id='bad'), tool('task', child_id='good')])
        child = transcript('good', [{'type': 'text', 'text': 'Synthetic advice'}], parent='parent')
        self.completed(case, parent)
        with patch.object(case, 'export', side_effect=[parent, RuntimeError('Synthetic export error'), child]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(case.load_sessions())
        self.assertIn('Synthetic export error', out.getvalue())
        self.assertIs(case.parent, parent)
        self.assertEqual(case.children, {'good': child})
        with patch.object(support.base, 'show_table'), patch.object(support.base, 'show_details'), \
                patch.object(support.base, 'show_tool_event') as events, contextlib.redirect_stdout(io.StringIO()):
            support.show_handoffs(case)
        events.assert_any_call({'part': parent['messages'][0]['parts'][0]},
                              title='What the main agent received', opened=True)

    def test_failed_parent_export_keeps_cli_evidence_and_unknown_counts(self) -> None:
        case = self.case()
        task = tool('task', 'error')
        self.completed(case, transcript('parent', []))
        case.result.events.append({'type': 'tool_use', 'part': task, 'sessionID': 'parent'})
        with patch.object(case, 'export', side_effect=ValueError('Synthetic invalid export')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(case.load_sessions())
        self.assertIsNone(case.parent)
        self.assertEqual(case.main_actions(), [task])
        with patch.object(support.base, 'show_tool_event') as events, contextlib.redirect_stdout(io.StringIO()):
            support.show_handoffs(case)
        events.assert_any_call({'part': task}, title='What the main agent received', opened=True)
        test = support.RunResult(1, False, 0.1, '', '')
        with patch.object(support.base, 'show_table') as table:
            support.compare_cases(case, case, single_test=test, delegated_test=test)
        rows = {row[0]: row[1:] for row in table.call_args.args[2]}
        self.assertIn('Unknown', rows['Task calls'][0])
        self.assertIn('Incomplete', rows['Child tool actions'][0])

    def test_reply_completion_requires_final_message_evidence(self) -> None:
        data = transcript('child', [{'type': 'text', 'text': 'Synthetic advice'}])
        self.assertEqual(support.recorded_reply(data), ('Synthetic advice', True))
        for change in ({'finish': 'tool-calls'}, {'finish': 'length'}, {'finish': 'unknown'},
                       {'time': {}}, {'error': {'message': 'Synthetic error'}}):
            with self.subTest(change=change):
                incomplete = copy.deepcopy(data)
                incomplete['messages'][0]['info'].update(change)
                self.assertFalse(support.recorded_reply(incomplete)[1])
        data['messages'].append({'info': {'role': 'assistant'}, 'parts': [tool('read', 'running')]})
        self.assertFalse(support.recorded_reply(data)[1])
        self.assertNotEqual(support.recorded_reply(data)[0], 'Synthetic advice')

    def test_unfinished_investigator_is_not_labeled_final(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', [tool('task', 'running', child_id='child')]))
        child = transcript('child', [{'type': 'text', 'text': 'I will inspect the remaining files.'}], parent='parent')
        child['messages'][0]['info'] = {'role': 'assistant', 'time': {'created': 1}}
        case.children['child'] = child
        with patch.object(support.base, 'show_table'), patch.object(support.base, 'show_details') as details, \
                patch.object(support.base, 'show_tool_event'), contextlib.redirect_stdout(io.StringIO()) as out:
            support.show_handoffs(case)
        titles = [c.args[0] for c in details.call_args_list]
        self.assertNotIn("Investigator agent's final reply", titles)
        self.assertIn("Investigator agent's last recorded text (completion not confirmed)", titles)
        self.assertIn('No completed reply confirmed', out.getvalue())
        self.assertNotIn('Final reply found in the task result: No', out.getvalue())

    def test_interrupted_main_reply_and_cli_only_error_remain_visible(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', [{'type': 'text', 'text': 'Synthetic reply'}]))
        case.result.timed_out = True
        test = support.RunResult(1, False, 0.1, '', '')
        with patch.object(support.base, 'show_table'), patch.object(support.base, 'show_details') as details, \
                patch.object(support.base, 'show_changes'), contextlib.redirect_stdout(io.StringIO()):
            support.show_case(case, test)
        self.assertNotIn("Main agent's final answer", [c.args[0] for c in details.call_args_list])
        case.parent = None
        case.result.stdout = 'Synthetic raw error with synthetic-test-key'
        case.result.events.append({'type': 'error', 'error': 'Synthetic failure'})
        with patch.object(support.base, 'show_table'), patch.object(support.base, 'show_details') as details, \
                patch.object(support.base, 'show_changes') as changes, contextlib.redirect_stdout(io.StringIO()):
            support.show_case(case, test)
        outputs = {c.args[0]: c.args[1] for c in details.call_args_list}
        self.assertIn('Recorded CLI output', outputs)
        self.assertIn('Run error', outputs)
        self.assertNotIn('synthetic-test-key', outputs['Recorded CLI output'])
        changes.assert_called_once()

    def test_export_validates_id_and_surfaces_failure(self) -> None:
        case = self.case()
        wrong = support.RunResult(0, False, 0.1, json.dumps(transcript('other', [])), '')
        with patch.object(support.base, 'run_command', return_value=wrong), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                case.export('parent')
        failed = support.RunResult(1, False, 0.1, '', 'Synthetic export failure')
        with patch.object(support.base, 'run_command', return_value=failed), \
                patch.object(support.base, 'show_details') as display, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                case.export('parent')
        self.assertIn('Synthetic export failure', display.call_args.args[1])

    def test_agent_inspection_uses_cli_tools_not_config_assumptions(self) -> None:
        case = self.case()
        info = {'tools': {'read': True, 'edit': False, 'bash': False}}
        result = support.RunResult(0, False, 0.1, json.dumps(info), '')
        with patch.object(support.base, 'run_command', return_value=result) as run, \
                patch.object(support.base, 'show_table') as table, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(case.inspect_agent('investigator'), info)
        self.assertEqual(run.call_args.args[0], [str(case.binary), 'debug', 'agent', 'investigator'])
        self.assertIn(['edit', 'No'], table.call_args.args[2])

    def test_handoff_displays_final_text_not_all_child_progress(self) -> None:
        case = self.case()
        task = tool('task', child_id='child')
        task['state']['input'] = {'prompt': 'Synthetic investigation request', 'subagent_type': 'investigator'}
        task['state']['output'] = '<task_result>Synthetic final recommendation</task_result>'
        self.completed(case, transcript('parent', [task]))
        case.children['child'] = transcript('child', [
            {'type': 'text', 'text': 'Synthetic intermediate progress'}, tool('read'),
            {'type': 'text', 'text': 'Synthetic final recommendation'}], parent='parent')
        with patch.object(support.base, 'show_table') as table, patch.object(support.base, 'show_details') as details, \
                patch.object(support.base, 'show_tool_event') as events, contextlib.redirect_stdout(io.StringIO()) as out:
            support.show_handoffs(case)
        events.assert_any_call({'part': task}, title='What the main agent received', opened=True)
        text = next(c for c in details.call_args_list if c.args[0] == "Investigator agent's final reply")
        self.assertEqual(text.args[1], 'Synthetic final recommendation')
        self.assertTrue(text.kwargs['opened'])
        self.assertIn('Final reply found in the task result: Yes', out.getvalue())
        self.assertIn(['Child parentID matches main session', 'Yes'], table.call_args_list[0].args[2])

    def test_wrong_parent_and_missing_return_are_not_marked_success(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', [tool('task', child_id='child')]))
        case.children['child'] = transcript('child', [{'type': 'text', 'text': 'Unreturned text'}], parent='other')
        with patch.object(support.base, 'show_table') as table, patch.object(support.base, 'show_details'), \
                patch.object(support.base, 'show_tool_event'), contextlib.redirect_stdout(io.StringIO()) as out:
            support.show_handoffs(case)
        self.assertIn(['Child parentID matches main session', 'No: inspect this mismatch'], table.call_args_list[0].args[2])
        self.assertIn('Final reply found in the task result: No', out.getvalue())

    def test_no_task_is_not_presented_as_delegation(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', [{'type': 'text', 'text': 'I delegated!'}]))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            support.show_handoffs(case)
        self.assertIn('does not demonstrate delegation', out.getvalue())

    def test_comparison_includes_child_actions_and_unknown_exports(self) -> None:
        a, b = self.case('A'), self.case('B')
        self.completed(a, transcript('a', [tool('read')]))
        self.completed(b, transcript('b', [tool('task', child_id='child'), tool('edit')]))
        b.children['child'] = transcript('child', [tool('read'), tool('read')], parent='b')
        test = support.RunResult(1, False, 0.1, '', 'Synthetic failing test')
        with patch.object(support.base, 'show_table') as table:
            support.compare_cases(a, b, single_test=test, delegated_test=test)
        rows = {row[0]: row[1:] for row in table.call_args.args[2]}
        self.assertEqual(rows['Child tool actions'], ['0', '2'])
        self.assertEqual(rows['Changed files'], ['None', 'None'])
        self.assertEqual(rows['Your test'], ['Failed', 'Failed'])
        b.children = {}
        with patch.object(support.base, 'show_table') as table:
            support.compare_cases(a, b, single_test=test, delegated_test=test)
        rows = {row[0]: row[1:] for row in table.call_args.args[2]}
        self.assertIn('Incomplete', rows['Child tool actions'][1])

    def test_all_changes_and_actual_final_answer_are_shown(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', [{'type': 'text', 'text': 'Synthetic incomplete repair'}]))
        (case.project / 'unexpected.txt').write_text('Synthetic unexpected file')
        case.after = support.base.snapshot(case.project)
        test = support.RunResult(1, False, 0.1, '', '')
        with patch.object(support.base, 'show_table') as table, patch.object(support.base, 'show_details') as details, \
                patch.object(support.base, 'show_changes') as changes:
            support.show_case(case, test)
        self.assertIn(['Changed files', 'unexpected.txt'], table.call_args.args[2])
        self.assertEqual(details.call_args.args[1], 'Synthetic incomplete repair')
        changes.assert_called_once_with(case.before, case.after, case.redact)

    def test_download_redacts_records_and_excludes_configuration(self) -> None:
        case = self.case()
        self.completed(case, transcript('parent', []))
        (case.records / 'synthetic.json').write_text('{"text": "synthetic-test-key"}')
        with contextlib.redirect_stdout(io.StringIO()):
            self.lab.download_records()
        archive = next(self.root.glob('subagents-records-*.zip'))
        with zipfile.ZipFile(archive) as bundle:
            self.assertFalse(any('opencode.json' in name for name in bundle.namelist()))
            self.assertTrue(all(b'synthetic-test-key' not in bundle.read(name) for name in bundle.namelist()))

    def test_notebook_cells_connect_with_synthetic_records(self) -> None:
        notebook = json.loads((HERE / 'opencode_subagents.ipynb').read_text())
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        namespace = {'lab': self.lab, 'support': support, 'copy': copy}
        runs = []

        def fake_command(args: list[str], **kwargs: object) -> support.RunResult:
            if args[0] != str(self.lab.binary):
                return support.RunResult(1, False, 0.1, '', 'Synthetic 60.0 != 90.0')
            if args[1] == 'debug':
                return support.RunResult(0, False, 0.1, json.dumps({'tools': {'read': True, 'edit': False}}), '')
            if args[1] == 'run':
                runs.append(args)
                return support.RunResult(0, False, 0.1, '', '', [{'type': 'text', 'sessionID': f'parent{len(runs)}'}])
            if args[1] == 'export':
                sid = args[-1]
                data = transcript(sid, [tool('task', child_id='child')] if sid == 'parent2' else [])
                if sid == 'child':
                    data = transcript('child', [tool('read'), {'type': 'text', 'text': 'Synthetic final'}], parent='parent2')
                return support.RunResult(0, False, 0.1, json.dumps(data), '')
            raise AssertionError(f'Unexpected synthetic command: {args}')

        sequence = ['initial-check', 'single-config', 'single-run', 'single-check', 'investigator-config',
                    'delegated-run', 'handoff', 'delegated-check', 'comparison', 'download']
        with patch.object(support.base, 'run_command', side_effect=fake_command), \
                patch.object(support.base, 'show_table'), patch.object(support.base, 'show_details'), \
                patch.object(support.base, 'show_tool_event'), patch.object(support.base, 'show_changes'), \
                contextlib.redirect_stdout(io.StringIO()):
            for cell_id in sequence:
                exec(compile(cells[cell_id], f'notebook:{cell_id}', 'exec'), namespace)
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(args[args.index('--agent') + 1] == 'build' for args in runs))
        self.assertEqual(namespace['single'].before, namespace['delegated'].before)
        self.assertEqual(namespace['SINGLE_AGENTS']['build']['permission']['task'], 'deny')
        self.assertEqual(namespace['DELEGATED_SETTINGS']['agent']['build']['permission']['task']['investigator'], 'allow')
        self.assertIn('120 words', namespace['INVESTIGATOR']['prompt'])

    def test_example_fails_before_fix_and_passes_in_temporary_copy(self) -> None:
        project = self.root / 'fixture-check'
        shutil.copytree(support._CHAPTER1 / 'grade-report', project)
        env = {'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1'}
        command = [sys.executable, '-m', 'unittest', '-v']
        before = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=20)
        self.assertNotEqual(before.returncode, 0)
        self.assertIn('60.0 != 90.0', before.stdout + before.stderr)
        implementation = project / 'grades.py'
        implementation.write_text(implementation.read_text().replace('len(rows)', 'len(scores)'))
        after = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn('len(rows)', (support._CHAPTER1 / 'grade-report/grades.py').read_text())


if __name__ == '__main__':
    unittest.main()

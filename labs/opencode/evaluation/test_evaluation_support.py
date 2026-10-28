"""Offline checks; synthetic error/usage records below are not live agent results."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluation_support as support

HERE = Path(__file__).resolve().parent


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = support.open_lab()
        self.output = io.StringIO()
        notebook = json.loads((HERE / 'opencode_evaluation.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if c['id'] == 'delivery-check')
        # Execute the student's actual rule, without running notebook setup or checks.
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'report_gaps'
                 or isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == 'REQUIRED_SECTIONS'
                     for target in node.targets)]
        namespace = {'support': support, 're': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'notebook-report-rule', 'exec'), namespace)
        self.report_gaps = namespace['report_gaps']
        self.required_sections = namespace['REQUIRED_SECTIONS']

    def case(self, name: str = 'incomplete') -> support.Delivery:
        with contextlib.redirect_stdout(self.output):
            return self.lab.historical(name)

    def test_all_archived_files_restore_exactly(self) -> None:
        for name in ('incomplete', 'complete', 'remediated'):
            case = self.case(name)
            self.assertEqual(support.base.snapshot(case.project), case.record['files']['after'])

    def test_interrupted_reply_is_not_labeled_as_final(self) -> None:
        case = self.case()
        # Synthetic interrupted reply, not a recorded agent run.
        text = 'I will inspect the report next.'
        case.transcript = {'messages': [{'info': {'role': 'assistant', 'finish': 'length'},
                                        'parts': [{'type': 'text', 'text': text}]}]}
        with patch.object(support.base, 'show_details') as display:
            support.show_answer(case)
        display.assert_called_once_with(f'{case.label}: Last recorded reply', text, opened=True)

    def test_missing_reply_is_labeled_as_unavailable(self) -> None:
        case = self.case()
        for transcript in (None, {'messages': []}):
            with self.subTest(transcript=transcript):
                case.transcript = transcript
                with patch.object(support.base, 'show_details') as display:
                    support.show_answer(case)
                display.assert_called_once_with(f'{case.label}: Last recorded reply',
                                                'No saved reply available.', opened=True)

    def test_real_incomplete_and_complete_reports(self) -> None:
        self.assertEqual(self.report_gaps(self.case().read('review.md')), list(self.required_sections))
        self.assertEqual(self.report_gaps(self.case('complete').read('review.md')), [])
        self.assertEqual(self.report_gaps(self.case('remediated').read('review.md')), [])

    def test_student_controls_required_sections_in_the_notebook(self) -> None:
        self.report_gaps.__globals__['REQUIRED_SECTIONS'] = ('Evidence',)
        self.assertEqual(self.report_gaps('## Changes\nChanged the report.'), ['Evidence'])
        self.assertEqual(self.report_gaps('## Evidence\nRecorded test output.'), [])
        self.assertEqual(self.report_gaps('## Evidence\n  \n-'), ['Evidence'])

    def test_report_parser_handles_case_and_code_fences(self) -> None:
        text = '```md\n## Test results\nFabricated heading inside code\n```\n'
        self.assertEqual(self.report_gaps(text), list(self.required_sections))
        text = '### TEST RESULTS\nPassed.\n## Changes\n-\n## Unresolved issues\nNone.\n'
        self.assertEqual(self.report_gaps(text), ['Changes'])

    def test_nonempty_prose_is_not_automatically_verified(self) -> None:
        text = '## Test results\nI ran a million tests.\n## Changes\nEverything.\n## Unresolved issues\nNone.\n'
        self.assertEqual(self.report_gaps(text), [])
        with contextlib.redirect_stdout(self.output):
            support.show_checks('Synthetic content', {'test_status': 'Unknown', 'report_gaps': [],
                'changed_files': [], 'unexpected_changes': [], 'comparison_start': 'Synthetic starting files'})
        self.assertIn('not automatically judged', self.output.getvalue())

    def test_report_subsection_body_counts_toward_parent(self) -> None:
        text = ('# Repair review\n## Test results\n### Command and result\n'
                'Ran python3 -m unittest -v: one test passed.\n'
                '#### Details\nNo failures.\n'
                '## Changes\nCompleted review.md only.\n## Unresolved issues\nNone.\n')
        self.assertEqual(self.report_gaps(text), [])
        sections = support.report_sections(text)
        self.assertIn('one test passed', sections['test results'])
        self.assertIn('No failures.', sections['test results'])
        self.assertNotIn('Completed review.md', sections['test results'])

    def test_empty_subheadings_do_not_count_as_report_content(self) -> None:
        text = ('## Test results\n### Command\n### Result\n'
                '## Changes\nCompleted review.md only.\n## Unresolved issues\nNone.\n')
        self.assertEqual(self.report_gaps(text), ['Test results'])

    def test_nested_code_fences_do_not_start_new_sections(self) -> None:
        text = ('## Test results\n### Output\n\x60\x60\x60text\n'
                '## Changes\nOne test passed.\n\x60\x60\x60\n## Unresolved issues\nNone.\n')
        self.assertEqual(self.report_gaps(text), ['Changes'])
        self.assertIn('## Changes', support.report_sections(text)['test results'])

    def test_historical_analysis_never_installs_or_reads_a_key(self) -> None:
        with patch.object(support.base, 'install_opencode') as install, \
             patch.object(support.base, 'credential') as credential:
            self.case()
        install.assert_not_called()
        credential.assert_not_called()

    def test_restoring_again_does_not_overwrite_previous_work(self) -> None:
        first = self.case()
        (first.project / 'review.md').write_text('My notes')
        second = self.case()
        self.assertNotEqual(first.project, second.project)
        self.assertEqual(first.read('review.md'), 'My notes')
        self.assertEqual(second.read('review.md'), '# Repair review\n')

    def test_bad_record_hash_and_path_are_rejected(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix='evaluation-invalid-'))
        with self.assertRaises(ValueError):
            support._restore({'../escape': {'kind': 'file'}}, folder)
        with self.assertRaises(ValueError):
            support._restore({'a': {'kind': 'file', 'text': 'a', 'sha256': 'wrong'}}, folder)

    def test_student_check_does_not_receive_provider_key(self) -> None:
        case = self.case()
        case.env['COURSE_API_KEY'] = 'synthetic-test-key'
        case.env['OPENCODE_CONFIG'] = '/synthetic/config'
        expected = support.base.RunResult(1, False, 0, '', 'Synthetic failed test')
        with patch.object(support.base, 'run_command', return_value=expected) as run:
            self.assertIs(case.check(['python3', '-m', 'unittest', '-v']), expected)
        self.assertNotIn('COURSE_API_KEY', run.call_args.kwargs['env'])
        self.assertNotIn('OPENCODE_CONFIG', run.call_args.kwargs['env'])

    def test_disabled_setup_cannot_contact_provider(self) -> None:
        with patch.object(support.base, 'configure_backend') as configure:
            with self.assertRaises(RuntimeError):
                self.lab.prepare_live(self.case(), 'deepseek', {}, settings={}, enabled=False)
        configure.assert_not_called()

    def test_live_setup_preserves_starting_bytes_and_permissions(self) -> None:
        failed = self.case()
        def configure(*args: object, **kwargs: object) -> tuple[dict, support.base.Redactor]:
            kwargs['env']['COURSE_API_KEY'] = 'synthetic-test-key'
            return {'base_url': 'https://example.invalid/v1', 'model': 'synthetic'}, support.base.Redactor(['synthetic-test-key'])
        with patch.object(support.base, 'configure_backend', side_effect=configure), \
             patch.object(support.base, 'install_opencode', return_value=Path('/not-executed/opencode')), \
             contextlib.redirect_stdout(self.output):
            live, binary, model = self.lab.prepare_live(failed, 'deepseek', {},
                settings={'permission': {'*': 'deny'},
                          'agent': {'build': {'permission': failed.record['permissions'], 'steps': 12}}}, enabled=True)
        self.assertEqual(support.base.snapshot(live.project), failed.record['files']['after'])
        text = Path(live.env['OPENCODE_CONFIG']).read_text()
        config = json.loads(text)
        self.assertEqual(config['agent']['build']['permission'], failed.record['permissions'])
        self.assertEqual(config['agent']['build']['steps'], 12)
        self.assertNotIn('synthetic-test-key', text)
        self.assertEqual(model, 'course/synthetic')

    def test_timeout_is_saved_and_never_retried(self) -> None:
        case = self.case()
        expected = support.base.RunResult(-15, True, 300, '', 'Synthetic timeout')
        command = ['opencode', 'run', '--format', 'json', 'Synthetic task']
        with patch.object(support.base, 'run_command', return_value=expected) as run, \
             contextlib.redirect_stdout(self.output):
            self.assertIs(self.lab.run_once(case, command, timeout=300), expected)
            with self.assertRaises(RuntimeError):
                self.lab.run_once(case, command, timeout=300)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs['timeout'], 300)
        self.assertTrue(json.loads((case.records / 'run.json').read_text())['timed_out'])
        self.assertEqual(case.read('review.md'), '# Repair review\n')

    def test_missing_session_does_not_invent_history(self) -> None:
        case = self.case()
        case.transcript = None
        case.result = support.base.RunResult(1, False, 1, '', 'Synthetic provider error')
        with patch.object(support.base, 'run_command') as run, contextlib.redirect_stdout(self.output):
            self.lab.collect(case, Path('/not-executed/opencode'))
            support.show_evidence(case)
        run.assert_not_called()
        self.assertIsNone(case.transcript)
        self.assertIn('actions are unknown', self.output.getvalue())
        self.assertIn('Whether an earlier error was resolved is unknown', self.output.getvalue())
        self.assertNotIn('No tool error recorded', self.output.getvalue())

    def test_tool_error_is_not_labelled_an_unresolved_task(self) -> None:
        case = self.case()
        with patch.object(support.base, 'show_table') as table, \
             patch.object(support.base, 'show_details'), \
             patch.object(support.base, 'show_tool_event') as event, \
             contextlib.redirect_stdout(self.output):
            support.show_evidence(case)
        rows = next(call.args[2] for call in table.call_args_list
                    if call.args[0] == 'Read the claim and check its evidence')
        issue = next(row[2] for row in rows if row[0] == 'Unresolved issues')
        self.assertIn('Tool errors to check: Action 1: grep', issue)
        self.assertIn('later actions', issue)
        self.assertTrue(event.call_args_list[0].kwargs['opened'])

    def test_file_comparison_names_each_starting_point(self) -> None:
        failed, saved = self.case(), self.case('remediated')
        self.assertEqual(failed.comparison_start, 'Before the original code repair')
        self.assertEqual(saved.comparison_start, 'Before completing the report (program already repaired)')
        def checks(case: support.Delivery) -> dict:
            return {'test_status': 'Synthetic status', 'report_gaps': [], 'unexpected_changes': [],
                    'changed_files': support.changed_files(case.before, support.base.snapshot(case.project)),
                    'comparison_start': case.comparison_start}
        with patch.object(support.base, 'show_table') as table:
            support.compare(checks(failed), checks(saved), saved)
        rows = table.call_args_list[0].args[2]
        self.assertIn(['File comparison starts at', failed.comparison_start, saved.comparison_start], rows)
        self.assertIn(['Changed files', 'grades.py', 'review.md'], rows)

    def test_invalid_export_keeps_inspection_available(self) -> None:
        case = self.case()
        case.transcript = None
        case.result = support.base.RunResult(0, False, 1, '', '', [{'sessionID': 'synthetic-id'}])
        with patch.object(support.base, 'run_command', return_value=support.base.RunResult(0, False, 0, '{}', '')), \
             contextlib.redirect_stdout(self.output):
            self.lab.collect(case, Path('/not-executed/opencode'))
        self.assertIsNone(case.transcript)
        self.assertIn('Cannot read', case.export_error)

    def test_usage_deduplicates_and_keeps_missing_unknown(self) -> None:
        message = {'info': {'id': 'synthetic-id', 'role': 'assistant',
                            'tokens': {'total': 20, 'cache': {'read': 4}}}}
        self.assertEqual(support.usage({'messages': [message, message]})['tokens'], 20)
        self.assertIsNone(support.usage(None)['tokens'])
        self.assertIsNone(support.usage({'messages': [{'info': {'id': 'x', 'role': 'assistant'}}]})['tokens'])

    def test_recorded_remediation_and_report_claims_match_evidence(self) -> None:
        failed, saved = self.case(), self.case('remediated')
        self.assertEqual(saved.before, failed.record['files']['after'])
        self.assertEqual(support.changed_files(saved.before, support.base.snapshot(saved.project)), ['review.md'])
        actions = support.tools(saved.transcript)
        test = next(p for p in actions if p['tool'] == 'bash')
        self.assertEqual(test['state']['metadata']['exit'], 0)
        self.assertIn(test['state']['output'].strip(), saved.read('review.md'))
        self.assertEqual(support.usage(saved.transcript)['tokens'], 27637)

    def test_no_unexpected_change_is_hidden(self) -> None:
        case = self.case('remediated')
        (case.project / 'extra.txt').write_text('Synthetic extra file')
        (case.project / 'scores.csv').write_text('Synthetic changed data')
        changed = support.changed_files(case.before, support.base.snapshot(case.project))
        self.assertIn('extra.txt', changed)
        self.assertIn('scores.csv', changed)

    def test_entire_notebook_runs_offline_without_install_or_model_calls(self) -> None:
        notebook = json.loads((HERE / 'opencode_evaluation.ipynb').read_text())
        state = {'__name__': '__main__'}
        old_cwd = Path.cwd()
        try:
            os.chdir(HERE)
            with patch.object(support.base, 'install_opencode') as install, \
                 patch.object(support.base, 'configure_backend') as configure, \
                 contextlib.redirect_stdout(self.output):
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        source = ''.join(cell['source'])
                        ast.parse(source)
                        exec(compile(source, cell['id'], 'exec'), state)
            install.assert_not_called()
            configure.assert_not_called()
        finally:
            os.chdir(old_cwd)
        self.assertTrue(state['incomplete_checks']['test_passed'])
        for name in ('incomplete_checks', 'complete_checks', 'saved_checks'):
            self.assertEqual(state[name]['test_status'], 'Passed')
        self.assertEqual(state['incomplete_checks']['report_gaps'], list(self.required_sections))
        self.assertEqual(state['complete_checks']['report_gaps'], [])
        self.assertEqual(state['saved_checks']['report_gaps'], [])
        self.assertEqual(state['saved_checks']['unexpected_changes'], [])
        self.assertIsNone(state['live_result'])


if __name__ == '__main__':
    unittest.main()

"""Check the student lab, reusing behavior checks from the full reference."""
from __future__ import annotations

import importlib.util
import ast
import functools
import hashlib
import io
import json
import re
import subprocess
import sys
import unittest
import zipfile
from types import SimpleNamespace, ModuleType
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nbformat
import lab_support as support

ROOT = Path(__file__).resolve().parent
ORIGINAL = nbformat.read(ROOT / 'semantic_query_processing_full.ipynb', as_version=4)
NB = nbformat.read(ROOT / 'semantic_query_processing.ipynb', as_version=4)

# Load a separate module instance so the original suite keeps its own notebook.
spec = importlib.util.spec_from_file_location('_streamlined_shared_tests', ROOT / 'test_full_lab.py')
assert spec is not None and spec.loader is not None
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.NB = NB
SUPPORT_SOURCE = (ROOT / 'lab_support.py').read_text()
notebook_cell = shared.cell_with_tag


def implementation_source(tag: str) -> str:
    """Point inherited recorder tests to the module containing their actual code."""
    if tag in {'query_helpers', 'pair_helpers'}:
        return SUPPORT_SOURCE
    if tag == 'live_sentiment':
        # The shared API check now inspects Method B, the sole demo map.
        return notebook_cell('live_pair_labels')
    return notebook_cell(tag)


def pure_namespace() -> dict:
    """Test extracted functions and the calculations still visible in the notebook."""
    namespace = {**vars(support), 'support': support}
    definitions = [node for cell in NB.cells if cell.cell_type == 'code'
                   for node in ast.parse(cell.source).body
                   if isinstance(node, ast.FunctionDef) and node.name in shared.FUNCTIONS | {'candidate_pair_quality', 'candidate_pair_report'}
                   and node.name != 'validate_search_record']
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<teaching calculations>', 'exec'), namespace)
    namespace['validate_search_record'] = functools.partial(
        support.validate_search_record, score_fn=namespace['score_sentiment_rows'])
    return namespace


shared.cell_with_tag = implementation_source
shared.pure_namespace = pure_namespace


class StreamlinedNotebookChecks(shared.NotebookChecks):
    """Adapt placement checks; the shared behavior and source tests still apply."""

    def test_schema_syntax_and_empty_outputs(self) -> None:
        nbformat.validate(NB)
        for cell in NB.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        ast.parse(SUPPORT_SOURCE)
        ast.parse(support.WORKER_CODE)
        ast.parse(support.PAIR_AUDIT_CODE)

    def test_model_gate_and_no_label_input(self) -> None:
        namespace = {}
        with redirect_stdout(io.StringIO()):
            exec(notebook_cell('provider_choice'), namespace)
        self.assertFalse(namespace['ENABLE_MODEL_CALLS'])
        for tag in ('query_inputs', 'define_filter', 'live_pair_labels'):
            self.assertNotIn('_labels.json', notebook_cell(tag))
            self.assertNotIn('scoreSentiment', notebook_cell(tag))
        with patch.object(support, 'run_logged') as run:
            with self.assertRaisesRegex(RuntimeError, 'Enable model calls'):
                support.run_query(None, 'disabled', run_dir=ROOT, model_ref='test', enabled=False)
            run.assert_not_called()
        self.assertIn('frame.collect(max_threads=1)', support.WORKER_CODE)
        self.assertIn('enabled=ENABLE_MODEL_CALLS', notebook_cell('query_helpers'))
        self.assertIn('enabled=ENABLE_MODEL_CALLS', notebook_cell('pair_helpers'))


shared.NotebookChecks = StreamlinedNotebookChecks


class StreamlinedMOARChecks(shared.MOARChecks):
    """Keep default-off coverage without the removed manual exercise."""

    def test_default_cells_do_not_run_models_or_open_test_labels(self) -> None:
        namespace = {'MODEL_REF': 'test/not-a-provider'}
        exec(shared.cell_with_tag('moar_controls'), namespace)
        self.assertFalse(namespace['RUN_MOAR'])
        self.assertIsNone(namespace['SAVED_MOAR_RECORD'])
        with redirect_stdout(io.StringIO()):
            for tag in ('live_moar', 'moar_candidate_table', 'moar_candidate_diff',
                        'moar_cost_plot', 'prompt_reference_case', 'heldout_test'):
                exec(shared.cell_with_tag(tag), namespace)
        self.assertIsNone(namespace['moar_record'])
        self.assertFalse(namespace['RUN_HELD_OUT_TEST'])
        source = shared.cell_with_tag('heldout_test')
        self.assertLess(source.index('selected-before-test.json'), source.index("DATA_DIR / 'test.json'"))
        self.assertLess(source.index('final_run = run_query'), source.index("DATA_DIR / 'test_labels.json'"))


shared.MOARChecks = StreamlinedMOARChecks


class PairCallChecks(shared.CallRecordChecks):
    """Exercise the same recorder, inspecting a pair rather than a single review."""

    def test_inspection_works_when_filter_returns_no_rows(self) -> None:
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Fixture {i}'} for i in range(2)]
        call = {'request': {'messages': [{'role': 'user', 'content': 'Fixture 0 / Fixture 1'}]},
                'response': [], 'parsed_output': [{'is_match': False}],
                'pair_keys': [['film', '0'], ['film', '1']]}
        namespace = {**pure_namespace(), 'method_a': {'example_call': call}, 'pair_reviews': inputs,
                     'pairs_a': set(), 'show_reviews': lambda rows: None,
                     'display': lambda value: None, 'HTML': lambda value: value}
        with redirect_stdout(io.StringIO()) as captured:
            exec(notebook_cell('inspect_saved_prompt'), namespace)
        self.assertIn('is_match = False', captured.getvalue())
        self.assertIn('leave this pair out', captured.getvalue())
        namespace['method_a'] = {'example_call': None}
        with self.assertRaisesRegex(ValueError, 'No complete pair-call'):
            exec(notebook_cell('inspect_saved_prompt'), namespace)


shared.CallRecordChecks = PairCallChecks


class PairPresentationChecks(shared.PresentationChecks):
    """Adapt displays to the single-query teaching sequence."""

    def test_candidate_view_retains_full_diff_and_escapes_stored_text(self) -> None:
        original = {'operations': [{'name': 'sentiment', 'type': 'map', 'prompt': 'Old question'}]}
        for name in ('sentiment', 'renamed-operation'):
            changed = {'operations': [{'name': name, 'type': 'map',
                                      'prompt': 'New question\n<script>test-only</script>'}]}
            record = {'initial_config': original, 'plans': [
                {'id': 'baseline', 'config': original, 'outputs': []},
                {'id': 'candidate', 'config': changed, 'outputs': []}]}
            displayed = []
            with patch('IPython.display.display', side_effect=displayed.append), redirect_stdout(io.StringIO()):
                support.show_candidate_change(record, 'candidate', [], {}, {})
            details = displayed[0].data
            self.assertIn('Full query configuration difference', details)
            self.assertIn('&lt;script&gt;test-only&lt;/script&gt;', details)
            self.assertNotIn('<script>', details)
            self.assertIn('Old question', details)
            self.assertTrue(details.startswith('<details>' if name == 'sentiment' else '<details open>'))

    def test_inspection_shows_compact_decision_and_full_escaped_call(self) -> None:
        import html
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'<script>fixture {i}</script>'}
                  for i in range(2)]
        pair = tuple(support.review_key(row) for row in inputs)
        for matched in (True, False):
            call = {'request': {'messages': [{'role': 'user', 'content': '\n'.join(r['reviewText'] for r in inputs)}]},
                    'response': [{'content': '<script>output</script>'}],
                    'parsed_output': [{'is_match': matched}], 'pair_keys': [list(key) for key in pair]}
            rendered = []
            namespace = {**self.ns, 'method_a': {'example_call': call}, 'pair_reviews': inputs,
                         'pairs_a': {pair} if matched else set(), 'show_reviews': lambda rows: None,
                         'HTML': lambda value: value, 'display': rendered.append}
            with redirect_stdout(io.StringIO()) as captured:
                exec(notebook_cell('inspect_saved_prompt'), namespace)
            self.assertIn(f'is_match = {matched}', captured.getvalue())
            self.assertIn(html.escape(json.dumps(call, ensure_ascii=False, indent=2)), rendered[0])
            self.assertNotIn('<script>', rendered[0])

    def test_three_review_example_has_six_actual_nonself_pairs(self) -> None:
        import pandas as pd
        displayed = []
        demo = json.loads((ROOT/'data/demo.json').read_text())
        namespace = {**self.ns, 'demo_index': support.index_inputs(demo), 'pd': pd,
                     'show_reviews': lambda rows: None, 'display': displayed.append}
        with redirect_stdout(io.StringIO()):
            exec(notebook_cell('pair_inputs'), namespace)
            exec(notebook_cell('pair_candidates_example'), namespace)
        rows = displayed[0].to_dict('records')
        allowed = {r['reviewId'] for r in namespace['pair_reviews'][:3]}
        self.assertEqual({(r['Left review ID'], r['Right review ID']) for r in rows},
                         {(a, b) for a in allowed for b in allowed if a != b})
        self.assertEqual(len(namespace['pair_reviews']), 8)

    def test_macro_example_reuses_existing_predictions_and_score(self) -> None:
        expected = {('film', '1'): 'POSITIVE', ('film', '2'): 'NEGATIVE'}
        for outputs in ([], [{'id': 'film', 'reviewId': '1', 'sentiment': 'POSITIVE'}]):
            namespace = {**self.ns, 'method_b_labels': outputs, 'reference': expected}
            with redirect_stdout(io.StringIO()) as captured:
                exec(notebook_cell('macro_f1_example'), namespace)
            self.assertEqual(namespace['demo_score'], self.ns['score_sentiment_rows'](outputs, expected))
            self.assertIn('not a MOAR result', captured.getvalue())


shared.PresentationChecks = PairPresentationChecks


class AuditedPairChecks(shared.PairChecks):
    """The first captured call now records its input IDs."""

    def test_pair_recorder_rejects_parse_failure_without_another_call(self) -> None:
        from typing import Any
        definitions = [n for n in ast.parse(support.PAIR_AUDIT_CODE).body if isinstance(n, ast.FunctionDef)]
        ns = {'Any': Any, 'pair_judgments': [], 'parsed_pair_answers': [], 'example_call': None,
              'base_recorded_parse': lambda *a, **k: [{'is_match': False}]}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pair audit>', 'exec'), ns)
        calls = []

        def compare(*args: Any, **kwargs: Any) -> tuple:
            calls.append(1)
            ns['example_call'] = {'parsed_output': [{'is_match': False}]}
            ns['audited_parse'](None, None)
            return False, 0.01

        ns['base_compare'] = compare
        self.assertEqual(ns['audited_compare'](None, 'prompt', 'model', *self.inputs[:2]), (False, 0.01))
        self.assertEqual(len(calls), 1)
        self.assertEqual(ns['example_call']['pair_keys'], [list(support.review_key(r)) for r in self.inputs[:2]])
        ns['base_compare'] = lambda *a, **k: (False, 0.01)
        with self.assertRaisesRegex(ValueError, 'valid Boolean'):
            ns['audited_compare'](None, 'prompt', 'model', *self.inputs[:2])


shared.PairChecks = AuditedPairChecks


class StreamlinedChecks(unittest.TestCase):
    """Check the copy's boundaries and the revised reading sequence."""

    def test_queries_prompts_controls_and_calculations_are_preserved(self) -> None:
        original_by_tag = {tag: c.source for c in ORIGINAL.cells for tag in c.metadata.get('tags', [])}
        for tag in ('provider_choice', 'docetl_settings', 'define_filter', 'live_first_filter',
                    'define_pair_query', 'live_pair_join', 'compare_pair_sets', 'pair_metrics', 'moar_controls'):
            if tag in {'provider_choice', 'moar_controls'}:
                actual = ast.parse(notebook_cell(tag)).body
                expected = ast.parse(original_by_tag[tag]).body
                self.assertEqual([ast.dump(node) for node in actual[:-1]],
                                 [ast.dump(node) for node in expected], tag)
                self.assertIsInstance(actual[-1], ast.Expr)
                self.assertIsInstance(actual[-1].value, ast.Call)
                self.assertEqual(actual[-1].value.func.id, 'print')
            else:
                self.assertEqual(notebook_cell(tag), original_by_tag[tag], tag)
        old = ast.parse(original_by_tag['pair_helpers'])
        old_pairing = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'pairs_from_labels')
        new = ast.parse(notebook_cell('live_pair_labels'))
        self.assertEqual(ast.dump(old_pairing), ast.dump(new.body[0]))
        tags = {tag for c in NB.cells for tag in c.metadata.get('tags', [])}
        self.assertFalse({'live_full_filter', 'live_sentiment', 'keyword_comparison'} & tags)
        self.assertIn('first_four = pair_reviews[:4]', notebook_cell('filter_inputs'))
        self.assertIn("method_b_labels = support.normalize_labels(method_b['rows'])", notebook_cell('live_pair_labels'))
        self.assertIn("movie_statistics(pair_reviews, method_b_labels)", notebook_cell('statistics_display'))

    def test_evidence_and_safety_entries_remain(self) -> None:
        tags = {tag for c in NB.cells for tag in c.metadata.get('tags', [])}
        self.assertTrue({'inspect_saved_prompt', 'pair_metrics',
                         'moar_candidate_diff', 'moar_record_helpers', 'heldout_test'} <= tags)
        self.assertFalse({'manual_prompt_exercise', 'manual_change_note',
                          'fusion_strategy_source', 'code_strategy_source'} & tags)
        text = '\n'.join(c.source for c in NB.cells if c.cell_type == 'markdown')
        for phrase in ('Logical Plan Rewriting', 'Physical Execution Optimization',
                       'Prompt Rewriting', 'Operator Fusion', 'Code Generation / Code Substitution',
                       'Two iterations are not a spending cap', "upstream dataset's stated CC0 terms"):
            self.assertIn(phrase, text)

    def test_candidate_sources_and_holdout_precede_conclusion(self) -> None:
        positions = {tag: i for i, c in enumerate(NB.cells) for tag in c.metadata.get('tags', [])}
        order = ['moar_candidate_diff', 'prompt_strategy_source', 'query_execution_source',
                 'candidate_execution_source', 'candidate_score_source', 'holdout_note',
                 'heldout_test', 'moar_conclusion']
        self.assertEqual(sorted(positions[t] for t in order), [positions[t] for t in order])
        conclusion = NB.cells[positions['moar_conclusion']].source
        self.assertTrue(conclusion.startswith('## 11. Conclusion'))
        self.assertNotIn('```', conclusion)

    def test_unique_ids_and_balanced_fences(self) -> None:
        ids = [c.id for c in NB.cells]
        self.assertEqual(len(ids), len(set(ids)))
        for c in NB.cells:
            if c.cell_type == 'markdown':
                self.assertEqual(len(re.findall(r'^```', c.source, re.M)) % 2, 0, c.id)


class SupportChecks(unittest.TestCase):
    """Check distribution and setup without installing software or calling providers."""

    def test_bundle_matches_code_and_unchanged_data(self) -> None:
        path = ROOT / 'movie_lab_setup.zip'
        constants = {n.targets[0].id: ast.literal_eval(n.value)
                     for n in ast.parse(notebook_cell('data_loading')).body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == 'COURSE_BUNDLE_SHA256'}
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constants['COURSE_BUNDLE_SHA256'])
        with zipfile.ZipFile(path) as bundle:
            self.assertEqual(sorted(bundle.namelist()), ['lab_support.py', 'movie_lab_data.zip'])
            for name in bundle.namelist():
                self.assertEqual(bundle.read(name), (ROOT / name).read_bytes())

    def test_clean_local_bootstrap_and_reviews_without_network(self) -> None:
        folder = shared.test_directory()
        (folder / 'movie_lab_setup.zip').write_bytes((ROOT / 'movie_lab_setup.zip').read_bytes())
        script = notebook_cell('data_loading') + '\n' + notebook_cell('inspect_data')
        script += "\nassert len(demo) == 32\nassert 'docetl' not in sys.modules\n"
        result = subprocess.run([sys.executable, '-c', script], cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('32 reviews', result.stdout)
        self.assertEqual((folder / 'data/demo.json').read_bytes(), (ROOT / 'data/demo.json').read_bytes())

    def test_colab_download_bootstrap_in_offline_simulation(self) -> None:
        folder = shared.test_directory()
        bundle = ROOT / 'movie_lab_setup.zip'
        setup = f'''import sys
import io
import urllib.request
from pathlib import Path
from types import ModuleType
google = ModuleType('google')
colab = ModuleType('google.colab')
def download(url, timeout):
    assert url.startswith('https://raw.githubusercontent.com/SleepyLGod/AIST4020-Lab-Tutorials/')
    assert timeout == 60
    blob = Path({str(bundle)!r}).read_bytes()
    return io.BytesIO(blob)
urllib.request.urlopen = download
google.colab = colab
sys.modules.update({{'google': google, 'google.colab': colab}})
'''
        # Redirect the Colab directory to a retained temp directory; run the real cell otherwise.
        code = notebook_cell('data_loading').replace("Path('/content/docetl-movie-lab')", f'Path({str(folder)!r})')
        result = subprocess.run([sys.executable, '-c', setup + code + '\nassert IN_COLAB'],
                                cwd=folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Course files ready', result.stdout)
        self.assertIn('Downloading the course files', result.stdout)

    def test_bad_download_is_not_saved_or_imported(self) -> None:
        folder = shared.test_directory()
        setup = "import io, urllib.request\nurllib.request.urlopen = lambda *a, **k: io.BytesIO(b'wrong ZIP')\n"
        result = subprocess.run([sys.executable, '-c', setup + notebook_cell('data_loading')],
                                cwd=folder, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('download failed verification', result.stderr)
        self.assertFalse((folder / 'movie_lab_setup.zip').exists())
        self.assertFalse((folder / 'data').exists())

    def test_pair_table_escapes_full_text_and_handles_empty_results(self) -> None:
        import pandas as pd
        namespace = {'pd': pd, 'HTML': lambda value: value, 'display': lambda value: rendered.append(value),
                     'pair_index': {('movie', '1'): {'reviewText': '<script>alert(1)</script>'},
                                    ('movie', '2'): {'reviewText': 'An opposite opinion.'}}}
        rendered = []
        nodes = [node for node in ast.parse(notebook_cell('pair_helpers')).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'show_pairs']
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pair table>', 'exec'), namespace)
        with redirect_stdout(io.StringIO()):
            namespace['show_pairs']({(('movie', '1'), ('movie', '2'))})
            namespace['show_pairs'](set())
        self.assertEqual(len(rendered), 1)
        self.assertIn('&lt;script&gt;', rendered[0])
        self.assertNotIn('<script>', rendered[0])
        self.assertIn('An opposite opinion.', rendered[0])

    def test_wrong_bundle_stops_before_importing_support(self) -> None:
        folder = shared.test_directory()
        (folder / 'movie_lab_setup.zip').write_bytes(b'wrong course bundle')
        code = "import sys\n" + notebook_cell('data_loading')
        result = subprocess.run([sys.executable, '-c', code], cwd=folder, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Use the course bundle', result.stderr)
        self.assertFalse((folder / 'data').exists())

    def test_provider_configuration_does_not_switch_or_mutate_options(self) -> None:
        original = {'max_tokens': 512, 'num_retries': 0}
        with patch.object(support, 'load_key') as keys, redirect_stdout(io.StringIO()):
            for provider, key in [('deepseek', 'DEEPSEEK_API_KEY'), ('nvidia', 'NVIDIA_NIM_API_KEY')]:
                options = support.configure_provider(provider, 'nvidia/nemotron-3-super-120b-a12b', original, False)
                keys.assert_called_with(key, False)
                self.assertIn('api_base', options)
            keys.reset_mock()
            options = support.configure_provider('ollama', '', original, False)
            keys.assert_not_called()
            self.assertEqual(options['num_ctx'], 8192)
            self.assertFalse(options['think'])
        self.assertEqual(original, {'max_tokens': 512, 'num_retries': 0})

    def test_logs_preserve_failures_and_timeouts(self) -> None:
        log = shared.test_directory() / 'run.log'
        for error, expected in [(subprocess.CalledProcessError(7, ['test']), RuntimeError),
                                (subprocess.TimeoutExpired(['test'], 1), TimeoutError)]:
            with patch.object(support.subprocess, 'run', side_effect=error):
                with self.assertRaises(expected) as raised:
                    support.run_logged(['test'], log, timeout=1)
                self.assertIn(str(log), str(raised.exception))
        self.assertTrue(log.exists())

    def test_pair_recorder_has_no_global_worker_mutation(self) -> None:
        before = support.WORKER_CODE
        with patch.object(support, 'run_query', return_value={'rows': []}) as query:
            support.run_pair_query(None, run_dir=ROOT, model_ref='test', enabled=False)
            worker = query.call_args.kwargs['worker_code']
            ast.parse(worker)
            self.assertIn(support.PAIR_AUDIT_CODE, worker)
            self.assertFalse(query.call_args.kwargs['enabled'])
        self.assertEqual(before, support.WORKER_CODE)

    def test_query_keeps_complete_record_and_uses_requested_deadline(self) -> None:
        from types import SimpleNamespace
        folder = shared.test_directory()
        frame = SimpleNamespace(to_yaml=lambda: 'operations: []\n')
        record = {'rows': [], 'elapsed_seconds': 0.1, 'token_usage': {},
                  'reported_cost_usd': None, 'example_call': None}

        def finish(command: list[str], log_path: Path, *, timeout: int, cwd: Path) -> None:
            self.assertEqual(timeout, 17)
            self.assertEqual(log_path, cwd / 'execution.log')
            self.assertEqual(Path(command[2]).read_text(), frame.to_yaml())
            self.assertEqual(Path(command[1]).read_text(), support.WORKER_CODE)
            Path(command[3]).write_text(json.dumps(record))

        with patch.object(support, 'run_logged', side_effect=finish), redirect_stdout(io.StringIO()):
            result = support.run_query(frame, 'test-only', 17, run_dir=folder,
                                       model_ref='test/not-a-provider', enabled=True)
        for key, value in record.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['model'], 'test/not-a-provider')
        self.assertEqual(result['cache'], 'bypassed')
        self.assertEqual(json.loads(next(folder.glob('*/result.json')).read_text()), result)


class QuerySpineChecks(unittest.TestCase):
    """Check new boundaries without providers, credentials, or model responses."""

    def test_method_b_names_its_missing_prerequisite(self) -> None:
        tree = ast.parse(notebook_cell('live_pair_labels'))
        prefix = ast.Module(body=tree.body[:2], type_ignores=[])
        with self.assertRaisesRegex(RuntimeError, 'Run Method A in Section 2'):
            exec(compile(prefix, '<prerequisite check>', 'exec'), pure_namespace())

    def test_count_display_explains_the_actual_ordered_pairs(self) -> None:
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Fixture {i}'} for i in range(3)]
        ns = pure_namespace()
        for labels, count in ((['POSITIVE', 'POSITIVE', 'NEGATIVE'], 4), (['POSITIVE'] * 3, 0)):
            outputs = [{**row, 'sentiment': label} for row, label in zip(inputs, labels)]
            namespace = {**ns, 'pair_reviews': inputs, 'method_b_labels': outputs,
                         'pairs_b': ns['pairs_from_labels'](inputs, outputs)}
            with redirect_stdout(io.StringIO()):
                exec(notebook_cell('statistics_display'), namespace)
            self.assertEqual(namespace['expected_pair_count'], count)

    def test_labels_preserve_raw_answers_and_reject_nonbinary_predictions(self) -> None:
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Fixture {i}'} for i in range(2)]
        raw = [{**inputs[0], 'sentiment': ' positive '}, {**inputs[1], 'sentiment': 'negative'}]
        normalized = support.normalize_labels(raw)
        self.assertEqual([r['sentiment'] for r in raw], [' positive ', 'negative'])
        self.assertEqual([r['sentiment'] for r in normalized], ['POSITIVE', 'NEGATIVE'])
        ns = pure_namespace()
        expected = {support.review_key(r): label for r, label in zip(inputs, support.LABELS)}
        self.assertEqual(ns['score_sentiment_rows'](raw, expected)['macro_f1'], 1.0)
        report = ns['candidate_pair_report'](inputs, raw, expected)
        self.assertEqual(report['Pair F1'], 1.0)
        for bad in (raw[:1], [], raw + [raw[0]], [raw[0], {**raw[1], 'sentiment': 'neutral'}],
                    [raw[0], {**raw[1], 'reviewId': 'unknown'}]):
            report = ns['candidate_pair_report'](inputs, bad, expected)
            self.assertIsNone(report['Pair F1'])
            self.assertNotEqual(report['Pair check'], 'Complete')
        invalid = ns['score_sentiment_rows']([{**raw[0], 'sentiment': 'neutral'}], expected)
        self.assertEqual(invalid['missing_rows'], 1)
        self.assertEqual(invalid['invalid_labels'], 1)

    def test_candidate_projection_keeps_original_reviews_for_pairing(self) -> None:
        inputs = [{'id': 'film', 'reviewId': 'one', 'reviewText': 'Original input'}]
        output = [{**inputs[0], 'reviewText': 'Truncated', 'sentiment': 'negative'}]
        projected = support.project_candidate_labels(inputs, output)
        self.assertEqual(projected[0]['reviewText'], 'Original input')
        self.assertEqual(output[0]['reviewText'], 'Truncated')
        with self.assertRaisesRegex(ValueError, 'text changed'):
            support.validate_labels(inputs, output)

    def test_prompt_case_checks_evidence_before_display(self) -> None:
        folder = shared.test_directory()
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Fixture {i}'} for i in range(2)]
        expected = {support.review_key(row): value for row, value in zip(inputs, support.LABELS)}
        before = [{**row, 'sentiment': expected[support.review_key(row)]} for row in inputs]
        after = [{**row, 'sentiment': 'POSITIVE'} for row in inputs]
        scorer = pure_namespace()['score_sentiment_rows']
        config = {'operations': [{'name': 'classify', 'type': 'map', 'prompt': 'Old question'}]}
        record = {'format': support.MOAR_RECORD_FORMAT, 'docetl_version': '0.3.0',
                  'optimization_sha256': 'test-data', 'reference_sha256': 'test-labels',
                  'initial_config': config, 'plans': [{'id': 'baseline', 'config': config,
                      'outputs': before, 'score': scorer(before, expected), 'score_error': None,
                      'reported_score': scorer(before, expected)['macro_f1']}]}
        output = folder/'result.json'
        output.write_text(json.dumps({'rows': after}))
        comparison = {'query_result_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'baseline_score': scorer(before, expected), 'candidate_score': scorer(after, expected),
                      'before_prompt': 'Old question', 'after_prompt': 'New question'}
        summary, baseline = folder/'comparison.json', folder/'baseline.json'
        summary.write_text(json.dumps(comparison))
        baseline.write_text(json.dumps(record))
        manifest = {'sha256': {'optimization.json': 'test-data', 'optimization_labels.json': 'test-labels'}}
        case = support.load_prompt_case(summary, output, baseline, inputs, expected, manifest, score_fn=scorer)
        self.assertEqual(len(case['changes']), 1)
        self.assertEqual(case['changes'][0]['Original review'], inputs[1]['reviewText'])
        comparison['candidate_score']['macro_f1'] = 1.0
        summary.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, 'scores disagree'):
            support.load_prompt_case(summary, output, baseline, inputs, expected, manifest, score_fn=scorer)
        output.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'recorded hash'):
            support.load_prompt_case(summary, output, baseline, inputs, expected, manifest, score_fn=scorer)

    def test_json_adapter_validates_and_limits_requests_without_switching(self) -> None:
        from pydantic import BaseModel, ValidationError

        class Rewrite(BaseModel):
            prompt: str
            operations: list[str]

        requests = []
        responses = ['{"prompt": 7}', '{"prompt": "Clear question", "operations": ["classify"]}']

        def complete(*args: object, **kwargs: object) -> object:
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))])

        stats = {'requests': 0, 'validated': 0, 'validation_failures': []}
        adapter = support.make_json_adapter(complete, BaseModel, stats)
        messages = [{'role': 'user', 'content': 'Test-only prompt'}]
        adapter(model='deepseek/test-only', response_format=Rewrite, messages=messages)
        self.assertEqual(len(messages), 1)
        self.assertEqual(stats['requests'], 2)
        self.assertEqual(stats['validated'], 1)
        self.assertEqual(requests[0]['response_format'], {'type': 'json_object'})
        self.assertEqual(requests[1]['model'], 'deepseek/test-only')
        responses.extend(['{}', '{}'])
        with self.assertRaises(ValidationError):
            adapter(model='deepseek/test-only', response_format=Rewrite, messages=messages)
        limited = support.make_json_adapter(complete, BaseModel, stats, limit=4)
        with self.assertRaisesRegex(RuntimeError, 'limit reached'):
            limited(model='deepseek/test-only', response_format=Rewrite, messages=messages)
        with patch.object(support, 'make_json_adapter') as factory:
            with support.moar_response_format('ollama_chat/test-only') as untouched:
                self.assertEqual(untouched['requests'], 0)
            factory.assert_not_called()

    def test_adapter_restores_imported_aliases_even_after_error(self) -> None:
        llm, doc = ModuleType('litellm'), ModuleType('docetl.fake_for_test')
        original = lambda *a, **k: None
        llm.completion = doc.completion = original
        late = ModuleType('docetl.late_for_test')
        with patch.dict(sys.modules, {'litellm': llm, 'docetl.fake_for_test': doc, 'docetl.late_for_test': late}):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with support.moar_response_format('deepseek/test-only') as stats:
                    self.assertIs(llm.completion, doc.completion)
                    self.assertIsNot(llm.completion, original)
                    late.completion = llm.completion
                    self.assertEqual(stats['requests'], 0)
                    raise RuntimeError('test failure')
            self.assertIs(llm.completion, original)
            self.assertIs(doc.completion, original)
            self.assertIs(late.completion, original)

    def test_sequential_notebook_dependency_flow_with_offline_fixture_answers(self) -> None:
        """Run teaching cells in order; all query execution is replaced in this test only."""
        import pandas as pd
        import time
        import uuid
        import yaml

        class Frame:
            def __init__(self, path: str) -> None:
                self.rows = json.loads(Path(path).read_text())
                self.prompt = ''

            def equijoin(self, other: object, **kwargs: object) -> object:
                self.prompt = kwargs['comparison_prompt']
                return self

            def filter(self, **kwargs: object) -> object:
                return self

            def map(self, **kwargs: object) -> object:
                self.prompt = kwargs['prompt']
                return self

            def to_yaml(self) -> str:
                return yaml.safe_dump({'operations': [{'name': 'classify', 'type': 'map', 'prompt': self.prompt}]})

        folder = shared.test_directory()
        namespace = {**pure_namespace(), 'DATA_DIR': ROOT/'data', 'RUNS_DIR': folder,
                     'manifest': json.loads((ROOT/'data/manifest.json').read_text()),
                     'pd': pd, 'time': time, 'uuid': uuid, 'IN_COLAB': False,
                     'HTML': lambda s: s, 'display': lambda value: None,
                     'docetl': SimpleNamespace(Frame=Frame, read_json=Frame)}

        def pair_run(frame: Frame, **kwargs: object) -> dict:
            pairs = [(a, b) for a in frame.rows for b in frame.rows if a['reviewId'] != b['reviewId']]
            left, right = pairs[0]
            return {'rows': [], 'model': namespace['MODEL_REF'], 'cache': 'bypassed',
                    'elapsed_seconds': 1.0, 'wall_seconds': 1.1, 'reported_cost_usd': None,
                    'pair_judgments': [{'left': list(support.review_key(a)), 'right': list(support.review_key(b)),
                                        'is_match': False} for a, b in pairs],
                    'example_call': {'request': {'messages': [{'role': 'user', 'content': left['reviewText'] + right['reviewText']}]},
                                     'response': [], 'parsed_output': [{'is_match': False}],
                                     'pair_keys': [list(support.review_key(left)), list(support.review_key(right))]}}

        def query_run(frame: Frame, name: str, *args: object, **kwargs: object) -> dict:
            rows = [] if name == 'first-filter' else [{**r, 'sentiment': 'positive' if i % 2 else 'negative'}
                                                      for i, r in enumerate(frame.rows)]
            return {'rows': rows, 'elapsed_seconds': 1.0, 'wall_seconds': 1.1, 'cache': 'bypassed',
                    'model': namespace['MODEL_REF'], 'reported_cost_usd': None}

        with patch.object(support, 'run_query', side_effect=query_run), \
             patch.object(support, 'run_pair_query', side_effect=pair_run), \
             patch.object(support, 'show_reviews'), patch.object(support, 'show_pairs'), \
             redirect_stdout(io.StringIO()):
            for cell in NB.cells:
                tags = set(cell.metadata.get('tags', []))
                if cell.cell_type != 'code' or tags & {'data_loading', 'install', 'provider_auth', 'ollama_setup'}:
                    continue
                exec(cell.source, namespace)
        self.assertEqual(len(namespace['pair_reviews']), 8)
        self.assertEqual(len(namespace['first_four']), 4)
        self.assertEqual(len(namespace['candidate_pairs']), 56)
        self.assertEqual(len(namespace['method_b_labels']), 8)
        self.assertEqual(set(namespace['reference']), set(namespace['pair_index']))
        self.assertIsNone(namespace['moar_record'])
        self.assertFalse(namespace['RUN_HELD_OUT_TEST'])


class CaseLectureChecks(unittest.TestCase):
    """Test examples and plots against data, without executing any query."""

    def setUp(self) -> None:
        import matplotlib
        matplotlib.use('Agg')
        self.inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'<script>review {i}</script>'}
                       for i in range(3)]
        self.keys = [support.review_key(row) for row in self.inputs]
        self.reference = dict(zip(self.keys, ['POSITIVE', 'NEGATIVE', 'POSITIVE']))
        self.labels = [{**row, 'sentiment': self.reference[support.review_key(row)]} for row in self.inputs]
        self.pair = (self.keys[0], self.keys[1])

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_pair_example_prefers_difference_then_shared_then_empty(self) -> None:
        reverse = tuple(reversed(self.pair))
        self.assertEqual(support.choose_example_pair({self.pair, reverse}, {reverse}), self.pair)
        self.assertEqual(support.choose_example_pair({self.pair}, {self.pair}), self.pair)
        self.assertIsNone(support.choose_example_pair(set(), set()))
        for a, b, phrase in [({self.pair}, set(), 'disagree'),
                              ({self.pair}, {self.pair}, 'agree'), (set(), set(), 'no pairs')]:
            rendered = []
            with patch('IPython.display.display', side_effect=rendered.append), redirect_stdout(io.StringIO()) as out:
                support.explain_pair(self.inputs, self.labels, a, b, self.reference, {'film': 'Film'})
            self.assertIn(phrase, out.getvalue())
            for value in rendered:
                self.assertNotIn('<script>', value.data)

    def test_matrices_preserve_direction_and_exclude_diagonal(self) -> None:
        matrix = support.pair_matrix(self.inputs, {self.pair})
        self.assertEqual(matrix, [[-1, 1, 0], [0, -1, 0], [0, 0, -1]])
        with self.assertRaises(ValueError):
            support.pair_matrix(self.inputs, {(self.keys[0], self.keys[0])})
        fig = support.plot_pair_matrices(self.inputs, {self.pair}, set(), {self.pair})
        self.assertEqual(fig.axes[0].images[0].get_array().tolist(), matrix)
        self.assertEqual(fig.axes[1].images[0].get_array().tolist(), support.pair_matrix(self.inputs, set()))
        fig.canvas.draw()

    def test_work_plots_use_measured_values_without_counting_totals(self) -> None:
        usage = {'model': {'prompt_tokens': 100, 'completion_tokens': 10, 'cached_tokens': 70, 'total_tokens': 110}}
        self.assertEqual(support.measured_tokens(usage, 'prompt_tokens'), 100)
        self.assertEqual(support.measured_tokens(usage, 'completion_tokens'), 10)
        for invalid in (None, {}, {'model': {}}, {'model': {'prompt_tokens': -1}},
                        {'model': {'prompt_tokens': float('nan')}}, {'model': {'prompt_tokens': True}}):
            self.assertIsNone(support.measured_tokens(invalid, 'prompt_tokens'))
        a = {'elapsed_seconds': 2, 'token_usage': usage}
        b = {'elapsed_seconds': 1, 'python_pairing_seconds': 0.2, 'token_usage': None}
        fig = support.plot_work_comparison(a, b)
        self.assertEqual([bar.get_height() for bar in fig.axes[0].patches], [2, 1.2])
        self.assertEqual([bar.get_height() for bar in fig.axes[1].patches], [100])
        self.assertIn('Unknown', [text.get_text() for text in fig.axes[1].texts])
        fig.canvas.draw()

    def record(self) -> dict:
        original = {'operations': [{'name': 'sentiment', 'type': 'map', 'prompt': 'Original'}]}
        return {'initial_config': original, 'plans': [
            {'id': 'baseline', 'config': original, 'outputs': self.labels},
            {'id': 'incomplete', 'config': {'operations': []}, 'outputs': self.labels[:1]},
            {'id': 'first', 'config': {'operations': [{'name': 'sentiment', 'type': 'map', 'prompt': 'First'}]},
             'outputs': self.labels, 'score': 0.2},
            {'id': 'higher', 'config': {'operations': [{'name': 'sentiment', 'type': 'map', 'prompt': 'Higher'}]},
             'outputs': self.labels, 'score': 1.0}]}

    def test_candidate_default_uses_record_order_not_score(self) -> None:
        record = self.record()
        self.assertEqual(support.first_comparable_candidate(record, self.inputs, self.reference), 'first')
        self.assertIsNone(support.first_comparable_candidate(None, self.inputs, self.reference))
        record['plans'] = record['plans'][:2]
        self.assertIsNone(support.first_comparable_candidate(record, self.inputs, self.reference))
        record['plans'] = record['plans'][:1]
        self.assertIsNone(support.first_comparable_candidate(record, self.inputs, self.reference))

    def test_candidate_view_handles_unchanged_and_incomplete_predictions(self) -> None:
        for plan, message in [('first', 'none of these sentiment labels changed'),
                               ('incomplete', 'Cannot compare complete predictions')]:
            with patch('IPython.display.display'), redirect_stdout(io.StringIO()) as out:
                support.show_candidate_change(self.record(), plan, self.inputs, self.reference, {'film': 'Film'})
            self.assertIn(message, out.getvalue())

    def test_lecture_order_and_short_plot_calls(self) -> None:
        tags = [tag for c in NB.cells for tag in c.metadata.get('tags', [])]
        self.assertLess(tags.index('pair_example_before_scores'), tags.index('pair_metrics'))
        self.assertLess(tags.index('moar_candidate_diff'), tags.index('moar_candidate_table'))
        for tag in ('plan_diagram', 'pair_matrix_display', 'work_charts'):
            self.assertNotIn('run_query', notebook_cell(tag))
            self.assertNotIn('plt.subplots', notebook_cell(tag))
        fig = support.plot_execution_plans(8)
        fig.canvas.draw()


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite,
               pattern: str | None) -> unittest.TestSuite:
    """Run shared behavior tests against the copy, then its scope checks."""
    return unittest.TestSuite([
        loader.loadTestsFromModule(shared),
        loader.loadTestsFromTestCase(StreamlinedChecks),
        loader.loadTestsFromTestCase(SupportChecks),
        loader.loadTestsFromTestCase(QuerySpineChecks),
        loader.loadTestsFromTestCase(CaseLectureChecks),
    ])


if __name__ == '__main__':
    unittest.main()

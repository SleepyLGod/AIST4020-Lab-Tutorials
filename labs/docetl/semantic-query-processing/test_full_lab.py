"""Offline checks for all eleven sections. Never invoke a model provider.

Run with: python -m unittest -v test_full_lab
Set DOCETL_SOURCE and SEMBENCH_DATA_DIR to enable source and original-data checks.
The notebook's input, output-validation, and statistics functions are tested directly.
Synthetic records below are test fixtures only, not classroom data or model results.
"""

import ast
import copy
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any

import nbformat

from prepare_data import QUOTAS, SPLITS, build_payloads, write_bundle

ROOT = Path(__file__).resolve().parent
NB = nbformat.read(ROOT / 'semantic_query_processing_full.ipynb', as_version=4)
DATA = ROOT / 'data'
FUNCTIONS = {
    'review_key', 'index_inputs', 'unpack_course_data',
    'validate_returned_rows', 'validate_labels', 'movie_statistics',
    'validate_pair_rows', 'pairs_from_labels', 'check_pair_judgments', 'pair_scores',
    'checked_reference', 'score_sentiment_rows', 'checked_reported_cost',
    'check_record_secrets', 'validate_search_record', 'rebind_single_input',
    'text_details', 'cost_for_display', 'operation_changes', 'pair_metric_explanation',
}


def cell_with_tag(tag: str) -> str:
    """Read one named notebook cell without executing it."""
    return next(cell.source for cell in NB.cells if tag in cell.metadata.get('tags', []))


def pure_namespace() -> dict:
    """Load only selected pure helpers, never notebook setup or query statements."""
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'zipfile': zipfile, 'io': io,
                 'os': os, 'copy': copy, 'MOAR_RECORD_FORMAT': 'docetl-movie-lab-moar-v1',
                 'INPUT_FIELDS': {'id', 'reviewId', 'reviewText'},
                 'LABELS': ('POSITIVE', 'NEGATIVE'),
                 'DATA_FILES': {f'{name}{suffix}.json' for name in SPLITS for suffix in ('', '_labels')}}
    for node in ast.parse(cell_with_tag('data_loading')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPECTED_MANIFEST_SHA256'
                                               for t in node.targets):
            namespace['EXPECTED_MANIFEST_SHA256'] = ast.literal_eval(node.value)
    definitions = [node for cell in NB.cells if cell.cell_type == 'code'
                   for node in ast.parse(cell.source).body
                   if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in definitions} != FUNCTIONS:
        raise AssertionError('A required notebook helper is missing.')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook helpers>', 'exec'), namespace)
    return namespace


def test_directory() -> Path:
    """Use a retained temp directory; do not delete user or temporary files."""
    return Path(tempfile.mkdtemp(prefix='docetl-lab-offline-'))


class NotebookChecks(unittest.TestCase):
    def test_schema_syntax_and_empty_outputs(self) -> None:
        nbformat.validate(NB)
        for cell in NB.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        worker = next(node.value for node in ast.parse(cell_with_tag('query_helpers')).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WORKER_CODE'
                                                             for t in node.targets))
        ast.parse(ast.literal_eval(worker))

    def test_scope_and_source_links(self) -> None:
        markdown = '\n'.join(c.source for c in NB.cells if c.cell_type == 'markdown')
        headings = re.findall(r'^## (\d+)\. (.+)$', markdown, re.MULTILINE)
        self.assertEqual([h[0] for h in headings], [str(i) for i in range(1, 12)])
        self.assertNotIn('/Users/', json.dumps(NB))
        for cell in NB.cells:
            if cell.cell_type == 'markdown':
                self.assertEqual(len(re.findall(r'^```', cell.source, re.MULTILINE)) % 2, 0)
                for snippet in re.findall(r'```python\n(.*?)\n```', cell.source, re.DOTALL):
                    ast.parse(snippet)
                    self.assertLessEqual(len(snippet.splitlines()), 25)
        self.assertIn('https://github.com/ucbepic/docetl/blob/0.3.0/', markdown)

    def test_model_gate_and_no_label_input(self) -> None:
        loader = cell_with_tag('data_loading')
        self.assertLess(loader.index("if (LAB_DIR / 'movie_lab_data.zip').exists():"),
                        loader.index('elif IN_COLAB:'))
        choice = ast.parse(cell_with_tag('provider_choice'))
        assignments = {node.targets[0].id: node.value for node in choice.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        self.assertFalse(ast.literal_eval(assignments['ENABLE_MODEL_CALLS']))
        for tag in ('query_inputs', 'define_filter', 'live_sentiment'):
            self.assertNotIn('_labels.json', cell_with_tag(tag))
            self.assertNotIn('scoreSentiment', cell_with_tag(tag))
        helpers = cell_with_tag('query_helpers')
        self.assertIn('if not ENABLE_MODEL_CALLS:', helpers)
        self.assertIn('timeout=deadline_seconds', helpers)
        self.assertIn('cwd=folder', helpers)
        self.assertIn('frame.collect(max_threads=1)', helpers)

    @unittest.skipUnless(os.environ.get('DOCETL_SOURCE'), 'Set DOCETL_SOURCE for exact source verification')
    def test_source_excerpts_and_frame_api(self) -> None:
        source = Path(os.environ['DOCETL_SOURCE'])
        tag = subprocess.check_output(['git', 'describe', '--tags', '--exact-match', 'HEAD'],
                                      cwd=source, text=True).strip()
        self.assertEqual(tag, '0.3.0')
        count = 0
        for cell in NB.cells:
            for excerpt in cell.metadata.get('source_excerpts', []):
                self.assertIn(excerpt['text'], (source / excerpt['path']).read_text())
                count += 1
        self.assertGreaterEqual(count, 4)
        tree = ast.parse((source / 'docetl/frame.py').read_text())
        frame = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Frame')
        methods = {node.name: node for node in frame.body if isinstance(node, ast.FunctionDef)}
        for name in ('filter', 'map', 'collect', 'from_yaml', 'to_yaml'):
            self.assertIn(name, methods)
        for tag, method in (('define_filter', 'filter'), ('live_sentiment', 'map')):
            call = next(node for node in ast.walk(ast.parse(cell_with_tag(tag)))
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == method)
            explicit = {arg.arg for arg in methods[method].args.kwonlyargs}
            forwarded = {keyword.arg for keyword in call.keywords} - explicit
            self.assertTrue(methods[method].args.kwarg)
            # Forwarded operation options are declared/read by the pinned implementation.
            operation_source = '\n'.join((source / path).read_text() for path in (
                'docetl/operations/map.py', 'docetl/operations/base.py'))
            for name in forwarded:
                self.assertIn(name, operation_source)


class ResultChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = pure_namespace()
        self.inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Test fixture {i}'} for i in range(4)]
        self.outputs = [{**row, 'sentiment': ('POSITIVE' if i == 0 else 'NEGATIVE')}
                        for i, row in enumerate(self.inputs)]

    def test_complete_count_and_reordered_output(self) -> None:
        stats = self.ns['movie_statistics'](self.inputs, list(reversed(self.outputs)))
        self.assertEqual(stats, [{'id': 'film', 'reviews': 4, 'positive': 1,
                                 'negative': 3, 'positive_fraction': 0.25}])

    def test_missing_rows_never_shrink_denominator(self) -> None:
        for outputs in (self.outputs[:-1], []):
            with self.assertRaisesRegex(ValueError, 'Cannot calculate percentages'):
                self.ns['movie_statistics'](self.inputs, outputs)

    def test_duplicates_unknown_ids_invalid_labels_and_changed_text(self) -> None:
        variants = [self.outputs + [self.outputs[0]],
                    [{**self.outputs[0], 'reviewId': 'unknown'}, *self.outputs[1:]],
                    [{**self.outputs[0], 'sentiment': 'NEUTRAL'}, *self.outputs[1:]],
                    [{**self.outputs[0], 'reviewText': 'changed'}, *self.outputs[1:]],
                    [{k: v for k, v in self.outputs[0].items() if k != 'sentiment'}, *self.outputs[1:]]]
        for outputs in variants:
            with self.subTest(outputs=outputs):
                with self.assertRaises(ValueError):
                    self.ns['movie_statistics'](self.inputs, outputs)

    def test_inputs_reject_gold_duplicates_and_empty_text(self) -> None:
        for inputs in ([{**self.inputs[0], 'scoreSentiment': 'POSITIVE'}],
                       [self.inputs[0], self.inputs[0]], [{**self.inputs[0], 'reviewText': ' '}], []):
            with self.assertRaises(ValueError):
                self.ns['index_inputs'](inputs)

    def test_filter_can_return_a_subset_without_changing_rows(self) -> None:
        returned, missing = self.ns['validate_returned_rows'](self.inputs, [self.inputs[0]])
        self.assertEqual(len(returned), 1)
        self.assertEqual(len(missing), 3)


class CallRecordChecks(unittest.TestCase):
    """Exercise the notebook recorder with test responses; never call a provider."""

    def setUp(self) -> None:
        tree = ast.parse(cell_with_tag('query_helpers'))
        worker = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'WORKER_CODE'
                              for target in node.targets))
        definitions = [node for node in ast.parse(worker).body if isinstance(node, ast.FunctionDef)]
        self.calls = []
        self.answer = [{'is_positive': False}]
        self.response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(model_dump=lambda **kwargs: {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'function': {'name': 'send_output',
                                            'arguments': '{"is_positive": false}'}}],
            }), finish_reason='tool_calls')])

        def completion(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((args, kwargs))
            return self.response

        self.ns = {'Any': Any, 'json': json, 'example_call': None, 'observed_response': None,
                   'original_completion': completion,
                   'original_parse': lambda *args, **kwargs: self.answer}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<recorder>', 'exec'), self.ns)

    def test_capture_preserves_call_and_parsed_result_but_excludes_credentials(self) -> None:
        messages = [{'role': 'system', 'content': 'Test instruction'},
                    {'role': 'user', 'content': 'Review: test fixture'}]
        kwargs = {'model': 'test/model', 'messages': messages,
                  'tools': [{'type': 'function', 'function': {'name': 'send_output'}}],
                  'tool_choice': {'type': 'function', 'function': {'name': 'send_output'}},
                  'api_key': 'TEST_ONLY_SECRET', 'extra_headers': {'Authorization': 'TEST_ONLY_HEADER'},
                  'api_base': 'https://example.invalid', 'max_tokens': 512}
        response = self.ns['recorded_completion'](**kwargs)
        self.assertIs(response, self.response)
        self.assertEqual(self.calls, [((), kwargs)])
        parsed = self.ns['recorded_parse'](object(), response, schema={'is_positive': 'bool'})
        self.assertIs(parsed, self.answer)
        record = self.ns['example_call']
        self.assertEqual(record['parsed_output'], [{'is_positive': False}])
        self.assertEqual(record['request']['messages'], messages)
        self.assertNotIn('TEST_ONLY_SECRET', json.dumps(record))
        self.assertNotIn('TEST_ONLY_HEADER', json.dumps(record))
        self.assertNotIn('api_base', record['request'])
        parsed[0]['sentiment'] = 'test-only later mutation'
        messages[1]['content'] = 'test-only later mutation'
        self.assertEqual(record['parsed_output'], [{'is_positive': False}])
        self.assertEqual(record['request']['messages'][1]['content'], 'Review: test fixture')

    def test_later_call_does_not_replace_first_record(self) -> None:
        first = self.ns['recorded_completion'](model='first', messages=[])
        self.ns['recorded_parse'](object(), first)
        saved = json.dumps(self.ns['example_call'])
        self.response = SimpleNamespace(choices=[])
        later = self.ns['recorded_completion'](model='later', messages=[])
        self.answer = [{'is_positive': True}]
        self.ns['recorded_parse'](object(), later)
        self.assertEqual(json.dumps(self.ns['example_call']), saved)
        self.assertEqual(len(self.calls), 2)

    def test_provider_error_is_not_hidden(self) -> None:
        def fail(**kwargs: Any) -> Any:
            raise RuntimeError('Test-only provider failure')

        self.ns['original_completion'] = fail
        with self.assertRaisesRegex(RuntimeError, 'Test-only provider failure'):
            self.ns['recorded_completion'](model='test', messages=[])
        self.assertIsNone(self.ns['example_call']['response'])

    def test_inspection_works_when_filter_returns_no_rows(self) -> None:
        row = {'id': 'film', 'reviewId': 'one', 'reviewText': 'Test-only negative review'}
        self.ns['recorded_completion'](model='test', messages=[
            {'role': 'user', 'content': 'Review: ' + row['reviewText']}])
        self.ns['recorded_parse'](object(), self.response)
        namespace = {'json': json, 'first_run': {'example_call': self.ns['example_call']},
                     'first_four': [row], 'first_kept': {}, 'show_reviews': lambda rows: None,
                     'review_key': pure_namespace()['review_key'],
                     'display': lambda value: None, 'HTML': lambda value: value,
                     'text_details': pure_namespace()['text_details']}
        with redirect_stdout(io.StringIO()) as captured:
            exec(cell_with_tag('inspect_saved_prompt'), namespace)
        self.assertIn('is_positive = False', captured.getvalue())
        self.assertIn('Filter result: leave this review out', captured.getvalue())
        namespace['first_run'] = {'example_call': None}
        with self.assertRaisesRegex(ValueError, 'no complete call record'):
            exec(cell_with_tag('inspect_saved_prompt'), namespace)


class PresentationChecks(unittest.TestCase):
    """Check display-only changes with synthetic records, without executing queries."""

    def setUp(self) -> None:
        self.ns = pure_namespace()

    def test_details_escape_all_text_and_preserve_full_record(self) -> None:
        text = '<script>alert("test")</script> & </pre><img src=x onerror=test>'
        rendered = self.ns['text_details']('<b>Record</b>', text)
        self.assertNotIn('<script>', rendered)
        self.assertNotIn('<img', rendered)
        self.assertIn(html.escape(text), rendered)
        self.assertIn('&lt;b&gt;Record&lt;/b&gt;', rendered)
        self.assertTrue(rendered.startswith('<details>'))
        self.assertTrue(self.ns['text_details']('Record', text, True).startswith('<details open>'))

    def test_inspection_shows_compact_decision_and_full_escaped_call(self) -> None:
        row = {'id': 'film', 'reviewId': 'test', 'reviewText': '<img src=x onerror=test>'}
        for kept in (True, False):
            call = {'request': {'model': 'test-only', 'messages': [
                {'role': 'user', 'content': 'Question: ' + row['reviewText']}],
                'tools': [{'name': 'test-answer-format'}]},
                'response': {'text': '<script>test-only</script>'},
                'parsed_output': [{'is_positive': kept}]}
            rendered = []
            namespace = {**self.ns, 'first_run': {'example_call': call}, 'first_four': [row],
                         'first_kept': {('film', 'test'): row} if kept else {},
                         'show_reviews': lambda rows: None,
                         'display': rendered.append, 'HTML': lambda text: text}
            with redirect_stdout(io.StringIO()) as printed:
                exec(cell_with_tag('inspect_saved_prompt'), namespace)
            self.assertIn(f'is_positive = {kept}', printed.getvalue())
            self.assertIn(html.escape(json.dumps(call, ensure_ascii=False, indent=2)), rendered[0])
            self.assertNotIn('<script>', rendered[0])
            self.assertEqual(namespace['call'], call)

    def test_pair_arithmetic_normal_and_empty_results(self) -> None:
        for predicted, expected in [({1, 2}, {2, 3, 4}), (set(), {1}), ({1}, set()), (set(), set())]:
            row = {'Method': 'Test method', **self.ns['pair_scores'](predicted, expected)}
            before = copy.deepcopy(row)
            text = self.ns['pair_metric_explanation'](row)
            self.assertEqual(row, before)
            if predicted and expected:
                self.assertIn('1/2 = 0.500', text)
                self.assertIn('1/3 = 0.333', text)
                self.assertIn('2 × 1/(2 + 3) = 0.400', text)
            else:
                self.assertIn('undefined', text)
                self.assertNotIn('nan', text)

    def test_unknown_costs_are_explicit_without_mutating_measurements(self) -> None:
        for value in (None, 0, -1, float('nan'), float('inf'), True, 'missing'):
            self.assertEqual(self.ns['cost_for_display'](value), 'Unknown')
        self.assertEqual(self.ns['cost_for_display'](0.025), 0.025)

    def test_operation_changes_are_literal_and_refuse_ambiguous_matches(self) -> None:
        before = {'operations': [{'name': 'classify', 'type': 'map', 'prompt': 'Old question'}]}
        after = copy.deepcopy(before)
        after['operations'][0]['prompt'] = 'New question'
        original = copy.deepcopy(after)
        summary = self.ns['operation_changes'](before, after)
        self.assertIn('classify / prompt', summary)
        self.assertIn('Old question', summary)
        self.assertIn('New question', summary)
        self.assertEqual(after, original)
        for variant in ({'operations': []}, {'operations': after['operations'] * 2},
                        {'operations': [{'name': 'renamed', 'type': 'map'}]},
                        {'operations': [{'name': 'classify', 'type': 'code_map'}]}):
            self.assertIsNone(self.ns['operation_changes'](before, variant))
        self.assertIn('unchanged', self.ns['operation_changes'](before, before))

    def test_candidate_view_retains_full_diff_and_escapes_stored_text(self) -> None:
        import pandas as pd
        original = {'operations': [{'name': 'sentiment', 'type': 'map', 'prompt': 'Old question'}]}
        tree = ast.parse(cell_with_tag('moar_candidate_diff'))
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'INSPECT_PLAN_ID' for t in node.targets))]
        for name in ('sentiment', 'renamed-operation'):
            config = {'operations': [{'name': name, 'type': 'map',
                       'prompt': 'New question\n<script>test-only</script>'}]}
            displayed = []
            namespace = {**self.ns, 'INSPECT_PLAN_ID': 'test-plan',
                         'moar_record': {'initial_config': original},
                         'candidate_lookup': {'test-plan': {'config': config, 'outputs': []}},
                         'pd': pd, 'HTML': lambda text: text, 'display': displayed.append,
                         'yaml': SimpleNamespace(safe_dump=lambda value, **kwargs: json.dumps(value, indent=2))}
            with redirect_stdout(io.StringIO()) as printed:
                exec(compile(tree, '<candidate display>', 'exec'), namespace)
            details = displayed[0]
            self.assertIn('Full query configuration difference', details)
            self.assertIn('&lt;script&gt;test-only&lt;/script&gt;', details)
            self.assertNotIn('<script>', details)
            self.assertIn('Old question', details)
            if name == 'sentiment':
                self.assertIn('sentiment / prompt', printed.getvalue())
                self.assertTrue(details.startswith('<details>'))
            else:
                self.assertIn('cannot identify the rewrite', printed.getvalue())
                self.assertTrue(details.startswith('<details open>'))

    def test_quality_display_handles_empty_predictions_without_hiding_counts(self) -> None:
        import pandas as pd
        inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': 'Test-only review'} for i in range(2)]
        displayed = []
        namespace = {**self.ns, 'pair_reviews': inputs, 'pd': pd, 'display': displayed.append,
                     'reference': {('film', '0'): 'POSITIVE', ('film', '1'): 'NEGATIVE'},
                     'pairs_a': set(), 'pairs_b': {(('film', '0'), ('film', '1'))},
                     'show_pairs': lambda *args, **kwargs: None}
        with redirect_stdout(io.StringIO()) as printed:
            exec(cell_with_tag('pair_metrics'), namespace)
        self.assertEqual(displayed[0]['Returned pairs'].tolist(), [0, 1])
        self.assertEqual(displayed[0]['Reference pairs'].tolist(), [2, 2])
        self.assertEqual(displayed[1].iloc[0]['Precision'], 'Undefined')
        self.assertEqual(displayed[1].iloc[0]['F1'], 0)
        self.assertIsNone(namespace['quality_table'][0]['Precision'])
        self.assertIn('Empty-set note', printed.getvalue())

    def test_three_review_example_has_six_actual_nonself_pairs(self) -> None:
        import pandas as pd
        demo = json.loads((DATA / 'demo.json').read_text())
        displays = []
        namespace = {**self.ns, 'demo_index': self.ns['index_inputs'](demo), 'pd': pd,
                     'write_inputs': lambda name, rows: Path('/test-only/not-written.json'),
                     'show_reviews': lambda rows: None, 'display': displays.append}
        with redirect_stdout(io.StringIO()):
            exec(cell_with_tag('pair_inputs'), namespace)
        pairs = displays[0].to_dict('records')
        self.assertEqual(len(pairs), 6)
        allowed = {r['reviewId'] for r in namespace['pair_reviews'][:3]}
        self.assertEqual({(r['Left review ID'], r['Right review ID']) for r in pairs},
                         {(a, b) for a in allowed for b in allowed if a != b})
        self.assertEqual(len(namespace['candidate_pairs']), 56)

    def test_macro_example_reuses_existing_predictions_and_score(self) -> None:
        expected = {('film', '1'): 'POSITIVE', ('film', '2'): 'NEGATIVE', ('film', '3'): 'NEGATIVE'}
        for outputs in ([{'id': 'film', 'reviewId': '1', 'sentiment': 'POSITIVE'},
                         {'id': 'film', 'reviewId': '2', 'sentiment': 'POSITIVE'}], []):
            namespace = {**self.ns, 'sentiment_run': {'rows': outputs}, 'reference': expected}
            with redirect_stdout(io.StringIO()) as printed:
                exec(cell_with_tag('macro_f1_example'), namespace)
            self.assertEqual(namespace['demo_score'], self.ns['score_sentiment_rows'](outputs, expected))
            self.assertIn('not a MOAR result', printed.getvalue())
            self.assertIn(f"= {namespace['demo_score']['macro_f1']:.3f}", printed.getvalue())


class DataChecks(unittest.TestCase):
    def test_local_bundle_counts_hashes_and_separation(self) -> None:
        ns = pure_namespace()
        manifest = json.loads((DATA / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
                         ns['EXPECTED_MANIFEST_SHA256'])
        ids, texts = set(), set()
        for split in SPLITS:
            rows = json.loads((DATA / f'{split}.json').read_text())
            labels = json.loads((DATA / f'{split}_labels.json').read_text())
            indexed = ns['index_inputs'](rows)
            self.assertEqual(len(rows), 32)
            keys = set(indexed)
            self.assertFalse(ids & keys)
            ids |= keys
            current_texts = {row['reviewText'] for row in rows}
            self.assertFalse(texts & current_texts)
            texts |= current_texts
            reference = {(row['id'], row['reviewId']): row['scoreSentiment'] for row in labels}
            self.assertEqual(len(reference), 32)
            self.assertEqual(set(reference), keys)
            for movie, quotas in QUOTAS.items():
                for label, count in quotas.items():
                    self.assertEqual(sum(key[0] == movie and value == label
                                         for key, value in reference.items()), count)
        for name, digest in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), digest)

    def test_upload_bundle_roundtrip_and_rejection(self) -> None:
        unpack = pure_namespace()['unpack_course_data']
        output = test_directory()
        unpack((ROOT / 'movie_lab_data.zip').read_bytes(), output)
        for path in DATA.glob('*.json'):
            self.assertEqual((output / path.name).read_bytes(), path.read_bytes())
        # Same data is safe to load again; different existing content must survive.
        unpack((ROOT / 'movie_lab_data.zip').read_bytes(), output)
        (output / 'demo.json').write_text('Test-only changed file')
        with self.assertRaises(FileExistsError):
            unpack((ROOT / 'movie_lab_data.zip').read_bytes(), output)
        self.assertEqual((output / 'demo.json').read_text(), 'Test-only changed file')
        bad_zip = io.BytesIO()
        with zipfile.ZipFile(bad_zip, 'w') as archive:
            archive.writestr('../escaped.json', 'test fixture')
        with self.assertRaises(ValueError):
            unpack(bad_zip.getvalue(), output)

    @unittest.skipUnless(os.environ.get('SEMBENCH_DATA_DIR'), 'Set SEMBENCH_DATA_DIR for original CSV checks')
    def test_sampling_is_reproducible_and_originals_unchanged(self) -> None:
        source = Path(os.environ['SEMBENCH_DATA_DIR'])
        payloads = build_payloads(source / 'Reviews.csv', source / 'Movies.csv')
        for name, contents in payloads.items():
            self.assertEqual(contents, (DATA / name).read_bytes())
        path = write_bundle(payloads, test_directory())
        self.assertEqual(path.read_bytes(), (ROOT / 'movie_lab_data.zip').read_bytes())
        original_hashes = json.loads(payloads['manifest.json'])['origin']['source_sha256']
        for name, digest in original_hashes.items():
            self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)

    def test_conflicting_duplicate_text_or_label_is_rejected(self) -> None:
        folder = test_directory()
        movies = folder / 'Movies.csv'
        with movies.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'title'])
            writer.writeheader()
        for changed in ({'reviewText': 'different'}, {'scoreSentiment': 'NEGATIVE'}):
            base = {'id': 'taken_3', 'reviewId': 'test', 'reviewText': 'Test fixture', 'scoreSentiment': 'POSITIVE'}
            reviews = folder / 'Reviews.csv'
            with reviews.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(base))
                writer.writeheader()
                writer.writerows([base, {**base, **changed}])
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate review'):
                build_payloads(reviews, movies)


class PairChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = pure_namespace()
        self.inputs = [{'id': 'movie', 'reviewId': str(i), 'reviewText': f'Test {i}'}
                       for i in range(3)]
        self.outputs = [{**r, 'sentiment': 'POSITIVE' if i == 0 else 'NEGATIVE'}
                        for i, r in enumerate(self.inputs)]

    def joined(self, left: int, right: int) -> dict:
        return {f'{key}_{side}': value for side, i in [('left', left), ('right', right)]
                for key, value in self.inputs[i].items()}

    def test_ordered_pairs_and_empty_matches(self) -> None:
        pairs = self.ns['pairs_from_labels'](self.inputs, self.outputs)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(left != right and (right, left) in pairs for left, right in pairs))
        self.assertEqual(self.ns['validate_pair_rows'](self.inputs, []), set())
        same = [{**r, 'sentiment': 'POSITIVE'} for r in self.inputs]
        self.assertEqual(self.ns['pairs_from_labels'](self.inputs, same), set())

    def test_reject_bad_join_rows(self) -> None:
        valid = self.joined(0, 1)
        cases = [[self.joined(0, 0)], [valid, valid], [{**valid, 'reviewId_left': 'unknown'}],
                 [{**valid, 'reviewText_right': 'changed'}], [None]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.ns['validate_pair_rows'](self.inputs, rows)

    def test_incomplete_or_bad_labels_cannot_shrink_denominator(self) -> None:
        for rows in [[], self.outputs[:-1], self.outputs + self.outputs[:1],
                     [{**r, 'sentiment': 'MIXED'} for r in self.outputs]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.ns['pairs_from_labels'](self.inputs, rows)

    def test_empty_score_conventions(self) -> None:
        score = self.ns['pair_scores']
        self.assertIsNone(score(set(), set())['F1'])
        self.assertIsNone(score(set(), {1})['Precision'])
        self.assertEqual(score(set(), {1})['F1'], 0)
        self.assertEqual(score({1}, set())['F1'], 0)
        self.assertEqual(score({1, 2}, {2, 3})['F1'], 0.5)

    def test_audit_requires_every_candidate(self) -> None:
        pair = (('movie', '0'), ('movie', '1'))
        judgment = {'left': list(pair[0]), 'right': list(pair[1]), 'is_match': False}
        audit = self.ns['check_pair_judgments']
        audit({pair}, [judgment], set())
        for judgments, returned in [([], set()), ([judgment, judgment], set()),
                                    ([{**judgment, 'is_match': 'false'}], set()),
                                    ([judgment], {pair})]:
            with self.assertRaises(ValueError):
                audit({pair}, judgments, returned)

    def test_fixed_subset_and_pair_api(self) -> None:
        source = cell_with_tag('pair_inputs')
        assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PAIR_REVIEW_IDS')
        ids = ast.literal_eval(assignment.value)
        rows = json.loads((DATA / 'demo.json').read_text())
        expected = sorted(r['reviewId'] for r in rows if r['id'] == 'ant_man_and_the_wasp_quantumania')[:8]
        self.assertEqual(list(ids), expected)
        self.assertEqual(len(set(ids)), 8)
        call = next(n for n in ast.walk(ast.parse(cell_with_tag('define_pair_query')))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'equijoin')
        conditions = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'blocking_conditions'))
        self.assertEqual(len(conditions), 1)
        self.assertIn(' and ', conditions[0])
        for tag in ('pair_inputs', 'define_pair_query', 'live_pair_labels'):
            self.assertNotIn('scoreSentiment', cell_with_tag(tag))
            self.assertNotIn('_labels.json', cell_with_tag(tag))

    def test_pair_recorder_rejects_parse_failure_without_another_call(self) -> None:
        tree = ast.parse(cell_with_tag('pair_helpers'))
        code = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PAIR_AUDIT_CODE')
        definitions = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
        ns = {'Any': Any, 'pair_judgments': [], 'parsed_pair_answers': [],
              'base_recorded_parse': lambda *args, **kwargs: [{'is_match': False}]}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pair recorder>', 'exec'), ns)
        calls = []

        def valid_compare(*args: Any, **kwargs: Any) -> tuple:
            calls.append('one request')
            ns['audited_parse'](None, None)
            return False, 0.01

        ns['base_compare'] = valid_compare
        result = ns['audited_compare'](None, 'prompt', 'model', self.inputs[0], self.inputs[1])
        self.assertEqual(result, (False, 0.01))
        self.assertEqual(calls, ['one request'])
        self.assertFalse(ns['pair_judgments'][0]['is_match'])
        # Model the pinned operator returning False after catching a parsing error.
        ns['base_compare'] = lambda *args, **kwargs: (False, 0.01)
        with self.assertRaisesRegex(ValueError, 'valid Boolean'):
            ns['audited_compare'](None, 'prompt', 'model', self.inputs[0], self.inputs[1])

    def test_cross_movie_pair_is_rejected(self) -> None:
        self.inputs[1]['id'] = 'other'
        with self.assertRaises(ValueError):
            self.ns['validate_pair_rows'](self.inputs, [self.joined(0, 1)])


class MOARChecks(unittest.TestCase):
    """Synthetic scoring/record checks; no candidate or model is executed."""

    def setUp(self) -> None:
        self.ns = pure_namespace()
        self.inputs = [{'id': 'film', 'reviewId': str(i), 'reviewText': f'Test-only {i}'}
                       for i in range(4)]
        self.reference = {('film', str(i)): 'POSITIVE' if i < 2 else 'NEGATIVE' for i in range(4)}
        self.outputs = [{**row, 'sentiment': self.reference[('film', row['reviewId'])]}
                        for row in self.inputs]

    def record(self) -> dict:
        score = self.ns['score_sentiment_rows'](self.outputs, self.reference)
        return {'format': 'docetl-movie-lab-moar-v1', 'docetl_version': '0.3.0',
                'optimization_sha256': 'data-hash', 'reference_sha256': 'label-hash',
                'initial_config': {}, 'plans': [{'id': 'test-only-plan', 'config': {},
                    'outputs': copy.deepcopy(self.outputs), 'score': score, 'score_error': None,
                    'reported_score': 1.0}]}

    def validate_record(self, record: dict) -> dict:
        return self.ns['validate_search_record'](record, self.reference, 'data-hash', 'label-hash')

    def test_macro_f1_keeps_missing_and_invalid_predictions(self) -> None:
        score = self.ns['score_sentiment_rows']
        self.assertEqual(score(self.outputs, self.reference)['macro_f1'], 1.0)
        partial = score(self.outputs[:2], self.reference)
        self.assertEqual(partial['macro_f1'], 0.5)
        self.assertEqual(partial['missing_rows'], 2)
        invalid = score([{**r, 'sentiment': 'MIXED'} for r in self.outputs], self.reference)
        self.assertEqual(invalid['macro_f1'], 0.0)
        self.assertEqual(invalid['invalid_labels'], 4)
        self.assertEqual(score([], self.reference)['macro_f1'], 0.0)
        reversed_labels = [{**r, 'sentiment': 'NEGATIVE' if i < 2 else 'POSITIVE'}
                           for i, r in enumerate(self.outputs)]
        self.assertEqual(score(reversed_labels, self.reference)['macro_f1'], 0.0)

    def test_duplicate_unknown_and_malformed_predictions_rejected(self) -> None:
        for rows in [self.outputs + self.outputs[:1], [{'id': 'film', 'reviewId': 'unknown'}],
                     [None], {'not': 'a list'}, [{'id': 'film'}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.ns['score_sentiment_rows'](rows, self.reference)

    def test_reference_coverage(self) -> None:
        labels = [{'id': key[0], 'reviewId': key[1], 'scoreSentiment': label}
                  for key, label in self.reference.items()]
        self.assertEqual(self.ns['checked_reference'](self.inputs, labels), self.reference)
        for rows in [labels[:-1], labels + labels[:1], [{**r, 'scoreSentiment': 'MIXED'} for r in labels]]:
            with self.assertRaises(ValueError):
                self.ns['checked_reference'](self.inputs, rows)

    def test_saved_record_scores_are_recomputed(self) -> None:
        record = self.record()
        self.assertIs(self.validate_record(record), record)
        changes = [('score', {'macro_f1': 1}), ('reported_score', 0.7),
                   ('reported_score', float('nan')), ('outputs', self.outputs[:2])]
        for field, value in changes:
            record = self.record()
            record['plans'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_record(record)
        record = self.record()
        record['plans'].append(copy.deepcopy(record['plans'][0]))
        with self.assertRaises(ValueError):
            self.validate_record(record)

    def test_invalid_saved_candidate_is_not_silently_dropped(self) -> None:
        record = self.record()
        plan = record['plans'][0]
        plan.update(outputs=self.outputs + self.outputs[:1], score=None, score_error='Duplicate ID')
        self.assertEqual(len(self.validate_record(record)['plans']), 1)
        plan['score_error'] = None
        with self.assertRaises(ValueError):
            self.validate_record(record)

    def test_record_version_data_and_credentials(self) -> None:
        for field, value in [('optimization_sha256', 'wrong'), ('reference_sha256', 'wrong'),
                             ('docetl_version', 'main'), ('format', 'other')]:
            record = self.record()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_record(record)
        record = self.record()
        record['initial_config'] = {'operations': [{'api_key': 'TEST_ONLY_NOT_A_REAL_KEY'}]}
        with self.assertRaisesRegex(ValueError, 'credential'):
            self.validate_record(record)

    def test_unknown_prices_are_not_zero_cost(self) -> None:
        cost = self.ns['checked_reported_cost']
        for value in [None, 0, -1, float('nan'), float('inf'), True, '0.01']:
            self.assertIsNone(cost(value))
        self.assertEqual(cost(0.01), 0.01)

    def test_rebind_only_changes_locations_and_cache(self) -> None:
        config = {'datasets': {'reviews': {'type': 'file', 'path': '/old/optimization.json'}},
                  'operations': [{'name': 'classify', 'type': 'map', 'prompt': '{{ input.reviewText }}'}],
                  'pipeline': {'steps': [{'name': 'one', 'input': 'reviews', 'operations': ['classify']}],
                               'output': {'path': '/old/out.json', 'type': 'file', 'intermediate_dir': '/old/cache'}}}
        original = copy.deepcopy(config)
        rebound = self.ns['rebind_single_input'](config, Path('/new/test-input.json'), Path('/new/output.json'))
        self.assertEqual(config, original)
        self.assertEqual(rebound['operations'], original['operations'])
        self.assertEqual(rebound['pipeline']['steps'], original['pipeline']['steps'])
        self.assertNotIn('/old/', json.dumps(rebound))
        self.assertTrue(rebound['bypass_cache'])
        config['operations'][0]['prompt'] += ' /old/optimization.json'
        with self.assertRaises(ValueError):
            self.ns['rebind_single_input'](config, Path('/new/test.json'), Path('/new/out.json'))
        config = copy.deepcopy(original)
        config['datasets']['another'] = {'type': 'file', 'path': '/old/extra.json'}
        with self.assertRaises(ValueError):
            self.ns['rebind_single_input'](config, Path('/new/test.json'), Path('/new/out.json'))

    def test_default_cells_do_not_run_models_or_open_test_labels(self) -> None:
        namespace = {'MODEL_REF': 'test/not-a-provider'}
        exec(cell_with_tag('moar_controls'), namespace)
        self.assertFalse(namespace['RUN_MOAR'])
        self.assertIsNone(namespace['SAVED_MOAR_RECORD'])
        # No model or file helpers are supplied: calling them would fail this test.
        with redirect_stdout(io.StringIO()):
            for tag in ('live_moar', 'moar_candidate_table', 'moar_candidate_diff',
                        'moar_cost_plot', 'manual_prompt_exercise', 'heldout_test'):
                exec(cell_with_tag(tag), namespace)
        self.assertIsNone(namespace['moar_record'])
        self.assertFalse(namespace['RUN_MANUAL_CHANGE'])
        self.assertFalse(namespace['RUN_HELD_OUT_TEST'])
        source = cell_with_tag('heldout_test')
        self.assertLess(source.index("selected-before-test.json"), source.index("DATA_DIR / 'test.json'"))
        self.assertLess(source.index('final_run = run_query'), source.index("DATA_DIR / 'test_labels.json'"))

    def test_optimizer_uses_explicit_model_and_metric_without_hidden_calls(self) -> None:
        calls = [n for n in ast.walk(ast.parse(cell_with_tag('live_moar')))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'optimize']
        self.assertEqual(len(calls), 1)
        kwargs = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(kwargs['models'], '[MODEL_REF]')
        self.assertEqual(kwargs['agent_model'], 'REWRITE_MODEL_REF')
        self.assertEqual(kwargs['eval_fn'], 'evaluate_sentiment')
        self.assertEqual(kwargs['metric_key'], "'macro_f1'")
        self.assertEqual(kwargs['max_iterations'], '2')
        self.assertEqual(kwargs['max_concurrent_agents'], '1')
        self.assertEqual(kwargs['max_threads'], '1')
        for tag in ('moar_initial_query', 'live_moar'):
            self.assertNotIn('test_labels', cell_with_tag(tag))


if __name__ == '__main__':
    unittest.main()

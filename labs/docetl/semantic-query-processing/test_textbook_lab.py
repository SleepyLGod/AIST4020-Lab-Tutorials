"""Offline checks for the active restaurant-review lab; no model calls."""
import ast
import hashlib
import io
import importlib.util
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import textbook_lab_support as lesson

ROOT = Path(__file__).parent
REFERENCE = json.loads((ROOT/'textbook_data/reference.json').read_text())
CORRECT = [{'doc_id': key, 'topic': value} for key, value in REFERENCE.items() if value]


class ScoringTests(unittest.TestCase):
    def test_correct(self) -> None:
        self.assertEqual(lesson.score(CORRECT, REFERENCE)['accuracy'], 1)

    def test_missing(self) -> None:
        self.assertEqual(lesson.score(CORRECT[:-1], REFERENCE)['correct'], 5)

    def test_wrongly_retained(self) -> None:
        self.assertEqual(lesson.score(CORRECT + [{'doc_id': 'R1', 'topic': 'Food'}], REFERENCE)['correct'], 5)

    def test_empty_is_scored(self) -> None:
        self.assertEqual(lesson.score([], REFERENCE)['correct'], 3)

    def test_swapped_topics_do_not_hide_errors(self) -> None:
        swapped = [dict(row) for row in CORRECT]
        swapped[0]['topic'], swapped[1]['topic'] = swapped[1]['topic'], swapped[0]['topic']
        self.assertEqual(lesson.score(swapped, REFERENCE)['correct'], 4)

    def test_invalid_outputs(self) -> None:
        for rows in [None, CORRECT+CORRECT[:1], [{'doc_id':'R7','topic':'Food'}],
                     [{'doc_id':'R2'}], [{'doc_id':'R2','topic':'Other'}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                lesson.score(rows, REFERENCE)

    def test_disabled_search_needs_no_runtime(self) -> None:
        self.assertIsNone(lesson.run_search(None, [], {}, '', ROOT, enabled=False))

    def test_excluded_display_preserves_scoring(self) -> None:
        before = lesson.score(CORRECT, REFERENCE)
        with patch('document_lab_support.show_rows') as show:
            lesson.show_decisions(CORRECT, REFERENCE)
        rows, columns = show.call_args.args
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['expected'], 'Excluded')
        self.assertEqual(rows[0]['actual'], 'Excluded')
        self.assertTrue(rows[0]['correct'])
        self.assertEqual(rows[3]['actual'], 'Service')
        self.assertEqual(lesson.score(CORRECT, REFERENCE), before)
        self.assertIsNone(before['decisions'][0]['actual'])


class ArtifactTests(unittest.TestCase):
    def test_source_records(self) -> None:
        reviews = json.loads((ROOT/'textbook_data/reviews.json').read_text())
        provenance = json.loads((ROOT/'textbook_data/provenance.json').read_text())
        self.assertEqual(len(reviews), 6)
        self.assertEqual({r['doc_id'] for r in reviews}, set(REFERENCE))
        for row, source in zip(reviews, provenance['rows']):
            self.assertEqual(set(row), {'doc_id', 'text'})
            self.assertEqual(hashlib.sha256(row['text'].encode()).hexdigest(), source['text_sha256'])

    def test_notebook(self) -> None:
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn('ENABLE_MODEL_CALLS = False', source)
        self.assertIn('RUN_MOAR = False', source)
        self.assertNotIn('/Users/von', source)
        self.assertNotIn('DataNeed', source)
        self.assertNotIn('RUN_MOAR = True\n', source)

    def test_bundle(self) -> None:
        bundle = ROOT/'textbook_lab_setup.zip'
        notebook = (ROOT/'semantic_query_processing.ipynb').read_text()
        self.assertIn(hashlib.sha256(bundle.read_bytes()).hexdigest(), notebook)
        with zipfile.ZipFile(bundle) as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT/name).read_bytes())

    def test_actual_saved_query(self) -> None:
        saved = json.loads((ROOT/'textbook_saved_run/lesson.json').read_text())
        self.assertEqual(saved['provenance'], 'real_colab_deepseek_run')
        ast.parse(saved['query_code'])
        source = json.loads((ROOT/'textbook_saved_run/search.json').read_text())
        self.assertTrue(any(p.get('config', {}).get('operations') == saved['operations'] for p in source['plans']))

    def test_visible_query_preserves_generated_logic(self) -> None:
        saved = json.loads((ROOT/'textbook_saved_run/lesson.json').read_text())

        class Frame:
            def __init__(self) -> None:
                self.operations = []

            def map(self, **kwargs: object) -> 'Frame':
                self.operations.append({'type': 'map', **kwargs})
                return self

            def code_filter(self, **kwargs: object) -> 'Frame':
                self.operations.append({'type': 'code_filter', **kwargs})
                return self

        namespace = {'docetl': SimpleNamespace(Frame=Frame), 'MODEL_REF': saved['model'], 'OPTIONS': {}}
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        code = next(''.join(c['source']) for c in notebook['cells'] if c['id'] == 'textbook-25')
        exec(compile(code, '<visible query>', 'exec'), namespace)
        actual = namespace['moar_query'](Frame()).operations
        self.assertEqual([op['name'] for op in actual], ['classify_review', 'keep_complaints'])
        expected = [{k: v for k, v in op.items() if k != 'name'} for op in saved['operations']]
        expected[0]['prompt'] = expected[0]['prompt'].replace(
            'return None for the topic.', 'return the string Excluded for the topic.'
        ).replace('(Food, Service, or None).', '(Food, Service, or Excluded).')
        self.assertEqual([{k: v for k, v in op.items() if k != 'name'} for op in actual], expected)

    @unittest.skipUnless(importlib.util.find_spec('docetl'), 'Requires the teaching DocETL package')
    def test_excluded_placeholder_survives_provider_parsing(self) -> None:
        from litellm import ModelResponse
        from docetl.operations.utils.api import APIWrapper
        from docetl.operations.utils.validation import validate_output_types

        parser = APIWrapper.__new__(APIWrapper)
        schema = {'keep': 'bool', 'topic': 'string'}
        for model in ['deepseek/deepseek-flash', 'ollama_chat/qwen3.5:9b-q4_K_M']:
            with self.subTest(model=model):
                response = ModelResponse(model=model, choices=[{
                    'index': 0, 'finish_reason': 'tool_calls',
                    'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
                        'id': 'offline-fixture', 'type': 'function',
                        'function': {'name': 'send_output', 'arguments': json.dumps(
                            {'keep': False, 'topic': 'Excluded'})},
                    }]},
                }])
                output = parser._parse_llm_response_helper(response, schema=schema)[0]
                self.assertEqual(output, {'keep': False, 'topic': 'Excluded'})
                valid, errors = validate_output_types(output, schema)
                self.assertTrue(valid, errors)


class ComparisonTests(unittest.TestCase):
    """Exercise displays with synthetic offline results, not model answers."""

    def setUp(self) -> None:
        import matplotlib
        matplotlib.use('Agg')

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_query_plan_diagram_distinguishes_model_and_code(self) -> None:
        from matplotlib.patches import FancyArrowPatch, Rectangle
        figure = lesson.show_query_plans()
        figure.canvas.draw()
        axis = figure.axes[0]
        labels = [text.get_text() for text in axis.texts]
        self.assertEqual(labels.count('Model'), 3)
        self.assertEqual(labels.count('Python'), 1)
        self.assertIn('code_filter', labels)
        self.assertEqual(sum(isinstance(p, Rectangle) for p in axis.patches), 4)
        self.assertEqual(sum(isinstance(p, FancyArrowPatch) for p in axis.patches), 4)
        bounds = figure.bbox
        for text in axis.texts:
            extent = text.get_window_extent(figure.canvas.get_renderer())
            self.assertTrue(bounds.contains(extent.x0, extent.y0))
            self.assertTrue(bounds.contains(extent.x1, extent.y1))

    def test_displayed_prompt_comes_from_executed_query(self) -> None:
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        by_id = {c['id']: ''.join(c['source']) for c in notebook['cells']}

        class Frame:
            def __init__(self) -> None:
                self.operations = []

            def filter(self, **kwargs: object) -> 'Frame':
                self.operations.append({'type': 'filter', **kwargs})
                return self

            def map(self, **kwargs: object) -> 'Frame':
                self.operations.append({'type': 'map', **kwargs})
                return self

            def to_yaml(self) -> str:
                return json.dumps({'operations': self.operations})

        namespace = {'docetl': SimpleNamespace(Frame=Frame, from_list=lambda *a, **k: Frame()),
                     'OPTIONS': {}, 'reviews': [], 'lesson': lesson,
                     'saved_run': {'rewritten_filter_prompt': 'Offline test rewrite'}}
        tree = ast.parse(by_id['textbook-08'])
        exec(compile(ast.Module(body=[tree.body[0]], type_ignores=[]), '<filter>', 'exec'), namespace)
        exec(by_id['textbook-20'], namespace)
        with redirect_stdout(io.StringIO()) as output:
            exec(by_id['textbook-30'], namespace)
        executed_prompt = namespace['original_query'].operations[0]['prompt']
        self.assertEqual(namespace['original_prompt'], executed_prompt)
        self.assertIn(executed_prompt, output.getvalue())
        with self.assertRaises(ValueError):
            lesson.filter_prompt(Frame())

    def test_changed_rows_charts_and_escaped_text(self) -> None:
        wrong = [dict(row) for row in CORRECT]
        wrong[1]['topic'] = 'Food'
        plans = [
            {'id': '<Original>', 'rows': CORRECT, 'execution_seconds': 2,
             'token_usage': {'offline': {'prompt_tokens': 100, 'completion_tokens': 10}}},
            {'id': 'Rewrite', 'rows': wrong, 'execution_seconds': 1,
             'token_usage': {'offline': {'prompt_tokens': 60, 'completion_tokens': 15}}},
        ]
        before = json.dumps(plans)
        with patch('IPython.display.display') as display:
            figure = lesson.compare(plans, REFERENCE)
        table = display.call_args_list[0].args[0].data
        details = display.call_args_list[1].args[0].data
        self.assertIn('&lt;Original&gt;', table)
        self.assertNotIn('<Original>', table)
        self.assertEqual(table.count('background:#fff1c2'), 1)
        self.assertIn('Changed', table)
        self.assertIn('Unknown', details)
        self.assertIn('<details>', details)
        self.assertEqual([p.get_height() for p in figure.axes[0].patches], [1, 5/6])
        self.assertEqual([p.get_height() for p in figure.axes[1].patches], [100, 60, 10, 15])
        self.assertEqual([p.get_height() for p in figure.axes[2].patches], [2, 1])
        self.assertEqual(json.dumps(plans), before)

    def test_empty_results_and_missing_usage_are_not_zero_usage(self) -> None:
        plans = [{'id': 'Original', 'rows': []}, {'id': 'Rewrite', 'rows': []}]
        with patch('IPython.display.display'), redirect_stdout(io.StringIO()) as output:
            figure = lesson.compare(plans, REFERENCE)
        self.assertIn('same decisions', output.getvalue())
        self.assertEqual([p.get_height() for p in figure.axes[0].patches], [.5, .5])
        self.assertEqual(len(figure.axes[1].patches), 0)
        self.assertEqual(len(figure.axes[2].patches), 0)
        self.assertTrue(any(t.get_text() == 'Unknown' for t in figure.axes[1].texts))

    def test_zero_measurements_remain_zero_and_invalid_rows_fail(self) -> None:
        plans = [{'id': name, 'rows': CORRECT, 'execution_seconds': 0,
                  'token_usage': {'offline': {'prompt_tokens': 0, 'completion_tokens': 0}}}
                 for name in ['Original', 'Rewrite']]
        with patch('IPython.display.display'):
            figure = lesson.compare(plans, REFERENCE)
        self.assertEqual([p.get_height() for p in figure.axes[1].patches], [0, 0, 0, 0])
        self.assertEqual([p.get_height() for p in figure.axes[2].patches], [0, 0])
        plans[1]['rows'] = CORRECT + CORRECT[:1]
        with self.assertRaises(ValueError):
            lesson.compare(plans, REFERENCE)


if __name__ == '__main__':
    unittest.main()

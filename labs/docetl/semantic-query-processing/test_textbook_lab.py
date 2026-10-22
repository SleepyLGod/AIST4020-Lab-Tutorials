"""Offline checks for the active restaurant-review lab; no model calls."""
import ast
import copy
import hashlib
import io
import importlib.util
import json
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import textbook_lab_support as lesson

ROOT = Path(__file__).parent
REFERENCE = json.loads((ROOT/'textbook_data/reference.json').read_text())
CORRECT = [{'review_id': key, 'topic': value} for key, value in REFERENCE.items() if value]


class ScoringTests(unittest.TestCase):
    def test_correct(self) -> None:
        self.assertEqual(lesson.score(CORRECT, REFERENCE)['accuracy'], 1)

    def test_missing(self) -> None:
        self.assertEqual(lesson.score(CORRECT[:-1], REFERENCE)['correct'], 5)

    def test_wrongly_retained(self) -> None:
        self.assertEqual(lesson.score(CORRECT + [{'review_id': 'R1', 'topic': 'Food'}], REFERENCE)['correct'], 5)

    def test_empty_is_scored(self) -> None:
        self.assertEqual(lesson.score([], REFERENCE)['correct'], 3)

    def test_swapped_topics_do_not_hide_errors(self) -> None:
        swapped = [dict(row) for row in CORRECT]
        swapped[0]['topic'], swapped[1]['topic'] = swapped[1]['topic'], swapped[0]['topic']
        self.assertEqual(lesson.score(swapped, REFERENCE)['correct'], 4)

    def test_invalid_outputs(self) -> None:
        for rows in [None, CORRECT+CORRECT[:1], [{'review_id':'R7','topic':'Food'}],
                     [{'review_id':'R2'}], [{'review_id':'R2','topic':'Other'}]]:
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
        self.assertEqual({r['review_id'] for r in reviews}, set(REFERENCE))
        for row, source in zip(reviews, provenance['rows']):
            self.assertEqual(set(row), {'review_id', 'text'})
            self.assertEqual(hashlib.sha256(row['text'].encode()).hexdigest(), source['text_sha256'])

    def test_notebook(self) -> None:
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn("backend = 'ollama'", source)
        self.assertIn('lesson.require_colab_gpu(backend, IN_COLAB)', source)
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

        namespace = {'reviews_df': Frame(), 'MODEL_REF': saved['model'], 'OPTIONS': {}}
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        code = next(''.join(c['source']) for c in notebook['cells'] if c['id'] == 'textbook-25')
        exec(compile(code, '<visible query>', 'exec'), namespace)
        actual = namespace['optimized_query'].operations
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


class QueryTests(unittest.TestCase):
    """Check real Frame construction and template inputs without executing a model."""

    def setUp(self) -> None:
        self.notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        self.cells = {c['id']: ''.join(c['source']) for c in self.notebook['cells']}
        self.reviews = json.loads((ROOT/'textbook_data/reviews.json').read_text())

    def definitions(self) -> dict:
        """Load only the visible operator dictionaries, not their execution cells."""
        namespace = {'OPTIONS': {'temperature': 0}}
        for cell_id in ['textbook-08', 'textbook-10', 'textbook-16']:
            tree = ast.parse(self.cells[cell_id])
            exec(compile(ast.Module(body=[tree.body[0]], type_ignores=[]),
                         '<operator config>', 'exec'), namespace)
        return namespace

    def test_parts_and_removed_experiments(self) -> None:
        markdown = '\n'.join(''.join(c['source']) for c in self.notebook['cells']
                             if c['cell_type'] == 'markdown')
        headings = [line for line in markdown.splitlines() if line.startswith('## ')]
        self.assertEqual(headings, [
            '## Part 1: Load the Review Data',
            '## Part 2: Semantic Operators and Queries',
            '## Part 3: Query Optimization with MOAR',
        ])
        code = '\n'.join(''.join(c['source']) for c in self.notebook['cells']
                         if c['cell_type'] == 'code')
        for removed in ['equijoin', 'describe_issue', 'match_topics', 'prompt_query',
                        'rewritten_filter_prompt', 'doc_id']:
            self.assertNotIn(removed, code)
        self.assertNotIn('Look Inside Prompt Rewriting', markdown)
        self.assertIn('one earlier MOAR run', markdown)

    def test_templates_read_reviews_and_grouped_text(self) -> None:
        from jinja2 import Environment, StrictUndefined
        templates = Environment(undefined=StrictUndefined)
        definitions = self.definitions()
        r4 = next(r for r in self.reviews if r['review_id'] == 'R4')
        for name in ['review_filter', 'classify_map']:
            text = templates.from_string(definitions[name]['prompt']).render(input=r4)
            self.assertIn(r4['text'], text)
        grouped = [r for r in self.reviews if REFERENCE[r['review_id']] == 'Food']
        text = templates.from_string(definitions['summary_reduce']['prompt']).render(inputs=grouped)
        self.assertTrue(all(r['text'] in text for r in grouped))
        self.assertNotIn(r4['text'], text)
        self.assertEqual(definitions['summary_reduce']['reduce_key'], 'topic')

    @unittest.skipUnless(importlib.util.find_spec('docetl'), 'Requires DocETL')
    def test_real_frame_chains_share_the_same_prefix(self) -> None:
        import docetl
        import yaml
        namespace = self.definitions()
        namespace['reviews_df'] = docetl.from_list(self.reviews, name='reviews')
        with patch.object(docetl.Frame, 'collect', side_effect=AssertionError('No model execution')):
            exec(self.cells['textbook-18'], namespace)
            exec(self.cells['textbook-20'], namespace)
        full = yaml.safe_load(namespace['query'].to_yaml())
        original = yaml.safe_load(namespace['original_query'].to_yaml())
        self.assertEqual([o['type'] for o in full['operations']], ['filter', 'map', 'reduce'])
        self.assertEqual(original['operations'], full['operations'][:2])
        self.assertEqual(yaml.safe_load(namespace['reviews_df'].to_yaml())['operations'], [])
        for operation in full['operations']:
            self.assertEqual(operation['litellm_completion_kwargs'], namespace['OPTIONS'])
        self.assertEqual(full['operations'][2]['reduce_key'], 'topic')

    def test_optional_search_is_off(self) -> None:
        search = SimpleNamespace(run_search=unittest.mock.Mock(
            side_effect=AssertionError('No optimization requests')))
        exec(self.cells['textbook-28'], {'lesson': search})
        search.run_search.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('docetl'), 'Requires DocETL')
    def test_reduce_groups_original_review_text_without_model_calls(self) -> None:
        from docetl.operations.reduce import ReduceOperation
        from rich.console import Console
        config = {'type': 'reduce', **self.definitions()['summary_reduce']}
        operation = ReduceOperation(SimpleNamespace(config={}), config, 'offline/model', 1,
                                    console=Console(file=io.StringIO()))
        classified = [dict(row, topic=REFERENCE[row['review_id']]) for row in self.reviews
                      if REFERENCE[row['review_id']]]
        with patch.object(operation, '_maybe_build_retrieval_context', return_value=''), \
                patch.object(operation, '_batch_reduce', return_value=(
                    {'summary': 'Synthetic offline fixture'}, 'offline prompt', 0.0)) as reduce:
            results, _ = operation.execute(classified)
        groups = [call.args[1] for call in reduce.call_args_list]
        self.assertEqual({frozenset(r['review_id'] for r in group) for group in groups},
                         {frozenset({'R2', 'R5'}), frozenset({'R4'})})
        originals = {r['review_id']: r['text'] for r in self.reviews}
        self.assertTrue(all(r['text'] == originals[r['review_id']] for g in groups for r in g))
        self.assertEqual(len(results), 2)


class ClassificationChecks(unittest.TestCase):
    """Regression fixtures for malformed labels and incomplete map results."""

    def setUp(self) -> None:
        self.inputs = [{'review_id': 'R2', 'text': 'Offline food complaint'},
                       {'review_id': 'R4', 'text': 'Offline service complaint'}]
        self.outputs = [dict(row, topic=topic) for row, topic in
                        zip(self.inputs, ['Food', 'Service'])]

    def test_valid_reordered_output_and_empty_input(self) -> None:
        lesson.check_classifications(self.inputs, list(reversed(self.outputs)))
        lesson.check_classifications([], [])

    def test_long_labels_are_rejected_without_repair(self) -> None:
        for label in ['Food for food taste or cooking',
                      'Food for food taste or cooking, or Service for staff attention or behavior. '
                      'The complaint in this review is about the Food (taste).',
                      'food', ' Food ', '', None, ['Food']]:
            output = copy.deepcopy(self.outputs)
            output[0]['topic'] = label
            before = copy.deepcopy(output)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'Invalid topic for R2'):
                lesson.check_classifications(self.inputs, output)
            self.assertEqual(output, before)

    def test_missing_duplicate_unknown_and_malformed_records(self) -> None:
        for output in [[], self.outputs[:1], self.outputs+self.outputs[:1],
                       [dict(self.outputs[0], review_id='R9'), self.outputs[1]],
                       [{'topic': 'Food'}, self.outputs[1]],
                       [dict(self.outputs[0], review_id=[]), self.outputs[1]],
                       [dict(self.outputs[0], text='Changed'), self.outputs[1]],
                       [dict(self.inputs[0]), self.outputs[1]], None]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                lesson.check_classifications(self.inputs, output)

    def test_fused_map_checks_excluded_rows_before_code_filter(self) -> None:
        output = [dict(self.inputs[0], keep=True, topic='Food'),
                  dict(self.inputs[1], keep=False, topic='Excluded')]
        lesson.check_classifications(self.inputs, output, fused=True)
        for keep, topic in [(False, 'Food'), (True, 'Excluded'), ('false', 'Excluded')]:
            bad = [output[0], dict(self.inputs[1], keep=keep, topic=topic)]
            with self.subTest(keep=keep, topic=topic), self.assertRaises(ValueError):
                lesson.check_classifications(self.inputs, bad, fused=True)


@unittest.skipUnless(importlib.util.find_spec('docetl'), 'Requires DocETL')
class CheckedExecutionTests(unittest.TestCase):
    """Use the real runner and chains, but replace operation responses with offline fixtures."""

    def setUp(self) -> None:
        import docetl
        notebook = json.loads((ROOT/'semantic_query_processing.ipynb').read_text())
        self.cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        self.reviews = json.loads((ROOT/'textbook_data/reviews.json').read_text())
        self.kept = [row for row in self.reviews if REFERENCE[row['review_id']]]
        self.classified = [dict(row, topic=REFERENCE[row['review_id']]) for row in self.kept]
        namespace = {'OPTIONS': {}}
        for cell_id in ['textbook-08', 'textbook-10', 'textbook-16']:
            tree = ast.parse(self.cells[cell_id])
            exec(compile(ast.Module(body=[tree.body[0]], type_ignores=[]), '<config>', 'exec'), namespace)
        self.configs = namespace
        namespace['reviews_df'] = docetl.from_list(self.reviews, name='reviews')
        exec(self.cells['textbook-18'], namespace)
        exec(self.cells['textbook-20'], namespace)
        self.query, self.original = namespace['query'], namespace['original_query']

    def run_chain(self, output: list[dict], *, fresh: bool = False) -> tuple[object, object]:
        from docetl.operations.filter import FilterOperation
        from docetl.operations.map import MapOperation
        from docetl.operations.reduce import ReduceOperation
        from docetl.operations.utils.api import APIWrapper
        with patch.object(FilterOperation, 'execute', return_value=(self.kept, 0)), \
                patch.object(MapOperation, 'execute', return_value=(output, 0)), \
                patch.object(ReduceOperation, 'execute', return_value=([], 0)) as reduce, \
                patch.object(APIWrapper, 'call_llm', side_effect=AssertionError('No model calls')), \
                redirect_stdout(io.StringIO()):
            try:
                if fresh:
                    directory = Path(tempfile.mkdtemp(prefix='docetl-checked-offline-'))
                    result = lesson.collect_fresh(self.original, self.reviews, 'offline/model', directory)
                else:
                    result = lesson.collect_checked(self.query)
            except ValueError:
                reduce.assert_not_called()
                raise
        return result, reduce

    def test_invalid_map_stops_real_chain_before_reduce(self) -> None:
        from docetl.runner import DSLRunner
        original_runner = DSLRunner._run_operation
        cases = [[], self.classified[:-1], self.classified+self.classified[:1],
                 [dict(self.classified[0], topic='Food for food taste or cooking'),
                  *self.classified[1:]]]
        for output in cases:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.run_chain(output)
            self.assertIs(DSLRunner._run_operation, original_runner)

    def test_valid_map_reaches_reduce_with_identical_rows(self) -> None:
        result, reduce = self.run_chain(self.classified)
        self.assertEqual(result, [])
        self.assertEqual(reduce.call_args.args[0], self.classified)

    def test_unchecked_frame_memo_cannot_bypass_validation(self) -> None:
        key = self.query._memo_key(self.query._build_config(checkpoint=False))
        self.query._memo = (key, [{'summary': 'Unchecked cached result'}], 0, {})
        with self.assertRaisesRegex(ValueError, 'missing='):
            self.run_chain([])

    def test_comparison_collection_rejects_missing_classifications(self) -> None:
        with self.assertRaisesRegex(ValueError, 'missing='):
            self.run_chain(self.classified[:-1], fresh=True)

    def test_standalone_reduce_rejects_long_labels(self) -> None:
        import docetl
        from docetl.operations.reduce import ReduceOperation
        bad = [dict(self.classified[0], topic='Food for food taste or cooking')]
        query = docetl.from_list(bad).reduce(**self.configs['summary_reduce'])
        with patch.object(ReduceOperation, 'execute') as reduce, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'Invalid topic'):
                lesson.collect_checked(query)
        reduce.assert_not_called()

    def test_notebook_uses_checked_collection_in_all_aggregation_paths(self) -> None:
        for cell_id in ['textbook-10', 'textbook-16', 'textbook-18']:
            tree = ast.parse(self.cells[cell_id])
            calls = [n.func.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            self.assertIn('collect_checked', calls)
            self.assertNotIn('collect', calls)


class SetupTests(unittest.TestCase):
    def test_no_gpu_probe_for_api_or_local_execution(self) -> None:
        with patch('textbook_lab_support.subprocess.run') as run:
            lesson.require_colab_gpu('deepseek', True)
            lesson.require_colab_gpu('ollama', False)
            lesson.require_colab_gpu('custom', True)
        run.assert_not_called()

    def test_colab_gpu_available(self) -> None:
        with patch('textbook_lab_support.subprocess.run',
                   return_value=SimpleNamespace(stdout='Tesla T4\n')) as run:
            lesson.require_colab_gpu('ollama', True)
        self.assertEqual(run.call_args.args[0][0], 'nvidia-smi')

    def test_colab_gpu_missing_or_unavailable(self) -> None:
        for error in [FileNotFoundError(), subprocess.CalledProcessError(1, 'nvidia-smi'),
                      subprocess.TimeoutExpired('nvidia-smi', 10)]:
            with self.subTest(error=type(error).__name__), \
                    patch('textbook_lab_support.subprocess.run', side_effect=error), \
                    self.assertRaisesRegex(RuntimeError, 'T4 GPU'):
                lesson.require_colab_gpu('ollama', True)
        with patch('textbook_lab_support.subprocess.run',
                   return_value=SimpleNamespace(stdout='')), \
                self.assertRaisesRegex(RuntimeError, 'T4 GPU'):
            lesson.require_colab_gpu('ollama', True)


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

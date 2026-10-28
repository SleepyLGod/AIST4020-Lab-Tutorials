"""Offline checks for the native DocETL review lab; no model requests."""
import ast
import base64
import copy
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterator
import unittest
from unittest.mock import patch
import zipfile

import docetl
import nbformat
import pandas as pd
import yaml

import textbook_lab_support as lesson

ROOT = Path(__file__).parent
NOTEBOOK = json.loads((ROOT / 'semantic_query_processing.ipynb').read_text())
CELLS = {c['id']: ''.join(c['source']) for c in NOTEBOOK['cells']}
REVIEWS = json.loads((ROOT / 'textbook_data/reviews.json').read_text())
REFERENCE = json.loads((ROOT / 'textbook_data/reference.json').read_text())
CORRECT = [dict(row, topic=REFERENCE[row['review_id']]) for row in REVIEWS
           if REFERENCE[row['review_id']]]


def definitions() -> dict:
    """Load visible definitions without executing any queries."""
    namespace = {'OPTIONS': {}, 'MODEL_REF': 'offline/model', 'pd': pd}
    for cell_id in ['textbook-08', 'textbook-10', 'textbook-16', 'textbook-25', 'textbook-22']:
        tree = ast.parse(CELLS[cell_id])
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<definitions>', 'exec'), namespace)
    return namespace


def assigned_query(cell_id: str, name: str, namespace: dict) -> object:
    """Build one public Frame expression without running its terminal action."""
    node = next(node for node in ast.parse(CELLS[cell_id]).body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<query>', 'exec'), namespace)
    return namespace[name]


@contextmanager
def fixture_directory() -> Iterator[str]:
    """Keep temporary test artifacts available for inspection."""
    yield tempfile.mkdtemp(prefix='docetl-native-offline-')


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.score = definitions()['score_reviews']

    def test_correct(self) -> None:
        self.assertEqual(self.score(CORRECT, REFERENCE)['correct'].sum(), 6)

    def test_missing_and_wrongly_retained(self) -> None:
        self.assertEqual(self.score(CORRECT[:-1], REFERENCE)['correct'].sum(), 5)
        self.assertEqual(self.score(CORRECT + [{'review_id': 'R1', 'topic': 'Food'}],
                                    REFERENCE)['correct'].sum(), 5)

    def test_empty_is_scored(self) -> None:
        self.assertEqual(self.score([], REFERENCE)['correct'].sum(), 3)

    def test_swapped_topics_do_not_hide_errors(self) -> None:
        swapped = copy.deepcopy(CORRECT)
        swapped[0]['topic'], swapped[1]['topic'] = swapped[1]['topic'], swapped[0]['topic']
        self.assertEqual(self.score(swapped, REFERENCE)['correct'].sum(), 4)

    def test_long_or_missing_labels_are_wrong_not_repaired(self) -> None:
        for label in ['Food for food taste or cooking', 'food', '', None, ['Food']]:
            rows = copy.deepcopy(CORRECT)
            rows[0]['topic'] = label
            before = copy.deepcopy(rows)
            self.assertEqual(self.score(rows, REFERENCE)['correct'].sum(), 5)
            self.assertEqual(rows, before)
        rows = copy.deepcopy(CORRECT)
        rows[0].pop('topic')
        self.assertEqual(self.score(rows, REFERENCE)['correct'].sum(), 5)

    def test_retained_excluded_or_null_label_is_not_exclusion(self) -> None:
        for label in ['Excluded', None]:
            rows = CORRECT + [{'review_id': 'R1', 'topic': label}]
            result = self.score(rows, REFERENCE).set_index('review_id')
            self.assertFalse(result.loc['R1', 'correct'])

    def test_duplicate_and_unknown_ids_do_not_get_silently_indexed(self) -> None:
        for rows in [CORRECT + CORRECT[:1], [{'review_id': 'R7', 'topic': 'Food'}]]:
            with self.assertRaises(ValueError):
                self.score(rows, REFERENCE)


class ArtifactTests(unittest.TestCase):
    def test_notebook_is_valid_and_unexecuted(self) -> None:
        nbformat.validate(nbformat.from_dict(NOTEBOOK))
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        self.assertIn("backend = 'ollama'", CELLS['textbook-05'])
        self.assertIn('RUN_MOAR = False', CELLS['textbook-28'])

    def test_three_parts_and_native_main_path(self) -> None:
        markdown = '\n'.join(''.join(c['source']) for c in NOTEBOOK['cells']
                             if c['cell_type'] == 'markdown')
        self.assertEqual([line for line in markdown.splitlines() if line.startswith('## ')], [
            '## Part 1: Load the Review Data',
            '## Part 2: Semantic Operators and Queries',
            '## Part 3: Query Optimization with MOAR',
        ])
        code = '\n'.join(''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code')
        for removed in ['collect_checked', 'collect_fresh', 'run_search(', 'show_query_plans(',
                        'lesson.score', 'lesson.compare', 'equijoin', '_execute(', '_run_operation']:
            self.assertNotIn(removed, code)
        self.assertNotIn('RUN_COMPLETE_QUERY', code)
        self.assertIn('results = query.collect()', CELLS['textbook-18'])
        self.assertIn('removes this field', CELLS['textbook-09'])

    def test_source_records(self) -> None:
        provenance = json.loads((ROOT / 'textbook_data/provenance.json').read_text())
        self.assertEqual(len(REVIEWS), 6)
        self.assertEqual({r['review_id'] for r in REVIEWS}, set(REFERENCE))
        for row, source in zip(REVIEWS, provenance['rows']):
            self.assertEqual(set(row), {'review_id', 'text'})
            self.assertEqual(hashlib.sha256(row['text'].encode()).hexdigest(), source['text_sha256'])

    def test_bundle_and_pinned_publication(self) -> None:
        bundle = ROOT / 'textbook_lab_setup.zip'
        self.assertIn(hashlib.sha256(bundle.read_bytes()).hexdigest(), CELLS['textbook-02'])
        with zipfile.ZipFile(bundle) as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
        self.assertRegex(
            CELLS['textbook-02'],
            r"COURSE_BUNDLE_URL = 'https://raw\.githubusercontent\.com/"
            r"SleepyLGod/AIST4020-Lab-Tutorials/[0-9a-f]{40}/"
            r"labs/docetl/semantic-query-processing/textbook_lab_setup\.zip'",
        )

    def test_static_image_embedded_and_exportable(self) -> None:
        from nbconvert import HTMLExporter
        cell = next(c for c in NOTEBOOK['cells'] if c['id'] == 'query-plan-diagram')
        raw = base64.b64decode(cell['attachments']['query-plans.png']['image/png'])
        self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))
        rendered, _ = HTMLExporter().from_notebook_node(nbformat.reads(json.dumps(NOTEBOOK), as_version=4))
        self.assertIn('data:image/png;base64,', rendered)
        self.assertNotIn('src="attachment:', rendered)

    def test_actual_saved_query_and_current_logic(self) -> None:
        saved = json.loads((ROOT / 'textbook_saved_run/lesson.json').read_text())
        search = json.loads((ROOT / 'textbook_saved_run/search.json').read_text())
        self.assertEqual(saved['provenance'], 'real_colab_deepseek_run')
        self.assertTrue(any(p.get('config', {}).get('operations') == saved['operations']
                            for p in search['plans']))
        definitions_ = definitions()
        definitions_['fused_map']['model'] = saved['model']
        expected = [{k: v for k, v in op.items() if k != 'name'} for op in saved['operations']]
        expected[0]['prompt'] = expected[0]['prompt'].replace(
            'return None for the topic.', 'return the string Excluded for the topic.'
        ).replace('(Food, Service, or None).', '(Food, Service, or Excluded).')
        actual = [
            {'type': 'map', **definitions_['fused_map']},
            {'type': 'code_filter', **definitions_['keep_filter']},
        ]
        self.assertEqual([{k: v for k, v in op.items() if k != 'name'} for op in actual], expected)


class NativeQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = definitions()
        self.ns.update(docetl=docetl, reviews=REVIEWS,
                       reviews_df=docetl.from_list(REVIEWS, name='reviews'))
        self.output = io.StringIO()
        self.calls_blocked = patch('docetl.operations.utils.api.APIWrapper.call_llm',
                                   side_effect=AssertionError('No model requests in offline tests'))
        self.calls_blocked.start()
        self.addCleanup(self.calls_blocked.stop)

    def test_chains_share_the_same_filter_map_prefix(self) -> None:
        full = assigned_query('textbook-18', 'query', self.ns)
        original = assigned_query('textbook-20', 'original_query', self.ns)
        full_ops = yaml.safe_load(full.to_yaml())['operations']
        original_ops = yaml.safe_load(original.to_yaml())['operations']
        self.assertEqual([op['type'] for op in full_ops], ['filter', 'map', 'reduce'])
        self.assertEqual(original_ops, full_ops[:2])
        self.assertEqual(full_ops[-1]['reduce_key'], 'topic')

    def test_filter_consumes_the_boolean_decision(self) -> None:
        from docetl.operations.filter import FilterOperation
        operation = FilterOperation.__new__(FilterOperation)
        operation._filter_key = 'keep'
        operation._filter_is_build = False
        kept, decision = operation._handle_result({'review_id': 'R2', 'keep': True})
        self.assertTrue(decision)
        self.assertEqual(kept, {'review_id': 'R2'})
        kept, decision = operation._handle_result({'review_id': 'R1', 'keep': False})
        self.assertFalse(decision)
        self.assertIsNone(kept)

    def test_templates_read_review_and_group(self) -> None:
        from jinja2 import Environment, StrictUndefined
        templates = Environment(undefined=StrictUndefined)
        r4 = REVIEWS[3]
        for name in ['review_filter', 'classify_map']:
            prompt = templates.from_string(self.ns[name]['prompt']).render(input=r4)
            self.assertIn(r4['text'], prompt)
        group = [r for r in CORRECT if r['topic'] == 'Food']
        prompt = templates.from_string(self.ns['summary_reduce']['prompt']).render(inputs=group)
        self.assertTrue(all(r['text'] in prompt for r in group))
        self.assertNotIn(r4['text'], prompt)

    def test_native_complete_query_does_not_repair_long_labels(self) -> None:
        from docetl.operations.filter import FilterOperation
        from docetl.operations.map import MapOperation
        from docetl.operations.reduce import ReduceOperation
        rows = copy.deepcopy(CORRECT)
        rows[0]['topic'] = 'Food for food taste or cooking'
        with patch.object(FilterOperation, 'execute', return_value=(CORRECT, 0)), \
                patch.object(MapOperation, 'execute', return_value=(rows, 0)), \
                patch.object(ReduceOperation, 'execute', return_value=([], 0)) as reduce, \
                redirect_stdout(self.output):
            exec(CELLS['textbook-18'], self.ns)
        self.assertEqual(reduce.call_args.args[0], rows)

    def test_fusion_walkthrough_preserves_keep_and_uses_real_code_filter(self) -> None:
        from docetl.operations.map import MapOperation
        synthetic = [dict(row, keep=REFERENCE[row['review_id']] is not None,
                          topic=REFERENCE[row['review_id']] or 'Excluded') for row in REVIEWS]
        with patch.object(MapOperation, 'execute', return_value=(synthetic, 0)) as model, \
                redirect_stdout(self.output):
            exec(CELLS['textbook-25'], self.ns)
            exec(CELLS['fused-filter-code'], self.ns)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(len(self.ns['fused_reviews']), 6)
        self.assertEqual({r['review_id'] for r in self.ns['selected_reviews']}, {'R2', 'R4', 'R5'})
        self.assertTrue(all(r['keep'] is True for r in self.ns['selected_reviews']))

    def test_reduce_groups_original_text(self) -> None:
        from docetl.operations.reduce import ReduceOperation
        from rich.console import Console
        operation = ReduceOperation(SimpleNamespace(config={}),
                                    {'type': 'reduce', **self.ns['summary_reduce']},
                                    'offline/model', 1, console=Console(file=io.StringIO()))
        with patch.object(operation, '_maybe_build_retrieval_context', return_value=''), \
                patch.object(operation, '_batch_reduce', return_value=(
                    {'summary': 'Synthetic offline fixture'}, 'offline prompt', 0.0)) as reducer:
            results, _ = operation.execute(CORRECT)
        groups = [call.args[1] for call in reducer.call_args_list]
        self.assertEqual({frozenset(r['review_id'] for r in g) for g in groups},
                         {frozenset({'R2', 'R5'}), frozenset({'R4'})})
        self.assertEqual(len(results), 2)

    def test_comparison_rebuilds_frames_and_times_only_collect(self) -> None:
        observed = []
        clock = iter([10., 12., 20., 21.])

        def collect(frame: object, **kwargs: object) -> list[dict]:
            config = yaml.safe_load(frame.to_yaml())
            self.assertTrue(config['bypass_cache'])
            self.assertNotIn('intermediate_dir', config['pipeline'])
            observed.append(frame)
            frame._total_cost = 0
            frame._token_usage = {'offline': {'prompt_tokens': 100, 'completion_tokens': 10}}
            return copy.deepcopy(CORRECT)

        self.ns.update(reference=REFERENCE, display=lambda value: None)
        with patch.object(docetl.Frame, 'collect', collect), \
                patch('time.perf_counter', side_effect=lambda: next(clock)), \
                patch.object(lesson, 'plot_comparison') as plot, redirect_stdout(self.output):
            exec(CELLS['textbook-20'], self.ns)
            exec(CELLS['textbook-22'], self.ns)
            exec(CELLS['textbook-26'], self.ns)
            self.ns['lesson'] = lesson
            exec(CELLS['query-measurements'], self.ns)
        self.assertEqual(len(observed), 2)
        self.assertIsNot(observed[0], observed[1])
        self.assertEqual([m['seconds'] for m in self.ns['metrics']], [2., 1.])
        self.assertEqual([m['input_tokens'] for m in self.ns['metrics']], [100, 100])
        self.assertTrue(all(m['estimated_usd'] is None for m in self.ns['metrics']))
        self.assertEqual(plot.call_args.args[0], self.ns['metrics'])

    def test_missing_token_field_is_unknown_not_partial_total(self) -> None:
        scored = definitions()['score_reviews'](CORRECT, REFERENCE)
        frame = SimpleNamespace(total_cost=0, token_usage={
            'a': {'prompt_tokens': 10, 'completion_tokens': 4},
            'b': {'completion_tokens': 3},
        })
        ns = dict(pd=pd, original_query=frame, optimized_query=frame,
                  original_decisions=scored, optimized_decisions=scored,
                  original_seconds=1., optimized_seconds=2., display=lambda value: None,
                  lesson=SimpleNamespace(plot_comparison=lambda values: None))
        exec(CELLS['query-measurements'], ns)
        self.assertIsNone(ns['metrics'][0]['input_tokens'])
        self.assertEqual(ns['metrics'][0]['output_tokens'], 7)


class OptimizationTests(unittest.TestCase):
    def test_optional_off_never_enters_context(self) -> None:
        context = unittest.mock.Mock(side_effect=AssertionError('Search must remain off'))
        exec(CELLS['textbook-28'], {'lesson': SimpleNamespace(optimization_environment=context)})
        context.assert_not_called()

    def test_visible_optimize_receives_visible_evaluator(self) -> None:
        with fixture_directory() as folder:
            path = Path(folder) / 'output.json'
            path.write_text(json.dumps(CORRECT))
            query = SimpleNamespace(optimize=unittest.mock.Mock(
                return_value=SimpleNamespace(to_yaml=lambda: 'offline result')))

            @contextmanager
            def environment(*args: object) -> Iterator[dict[str, str]]:
                yield {'query_path': 'offline.yaml', 'dataset_path': 'inputs.json', 'save_dir': folder}

            ns = definitions()
            ns.update(Path=Path, json=json, reference=REFERENCE,
                      original_query=object(), reviews=REVIEWS, RUN_DIR=Path(folder),
                      lesson=SimpleNamespace(optimization_environment=environment),
                      docetl=SimpleNamespace(Frame=SimpleNamespace(from_yaml=lambda path: query)))
            with redirect_stdout(io.StringIO()):
                exec(CELLS['textbook-28'].replace('RUN_MOAR = False', 'RUN_MOAR = True'), ns)
            args = query.optimize.call_args.kwargs
            self.assertEqual(args['models'], ['offline/model'])
            self.assertEqual(args['agent_model'], 'offline/model')
            self.assertEqual(args['max_iterations'], 2)
            self.assertEqual(args['max_concurrent_agents'], 1)
            self.assertEqual(args['max_threads'], 1)
            self.assertEqual(args['eval_fn'](str(path)), {'accuracy': 1.})
            path.write_text(json.dumps(CORRECT[:2]))
            self.assertEqual(args['eval_fn'](str(path)), {'accuracy': 5/6})

    def test_context_does_not_run_search_and_restores_guards(self) -> None:
        from docetl.moar.Node import Node
        from docetl.moar import optimizer
        ns = definitions()
        query = docetl.from_list(REVIEWS).filter(**ns['review_filter']).map(**ns['classify_map'])
        execute = Node.execute_plan
        directives = optimizer.ALL_DIRECTIVES
        with fixture_directory() as folder, \
                patch.dict(sys.modules, {'google.colab': SimpleNamespace()}), \
                patch.object(Node, 'execute_plan', return_value=0.) as run:
            with lesson.optimization_environment(query, REVIEWS, 'offline/model', Path(folder)) as files:
                config = yaml.safe_load(Path(files['query_path']).read_text())
                self.assertEqual(json.loads(Path(files['dataset_path']).read_text()), REVIEWS)
                self.assertEqual({d.name for d in optimizer.ALL_DIRECTIVES},
                                 {'clarify_instructions', 'operator_fusion'})
                run.assert_not_called()
                bad = copy.deepcopy(config)
                bad['operations'][0]['model'] = 'openai/not-allowed'
                with self.assertRaises(ValueError):
                    Node.execute_plan(SimpleNamespace(parsed_yaml=bad))
                run.assert_not_called()
            records = list(Path(folder).glob('search/*/record.json'))
            record = json.loads(records[0].read_text())
            self.assertIn('not-allowed', record['plans'][0]['error'])
        self.assertIs(Node.execute_plan, execute)
        self.assertIs(optimizer.ALL_DIRECTIVES, directives)

    def test_context_failure_is_saved_and_raised(self) -> None:
        ns = definitions()
        query = docetl.from_list(REVIEWS).filter(**ns['review_filter'])
        with fixture_directory() as folder, \
                patch.dict(sys.modules, {'google.colab': SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, 'offline failure'):
                with lesson.optimization_environment(query, REVIEWS, 'offline/model', Path(folder)):
                    raise RuntimeError('offline failure')
            record = json.loads(next(Path(folder).glob('search/*/record.json')).read_text())
            self.assertEqual(record['status'], 'failed')


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


class PlotTests(unittest.TestCase):
    def tearDown(self) -> None:
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_plots_use_supplied_scores_and_measurements(self) -> None:
        metrics = [
            {'query': 'Original', 'accuracy': 1., 'input_tokens': 100, 'output_tokens': 10, 'seconds': 2.},
            {'query': 'Rewrite', 'accuracy': 5/6, 'input_tokens': 60, 'output_tokens': 15, 'seconds': 1.},
        ]
        before = copy.deepcopy(metrics)
        figure = lesson.plot_comparison(metrics)
        self.assertEqual([p.get_height() for p in figure.axes[0].patches], [1., 5/6])
        self.assertEqual([p.get_height() for p in figure.axes[1].patches], [100, 10, 60, 15])
        self.assertEqual([p.get_height() for p in figure.axes[2].patches], [2., 1.])
        self.assertEqual(metrics, before)
        figure.canvas.draw()
        for axis in figure.axes:
            for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.get_xticklabels()]:
                bounds = text.get_window_extent(figure.canvas.get_renderer())
                self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0))
                self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1))

    def test_unknown_values_are_not_zero(self) -> None:
        figure = lesson.plot_comparison([
            {'query': 'Original', 'accuracy': .5},
            {'query': 'Rewrite', 'accuracy': .5, 'input_tokens': 0, 'seconds': 0},
        ])
        self.assertEqual(len(figure.axes[1].patches), 1)
        self.assertEqual(figure.axes[1].patches[0].get_height(), 0)
        self.assertTrue(any(t.get_text() == 'Unknown' for t in figure.axes[1].texts))


if __name__ == '__main__':
    unittest.main()

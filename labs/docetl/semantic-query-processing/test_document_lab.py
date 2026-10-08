"""Offline checks for the six-document notebook; never call a model."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import unittest
import zipfile
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat
import document_lab_support as support

ROOT = Path(__file__).resolve().parent
NB = nbformat.read(ROOT / 'semantic_query_processing.ipynb', as_version=4)


def cell(tag: str) -> str:
    """Find a named code cell."""
    return next(item.source for item in NB.cells if tag in item.metadata.get('tags', []))


class DocumentLabChecks(unittest.TestCase):
    """Check the real plan construction and local displays, not model behavior."""

    def test_baseline_measurements_do_not_add_calls(self) -> None:
        usage = {'offline/model': {'prompt_tokens': 120, 'completion_tokens': 12}}
        frame = SimpleNamespace(collect=Mock(return_value=[{'doc_id': 'offline'}]),
                                _memo=None, total_cost=.012, token_usage=usage)
        with patch.object(support.time, 'perf_counter', side_effect=[10.0, 12.5]):
            rows, work = support.collect_measured(frame)
        frame.collect.assert_called_once_with()
        self.assertEqual(rows, [{'doc_id': 'offline'}])
        self.assertEqual(work['execution_seconds'], 2.5)
        self.assertEqual(work['execution_cost_estimate_usd'], .012)
        self.assertEqual(work['token_usage'], usage)
        usage['offline/model']['prompt_tokens'] = 999
        self.assertEqual(work['token_usage']['offline/model']['prompt_tokens'], 120)
        with patch.object(support, 'show_rows') as table:
            support.show_query_work([{'id': 'original', **work}, {'id': 'missing'}])
        shown = table.call_args.args[0]
        self.assertEqual(shown[0]['reported_input_tokens'], 120)
        self.assertEqual(shown[1]['reported_input_tokens'], 'Unknown')
        frame._memo = ('offline cached execution',)
        _, cached = support.collect_measured(frame)
        self.assertIsNone(cached['token_usage'])
        self.assertIsNone(cached['execution_cost_estimate_usd'])
        frame._memo, frame.total_cost = None, 0
        _, zero = support.collect_measured(frame)
        self.assertIsNone(zero['execution_cost_estimate_usd'])
        frame.collect.side_effect = RuntimeError('offline execution failed')
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            support.collect_measured(frame)

    def test_bundled_reference_opens_without_web_links(self) -> None:
        with patch('IPython.display.display') as display:
            support.show_scoring_reference(ROOT)
        self.assertEqual(display.call_count, 2)
        self.assertIn('Reference decisions', display.call_args_list[0].args[0].data)
        self.assertIn('R05', display.call_args_list[1].args[0].data)
        with patch.object(Path, 'read_text', return_value='<script>offline</script>'), \
                patch('IPython.display.display') as display:
            support.show_scoring_reference(ROOT)
        self.assertNotIn('<script>', display.call_args.args[0].data)
        self.assertIn('&lt;script&gt;', display.call_args.args[0].data)

    def test_optimizer_context_routes_and_restores(self) -> None:
        import litellm
        import sys
        fake = Mock(return_value='offline response')
        alias = SimpleNamespace(completion=fake)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(litellm, 'completion', fake), \
                patch.dict(sys.modules, {'docetl.offline_alias': alias}):
            with support.optimizer_context('ollama_chat/test', Path(folder)):
                for call in (litellm.completion, alias.completion):
                    self.assertEqual(call(model='ollama_chat/test', messages=[]), 'offline response')
                    self.assertEqual(fake.call_args.kwargs['num_ctx'], 16384)
                    self.assertEqual(fake.call_args.kwargs['max_tokens'], 2048)
                    self.assertFalse(fake.call_args.kwargs['think'])
                litellm.completion(model='deepseek/test', messages=[])
                self.assertNotIn('num_ctx', fake.call_args.kwargs)
            self.assertIs(alias.completion, fake)
            self.assertIs(litellm.completion, fake)
            with support.optimizer_context('deepseek/test', Path(folder)):
                self.assertIs(litellm.completion, fake)

    def test_optimizer_options_reach_ollama_payload(self) -> None:
        import litellm
        from litellm.utils import get_optional_params
        fake = Mock(return_value='offline response')
        with tempfile.TemporaryDirectory() as folder, patch.object(litellm, 'completion', fake):
            with support.optimizer_context('ollama_chat/qwen3.5:9b-q4_K_M', Path(folder)):
                litellm.completion(model='ollama_chat/qwen3.5:9b-q4_K_M')
        options = dict(fake.call_args.kwargs)
        options['model'] = 'qwen3.5:9b-q4_K_M'
        payload = get_optional_params(custom_llm_provider='ollama_chat', **options)
        self.assertEqual(payload['num_ctx'], 16384)
        self.assertEqual(payload['num_predict'], 2048)
        self.assertFalse(payload['think'])

    def test_optimizer_truncation_stops_even_if_caught_by_search(self) -> None:
        import litellm
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'ollama-server.log'
            log.write_text('old truncating input prompt\n')

            def request(**kwargs: object) -> str:
                with log.open('a') as output:
                    output.write('new truncating input prompt\n')
                return 'invalid response'

            fake = Mock(side_effect=request)
            with patch.object(litellm, 'completion', fake):
                with self.assertRaisesRegex(RuntimeError, 'did not complete safely'):
                    with support.optimizer_context('ollama_chat/test', Path(folder)):
                        with self.assertRaisesRegex(RuntimeError, 'response is invalid'):
                            litellm.completion(model='ollama_chat/test')
                        with self.assertRaisesRegex(RuntimeError, 'stop this run'):
                            litellm.completion(model='ollama_chat/test')
                self.assertEqual(fake.call_count, 1)
                self.assertIs(litellm.completion, fake)

    def test_notebook_structure_and_syntax(self) -> None:
        nbformat.validate(NB)
        for item in NB.cells:
            if item.cell_type == 'code':
                ast.parse(item.source)
                self.assertEqual(item.outputs, [])
                self.assertIsNone(item.execution_count)
        text = '\n'.join(item.source for item in NB.cells)
        self.assertNotIn('/Users/', text)
        self.assertNotIn('frame.optimize(', text)
        self.assertIn('ENABLE_MODEL_CALLS = False', cell('backend'))
        self.assertIn('RUN_COMPLETE_QUERY = False', cell('complete_run'))
        self.assertIn('RUN_MOAR = False', cell('document_moar'))
        self.assertIn('ALLOW_LOCAL_EMBEDDINGS = True', cell('document_moar'))

    def test_fixed_inputs_and_bundle(self) -> None:
        docs, sectors = support.load_data(ROOT / 'document_data')
        self.assertEqual(len(docs), 6)
        self.assertEqual(len(sectors), 2)
        self.assertTrue(all(set(row) == {'doc_id', 'text'} for row in docs))
        blob = (ROOT / 'document_lab_setup.zip').read_bytes()
        self.assertIn(hashlib.sha256(blob).hexdigest(), cell('load_data'))
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                self.assertNotIn('..', Path(name).parts)
                self.assertFalse(Path(name).is_absolute())
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())

    def test_disabled_run(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'ENABLE_MODEL_CALLS'):
            support.require_enabled(False)
        support.require_enabled(True)
        with redirect_stdout(io.StringIO()), patch.object(support, 'require_enabled') as gate:
            exec(cell('complete_run'), {})
            gate.assert_not_called()

    def test_row_validation(self) -> None:
        inputs = [{'doc_id': 'x', 'text': 'offline fixture'}]
        good = [{**inputs[0], 'company': 'C', 'business': 'B', 'actions': ['a']}]
        support.check_rows(inputs, good, complete=True)
        support.check_rows(inputs, [], complete=False)
        for bad in [[], good + good, [{**good[0], 'doc_id': 'unknown'}],
                    [{**good[0], 'text': 'changed'}], [{**good[0], 'actions': 'not a list'}]]:
            with self.assertRaises(ValueError):
                support.check_rows(inputs, bad, complete=True)

    def test_empty_and_multiple_join_results(self) -> None:
        facts = [{'doc_id': 'x'}]
        sectors = [{'sector': 'A'}, {'sector': 'B'}]
        for matches in [[], [{'doc_id': 'x', 'sector': 'A'}, {'doc_id': 'x', 'sector': 'B'}]]:
            output = io.StringIO()
            with redirect_stdout(output):
                support.inspect_matches(facts, sectors, matches)
            self.assertIn('closer look', output.getvalue())
        for bad in [[{'doc_id': 'z', 'sector': 'A'}], [{'doc_id': 'x', 'sector': 'C'}],
                    [{'doc_id': 'x', 'sector': 'A'}] * 2]:
            with self.assertRaises(ValueError):
                support.inspect_matches(facts, sectors, bad)

    def test_summary_group_coverage(self) -> None:
        support.check_summaries([], [])
        support.check_summaries([{'sector': 'A'}], [{'sector': 'A', 'summary': 'Text'}])
        for bad in [[], [{'sector': 'B', 'summary': 'Text'}], [{'sector': 'A', 'summary': ''}]]:
            with self.assertRaises(ValueError):
                support.check_summaries([{'sector': 'A'}], bad)

    def test_html_escaping(self) -> None:
        text = support.table_html([{'text': '<script>alert(1)</script>'}], ['text'])
        self.assertNotIn('<script>', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertIn('<table ', support.table_html([], ['text']))

    def test_company_counts_use_matched_documents(self) -> None:
        cases = [
            ([], [], []),
            ([{'doc_id': 'x', 'sector': 'A'}, {'doc_id': 'y', 'sector': 'A'},
              {'doc_id': 'z', 'sector': 'B'}],
             [{'sector': 'B', 'summary': 'Offline B'}, {'sector': 'A', 'summary': 'Offline A'}],
             [1, 2]),
            ([{'doc_id': 'x', 'sector': 'A'}, {'doc_id': 'x', 'sector': 'B'}],
             [{'sector': 'A', 'summary': 'Offline A'}, {'sector': 'B', 'summary': 'Offline B'}],
             [1, 1]),
        ]
        for matches, summaries, counts in cases:
            with self.subTest(matches=matches):
                saved = [{**row, 'summarize_by_sector_lineage': [
                    {'doc_id': item['doc_id']} for item in matches if item['sector'] == row['sector']
                ]} for row in summaries]
                rows = support.add_company_counts(saved)
                self.assertEqual([row['company_count'] for row in rows], counts)
                self.assertEqual([{k: v for k, v in row.items() if k != 'company_count'}
                                  for row in rows], saved)
                self.assertTrue(all('company_count' not in row for row in saved))

    def test_counts_reject_missing_or_duplicate_sources(self) -> None:
        row = {'sector': 'A', 'summary': 'Offline fixture'}
        for lineage in [None, [], [{}], [{'doc_id': ''}], ['x'],
                        [{'doc_id': 'x'}, {'doc_id': 'x'}]]:
            with self.subTest(lineage=lineage), self.assertRaises(ValueError):
                support.add_company_counts([{**row, 'summarize_by_sector_lineage': lineage}])
        valid = {**row, 'summarize_by_sector_lineage': [{'doc_id': 'x'}]}
        for invalid in [[valid, valid], [{**valid, 'summary': ''}], [{**valid, 'sector': None}]]:
            with self.assertRaises(ValueError):
                support.add_company_counts(invalid)

    def test_complete_display_uses_its_own_run(self) -> None:
        current = [{'sector': 'A', 'summary': 'Offline new result',
                    'summarize_by_sector_lineage': [{'doc_id': 'new'}]}]
        query = Mock()
        query.collect.return_value = current
        namespace = {'support': support, 'ENABLE_MODEL_CALLS': True, 'query': query,
                     'matched_rows': [{'doc_id': 'old', 'sector': 'B'}],
                     'company_counts': {'A': 99}}
        with patch.object(support, 'show_rows') as show:
            exec(cell('complete_run').replace('RUN_COMPLETE_QUERY = False',
                                             'RUN_COMPLETE_QUERY = True', 1), namespace)
        query.collect.assert_called_once_with()
        self.assertEqual(namespace['complete_analysis_rows'][0]['company_count'], 1)
        show.assert_called_once_with(namespace['complete_analysis_rows'],
                                     ['sector', 'company_count', 'summary'])

    def test_real_reduce_saves_source_ids_without_model_call(self) -> None:
        from docetl.operations.reduce import ReduceOperation
        from rich.console import Console
        config = {
            'name': 'summarize_by_sector', 'type': 'reduce', 'reduce_key': 'sector',
            'prompt': 'Summarize {{ inputs }}',
            'output': {'schema': {'summary': 'string'}, 'lineage': ['doc_id']},
            'synthesize_resolve': False,
        }
        operation = ReduceOperation(SimpleNamespace(config={}), config, 'offline/model', 1,
                                    console=Console(file=io.StringIO()))
        matches = [{'doc_id': 'x', 'sector': 'A'}, {'doc_id': 'y', 'sector': 'A'}]
        with patch.object(operation, '_maybe_build_retrieval_context', return_value=''), \
                patch.object(operation, '_batch_reduce', return_value=(
                    {'sector': 'A', 'summary': 'Offline fixture'}, 'offline prompt', 0.0)):
            rows, _ = operation.execute(matches)
        self.assertEqual(rows[0]['summarize_by_sector_lineage'], [{'doc_id': 'x'}, {'doc_id': 'y'}])
        self.assertEqual(support.add_company_counts(rows)[0]['company_count'], 2)

    def test_bundle_download_and_failure_paths(self) -> None:
        import urllib.error
        source = cell('load_data').split("COURSE_DIR =", 1)[0]
        self.assertNotIn('files.upload', source)
        self.assertNotIn('Upload document_lab_setup.zip', source)
        self.assertIn('https://raw.githubusercontent.com/', source)
        # Exercise download failures without making network requests.
        parsed = ast.parse(source)
        for node in parsed.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'COURSE_BUNDLE_URL'
                    for target in node.targets):
                node.value = ast.Constant('https://example.invalid/course.zip')
        source = ast.unparse(parsed)
        payload = (ROOT / 'document_lab_setup.zip').read_bytes()
        for cached, result, error in [
            (True, payload, None), (False, payload, None),
            (False, b'bad zip', ValueError), (True, b'bad cache', ValueError),
            (False, b'x' * 1_000_001, ValueError),
            (False, urllib.error.URLError('offline failure'), RuntimeError),
        ]:
            with self.subTest(cached=cached, error=error):
                bundle = Mock()
                bundle.exists.return_value = cached
                bundle.read_bytes.return_value = result if cached else payload
                with patch('pathlib.Path', return_value=bundle), \
                        patch('urllib.request.urlopen') as download, redirect_stdout(io.StringIO()):
                    if isinstance(result, Exception):
                        download.side_effect = result
                    else:
                        download.return_value.__enter__.return_value.read.return_value = result
                    if error:
                        with self.assertRaises(error):
                            exec(source, {})
                    else:
                        exec(source, {})
                    if cached:
                        download.assert_not_called()
                    else:
                        download.assert_called_once()
                    if not cached and error is None:
                        bundle.write_bytes.assert_called_once_with(payload)
                    else:
                        bundle.write_bytes.assert_not_called()

    def test_model_setup_uses_fresh_options(self) -> None:
        import lab_support
        options = {'max_tokens': 512, 'num_retries': 0}
        with patch.object(lab_support, 'configure_backend', return_value=('test/model', options)), \
                patch.object(lab_support, 'start_ollama') as start, redirect_stdout(io.StringIO()):
            model, configured = support.setup_model('custom', {}, False, ROOT / 'runs/offline-tests')
        start.assert_not_called()
        self.assertEqual(model, 'test/model')
        self.assertEqual(options['max_tokens'], 512)
        self.assertEqual(configured['max_tokens'], 1536)

    def test_real_frame_plan_without_execution(self) -> None:
        os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
        import docetl
        from docetl.frame import Frame
        docs, sectors = support.load_data(ROOT / 'document_data')
        namespace = {'docetl': docetl, 'MODEL_REF': 'deepseek/deepseek-flash', 'OPTIONS': {'max_tokens': 1536}}
        with patch.object(Frame, 'collect', side_effect=AssertionError('No model execution in offline tests')):
            for tag in ['filter', 'map', 'join', 'reduce']:
                definition = ast.parse(cell(tag)).body[0]
                exec(compile(ast.Module(body=[definition], type_ignores=[]), tag, 'exec'), namespace)
            query = namespace['keep_reports'](docetl.from_list(docs, name='documents'))
            query = namespace['extract_actions'](query)
            query = namespace['match_sectors'](query, docetl.from_list(sectors, name='sectors'))
            query = namespace['summarize_sectors'](query)
            import yaml
            plan = yaml.safe_load(query.to_yaml())
        self.assertEqual([op['type'] for op in plan['operations']], ['filter', 'map', 'equijoin', 'reduce'])
        self.assertEqual(plan['operations'][2]['blocking_conditions'], ['True'])
        self.assertEqual(plan['operations'][3]['reduce_key'], 'sector')
        self.assertFalse(plan['operations'][3]['synthesize_resolve'])
        self.assertEqual(plan['operations'][3]['output']['lineage'], ['doc_id'])
        for op in plan['operations']:
            self.assertEqual(op['litellm_completion_kwargs']['max_tokens'], 1536)

    def test_prompt_examples_render(self) -> None:
        from jinja2 import StrictUndefined, Template
        docs, sectors = support.load_data(ROOT / 'document_data')
        for tag in ['filter', 'map', 'join', 'reduce']:
            fn = ast.parse(cell(tag)).body[0]
            call = fn.body[-1].value
            prompt = next(ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in {'prompt', 'comparison_prompt'})
            context = {'input': docs[2], 'left': {'business': 'offline fixture'}, 'right': sectors[0],
                       'inputs': [{'company': 'offline fixture', 'actions': ['test action']}]}
            rendered = Template(prompt, undefined=StrictUndefined).render(**context)
            self.assertNotIn('{{', rendered)

    def test_source_snippet(self) -> None:
        source_root = Path(os.environ.get('DOCETL_SOURCE', '/Users/von/Projects/docetl-source'))
        paths = ['docetl/operations/filter.py',
                 'docetl/reasoning_optimizer/directives/clarify_instructions.py',
                 'docetl/reasoning_optimizer/directives/clarify_instructions.py']
        blocks = re.findall(r'```python\n(.*?)```', '\n'.join(item.source for item in NB.cells if item.cell_type == 'markdown'), re.S)
        self.assertEqual(len(blocks), len(paths))
        for block, path in zip(blocks, paths):
            actual = [line.strip() for line in (source_root / path).read_text().splitlines()]
            cursor = 0
            for chunk in block.strip().split('# ...'):
                expected = [line.strip() for line in chunk.strip().splitlines()]
                match = next((i for i in range(cursor, len(actual))
                              if actual[i:i + len(expected)] == expected), None)
                self.assertIsNotNone(match, path)
                cursor = match + len(expected)

    def test_record_trace_uses_actual_rows(self) -> None:
        docs = [{'doc_id': 'R05', 'text': 'OFFLINE <script>source</script>'}]
        facts = [{'doc_id': 'R05', 'company': 'XPO', 'business': 'Offline transport',
                  'actions': ['<b>Offline action</b>']}]
        matches = [{'doc_id': 'R05', 'sector': 'Transport'}]
        summaries = [{'sector': 'Transport', 'summary': 'Offline summary',
                      'summarize_by_sector_lineage': [{'doc_id': 'R05'}]}]
        trace = support.trace_document(docs, docs, facts, matches, summaries)
        self.assertEqual([r['step'] for r in trace], ['Source', 'Filter', 'Map', 'Join', 'Reduce'])
        self.assertEqual(trace[2]['observed'][0]['actions'], facts[0]['actions'])
        self.assertEqual(trace[4]['observed'][0]['summary'], summaries[0]['summary'])
        excluded = support.trace_document(docs, [], [], [], [])
        self.assertIn('Excluded', excluded[1]['observed'])
        self.assertIn('No extraction', excluded[2]['observed'])
        self.assertIn('No sector', excluded[3]['observed'])
        self.assertIn('No summary', excluded[4]['observed'])
        missing = support.trace_document(docs, docs, [], [], summaries)
        self.assertIn('No extraction', missing[2]['observed'])
        multi = support.trace_document(docs, docs, facts,
                                       matches + [{'doc_id': 'R05', 'sector': 'Manufacturing'}], summaries)
        self.assertEqual(len(multi[3]['observed']), 2)
        self.assertIn('Multiple', multi[3]['note'])
        unrelated = [{**summaries[0], 'summarize_by_sector_lineage': [{'doc_id': 'R06'}]}]
        self.assertIn('No summary', support.trace_document(docs, docs, facts, matches, unrelated)[4]['observed'])
        with patch('IPython.display.display') as display, redirect_stdout(io.StringIO()):
            support.show_trace(docs, docs, facts, matches, summaries)
        output = '\n'.join(call.args[0].data for call in display.call_args_list)
        self.assertNotIn('<script>', output)
        self.assertNotIn('<b>Offline', output)
        self.assertIn('&lt;script&gt;', output)

    def test_plan_plot(self) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = support.plot_plan()
        stream = io.BytesIO()
        fig.savefig(stream, format='png')
        self.assertGreater(len(stream.getvalue()), 5000)
        self.assertEqual(len(fig.axes[0].texts), 11)
        plt.close(fig)


class SectorQueryChecks(unittest.TestCase):
    """Use labelled offline fixtures, never provider responses, to test scoring."""

    def setUp(self) -> None:
        import docetl
        self.docs, self.sectors = support.load_data(ROOT / 'document_data')
        self.reference = support.load_reference(ROOT / 'document_data')
        self.rows = self.reference['expected_join_pairs']
        self.folder = Path(tempfile.mkdtemp(prefix='docetl-sector-test-', dir='/private/tmp'))
        ns = {'docetl': docetl, 'OPTIONS': {'max_tokens': 1536},
              'sector_rows': self.sectors, 'documents': self.docs}
        definition = ast.parse(cell('filter')).body[0]
        exec(compile(ast.Module(body=[definition], type_ignores=[]), 'filter', 'exec'), ns)
        exec(cell('sector_plan'), ns)
        self.query = ns['sector_query']

    def run_search(self, **kwargs: object) -> dict:
        return support.run_optimization(self.query, self.docs, self.rows, self.reference,
                                        'offline/model', self.folder,
                                        enabled=True, in_colab=True, **kwargs)

    def test_exact_score_and_fixed_denominator(self) -> None:
        self.assertEqual(support.score_sectors(self.rows, self.reference)['accuracy'], 1)
        self.assertEqual(support.score_sectors(self.rows[:-1], self.reference)['accuracy'], 5 / 6)
        self.assertEqual(support.score_sectors([], self.reference)['accuracy'], 2 / 6)
        extra = self.rows + [{'doc_id': 'R08', 'sector': 'Manufacturing'}]
        self.assertEqual(support.score_sectors(extra, self.reference)['accuracy'], 5 / 6)
        swapped = [dict(row) for row in self.rows]
        swapped[0]['sector'], swapped[2]['sector'] = swapped[2]['sector'], swapped[0]['sector']
        self.assertEqual(support.sector_counts(swapped, self.reference),
                         support.sector_counts(self.rows, self.reference))
        self.assertEqual(support.score_sectors(swapped, self.reference)['accuracy'], 4 / 6)
        self.assertEqual([r['company_count'] for r in support.sector_counts([], self.reference)], [0, 0])

    def test_invalid_outputs_are_not_deduplicated_or_coerced(self) -> None:
        for rows in [None, {}, self.rows * 2, [{'doc_id': 'unknown', 'sector': 'Manufacturing'}],
                     [{'doc_id': 'R01'}], [{'sector': 'Manufacturing'}],
                     [{'doc_id': 'R01', 'sector': 'manufacturing'}],
                     [{'doc_id': 'R01', 'sector': ['Manufacturing']}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                support.score_sectors(rows, self.reference)

    def test_query_and_frozen_data_have_no_gold(self) -> None:
        import yaml
        original, path = support._freeze_query(self.query, self.docs, self.folder, 'offline/model')
        self.assertEqual([op['type'] for op in original['operations']], ['filter', 'map'])
        for sector in self.sectors:
            self.assertIn(sector['definition'], original['operations'][1]['prompt'])
        self.assertEqual(json.loads(path.read_text()), self.docs)
        self.assertNotIn('expected_join_pairs', yaml.safe_dump(original))
        self.assertNotIn('golden_answers', yaml.safe_dump(original))
        self.assertNotIn('R01', yaml.safe_dump(original))
        with self.assertRaises(ValueError):
            support._freeze_query(self.query, self.docs[:4], self.folder, 'offline/model')

    def test_defaults_and_model_adapter(self) -> None:
        with patch('litellm.completion', side_effect=AssertionError('No model requests')):
            self.assertIsNone(support.run_optimization(None, [], [], {}, '', self.folder,
                                                       enabled=False, in_colab=False))
            ns = {'support': support, 'documents': self.docs}
            with redirect_stdout(io.StringIO()):
                exec(cell('document_moar'), ns)
                exec(cell('document_prompt_case'), ns)
        with self.assertRaisesRegex(RuntimeError, 'Colab'):
            support.run_optimization(None, [], [], {}, '', self.folder, enabled=True, in_colab=False)
        original = Mock(return_value='offline')
        wrapped = support._moar_tool_options(original)
        wrapped(model='deepseek/deepseek-flash', tool_choice='required')
        self.assertEqual(original.call_args.kwargs['extra_body']['thinking'], {'type': 'disabled'})
        wrapped(model='other/model', tool_choice='required')
        self.assertNotIn('extra_body', original.call_args.kwargs)
        wrapped(model='deepseek/deepseek-flash', tool_choice='auto')
        self.assertNotIn('extra_body', original.call_args.kwargs)

    def test_candidate_boundaries(self) -> None:
        original, _ = support._freeze_query(self.query, self.docs, self.folder, 'offline/model')
        support._check_candidate(original, original, self.folder, 'offline/model')
        changes = [
            lambda c: c.update(datasets={}),
            lambda c: c.update(default_model='other/model'),
            lambda c: c.update(bypass_cache=False),
            lambda c: c['pipeline']['output'].update(path='/private/tmp/outside.json'),
            lambda c: c['operations'][0].update(model='other/model'),
            lambda c: c['operations'][0].update(litellm_completion_kwargs={'api_base': 'https://invalid.example'}),
        ]
        for change in changes:
            candidate = json.loads(json.dumps(original))
            change(candidate)
            with self.assertRaises(ValueError):
                support._check_candidate(candidate, original, self.folder, 'offline/model')

    def test_local_embeddings_are_role_scoped(self) -> None:
        original, _ = support._freeze_query(self.query, self.docs, self.folder, 'offline/model')
        candidate = json.loads(json.dumps(original))
        candidate['operations'].append({'type': 'topk', 'method': 'embedding',
                                        'embedding_model': support.LOCAL_EMBEDDING_MODEL})
        with self.assertRaisesRegex(support.CandidateNotExecuted, 'not enabled'):
            support._check_candidate(candidate, original, self.folder, 'offline/model')
        support._check_candidate(candidate, original, self.folder, 'offline/model',
                                 allow_local_embeddings=True)
        candidate['operations'][0]['model'] = support.LOCAL_EMBEDDING_MODEL
        with self.assertRaises(support.CandidateNotExecuted):
            support._check_candidate(candidate, original, self.folder, 'offline/model',
                                     allow_local_embeddings=True)
        candidate['operations'][0]['model'] = 'gpt-5-nano'
        with self.assertRaisesRegex(support.CandidateNotExecuted, 'gpt-5-nano'):
            support._check_candidate(candidate, original, self.folder, 'offline/model',
                                     allow_local_embeddings=True)

    def test_local_embedding_defaults_and_request(self) -> None:
        ops = [{'type': 'topk', 'method': 'embedding'},
               {'type': 'filter', 'model': 'gpt-5-nano'},
               {'type': 'topk', 'method': 'embedding', 'embedding_model': 'other/model'}]
        changed = support._local_topk_operators(ops)
        self.assertEqual(changed[0]['embedding_model'], support.LOCAL_EMBEDDING_MODEL)
        self.assertNotIn('embedding_model', ops[0])
        self.assertEqual(changed[1:], ops[1:])
        call = Mock(return_value={'data': []})
        embed = support._local_embedding_call(call)
        with patch.object(support, 'embedding_device', return_value='CPU'):
            embed(model=support.LOCAL_EMBEDDING_MODEL, input=['source text'])
        self.assertFalse(call.call_args.kwargs['truncate'])
        self.assertNotIn('options', call.call_args.kwargs)
        self.assertEqual(call.call_args.kwargs['api_base'], support.LOCAL_EMBEDDING_API)
        with self.assertRaises(support.CandidateNotExecuted):
            embed(model='text-embedding-3-small', input=['source text'])
        self.assertEqual(call.call_count, 1)

    def test_embedding_device_reports_actual_placement(self) -> None:
        for size, vram, expected in [(100, 0, 'CPU'), (100, 100, 'GPU'),
                                     (100, 50, 'CPU + GPU')]:
            data = {'models': [{'name': 'nomic-embed-text:v1.5', 'size': size, 'size_vram': vram}]}
            with patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(data))):
                self.assertEqual(support.embedding_device(), expected)
        with patch('urllib.request.urlopen', side_effect=TimeoutError):
            self.assertIn('Unknown', support.embedding_device())

    def test_local_embedding_rejects_silent_truncation(self) -> None:
        retrieve = Mock(return_value=([[1.0]], 0))
        guarded = support._bounded_retrieval(retrieve)
        config = {'embedding_model': support.LOCAL_EMBEDDING_MODEL, 'embedding_keys': ['text']}
        guarded([{'text': 'x' * 1000}], config, None)
        with self.assertRaisesRegex(ValueError, '1000-character'):
            guarded([{'text': 'x' * 1001}], config, None)
        self.assertEqual(retrieve.call_count, 1)

    def test_disabled_embedding_support_has_no_setup(self) -> None:
        import lab_support
        with patch.object(lab_support, 'start_ollama') as start:
            with support.local_embedding_support(False, False, self.folder):
                pass
        start.assert_not_called()

    def test_embedding_directive_binding_and_restoration(self) -> None:
        import lab_support
        from docetl.reasoning_optimizer.directives.doc_chunking_topk import DocumentChunkingTopKDirective
        from docetl.operations.utils import api
        original_embedding = api.embedding
        directive = DocumentChunkingTopKDirective()
        generated = [{'type': 'topk', 'method': 'embedding'}]
        with patch.object(lab_support, 'start_ollama') as start, \
                patch.object(DocumentChunkingTopKDirective, 'apply', return_value=generated):
            with support.local_embedding_support(True, False, self.folder):
                output = directive.apply()
                self.assertEqual(output[0]['embedding_model'], support.LOCAL_EMBEDDING_MODEL)
            self.assertNotIn('embedding_model', generated[0])
            start.assert_called_once_with('ollama', support.LOCAL_EMBEDDING_MODEL, False, self.folder)
        self.assertIs(api.embedding, original_embedding)

    def test_real_optimize_call_with_fake_execution_and_search(self) -> None:
        import yaml
        from docetl.frame import Frame
        from docetl.moar.Node import Node
        def execute(node: object, max_threads: int = 1) -> float:
            self.assertEqual(max_threads, 1)
            Path(node.parsed_yaml['pipeline']['output']['path']).write_text(json.dumps(self.rows))
            return .01
        def optimize(frame: object, **kwargs: object) -> object:
            from docetl.moar import optimizer
            names = {action.name for action in optimizer.ALL_DIRECTIVES}
            self.assertNotIn('cascade_filtering', names)
            self.assertIn('doc_chunking_topk', names)
            self.assertEqual(kwargs['metric_key'], 'accuracy')
            self.assertEqual(kwargs['models'], ['offline/model'])
            self.assertEqual(kwargs['agent_model'], 'offline/model')
            self.assertEqual(kwargs['max_iterations'], 2)
            self.assertEqual(kwargs['max_concurrent_agents'], 1)
            self.assertEqual(kwargs['max_threads'], 1)
            self.assertEqual(json.loads(Path(kwargs['dataset_path']).read_text()), self.docs)
            config = yaml.safe_load(frame.to_yaml())
            config['operations'][0]['prompt'] += '\nOffline test modification.'
            path = Path(kwargs['save_dir']) / 'candidate.yaml'
            config['pipeline']['output']['path'] = str(path.with_suffix('.json'))
            path.write_text(yaml.safe_dump(config))
            node = Node(str(path))
            node.execute_plan()
            self.assertEqual(kwargs['eval_fn'](config['pipeline']['output']['path']), {'accuracy': 1})
            return SimpleNamespace(search_results=SimpleNamespace(total_search_cost=.02))
        with patch.object(Node, 'execute_plan', execute), patch.object(Frame, 'optimize', optimize), \
                patch('litellm.completion', side_effect=AssertionError('No judge or model requests')):
            record = support.run_optimization(self.query, self.docs, self.rows[:-1], self.reference,
                                               'offline/model', self.folder, enabled=True, in_colab=True,
                                               baseline_work={'execution_seconds': 2.5,
                                                              'execution_cost_estimate_usd': .003,
                                                              'token_usage': {'offline/model': {'prompt_tokens': 12}},
                                                              'cache': 'Query cache allowed'})
        self.assertEqual(record['selected_id'], 'trial-1')
        self.assertEqual(record['plans'][0]['execution_seconds'], 2.5)
        self.assertEqual(record['plans'][0]['execution_cost_estimate_usd'], .003)
        self.assertEqual(record['plans'][0]['token_usage']['offline/model']['prompt_tokens'], 12)
        self.assertEqual(record['plans'][0]['rate'], 5 / 6)
        self.assertEqual(support.selected_rows(record), self.rows)
        self.assertEqual(record['format'], 'docetl-sector-search-v2')
        self.assertEqual(record['disabled_search_directives'], ['cascade_filtering'])
        with patch.object(support, 'show_rows'), patch.object(support, 'show_document'), \
                patch('IPython.display.display'), redirect_stdout(io.StringIO()):
            support.show_search(record, self.docs)
        import matplotlib.pyplot as plt
        figure = support.plot_search(record)
        self.assertEqual([bar.get_height() for bar in figure.axes[0].patches], [5 / 6, 1])
        self.assertEqual(figure.axes[0].get_ylabel(), 'Document accuracy')
        plt.close(figure)

    def test_cascade_exclusion_is_scoped_and_restored_on_error(self) -> None:
        from docetl.moar import optimizer
        original = optimizer.ALL_DIRECTIVES
        self.assertIn('cascade_filtering', {action.name for action in original})
        with self.assertRaisesRegex(RuntimeError, 'offline failure'):
            with support.classroom_search_actions():
                self.assertEqual({action.name for action in optimizer.ALL_DIRECTIVES},
                                 {action.name for action in original} - {'cascade_filtering'})
                raise RuntimeError('offline failure')
        self.assertIs(optimizer.ALL_DIRECTIVES, original)

    def test_actual_moar_initial_trial_without_model_requests(self) -> None:
        from docetl.moar.Node import Node
        from docetl.moar.MOARSearch import MOARSearch
        def execute(node: object, max_threads: int = 1) -> float:
            Path(node.parsed_yaml['pipeline']['output']['path']).write_text(json.dumps(self.rows))
            node.cost = .01
            return node.cost
        with patch.object(Node, 'execute_plan', execute), patch.object(Node, 'delete'), \
                patch.object(MOARSearch, 'search'), \
                patch('litellm.completion', side_effect=AssertionError('No live requests')):
            record = self.run_search()
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['selected_id'], 'original')
        self.assertEqual(len(record['plans']), 2)

    def test_single_strategy_tie_and_failed_execution(self) -> None:
        import yaml
        from docetl.moar.Node import Node
        from docetl.reasoning_optimizer.directives.clarify_instructions import ClarifyInstructionsDirective
        def instantiate(directive: object, **kwargs: object) -> tuple:
            self.assertEqual(len(kwargs['operators']), 2)
            self.assertEqual(kwargs['target_ops'], ['keep_sustainability_reports'])
            self.assertEqual(json.loads(Path(kwargs['input_file_path']).read_text()), self.docs)
            ops = json.loads(json.dumps(kwargs['operators']))
            ops[0]['prompt'] += '\nOffline modification.'
            return ops, [], .01
        def execute(node: object, max_threads: int = 1) -> float:
            Path(node.parsed_yaml['pipeline']['output']['path']).write_text(json.dumps(self.rows))
            return 0
        with patch.object(ClarifyInstructionsDirective, 'instantiate', instantiate), \
                patch.object(Node, 'execute_plan', execute):
            record = self.run_search(prompt_only=True)
        self.assertEqual(record['kind'], 'separate_prompt_case')
        self.assertEqual(record['selected_id'], 'original')
        self.assertIsNone(record['plans'][1]['execution_cost_estimate_usd'])
        with patch.object(ClarifyInstructionsDirective, 'instantiate', instantiate), \
                patch.object(Node, 'execute_plan', return_value=-1), \
                self.assertRaisesRegex(RuntimeError, 'execution failed'):
            self.run_search(prompt_only=True)
        records = [json.loads(p.read_text()) for p in self.folder.rglob('search.json')]
        failed = next(r for r in records if r['status'] == 'failed')
        self.assertIsNone(failed['selected_id'])
        self.assertIsNone(failed['plans'][1]['rate'])

    def test_invalid_trial_cannot_become_empty_success(self) -> None:
        from docetl.frame import Frame
        from docetl.moar.Node import Node
        import yaml
        def execute(node: object, max_threads: int = 1) -> float:
            Path(node.parsed_yaml['pipeline']['output']['path']).write_text('[{"doc_id":"R01"}]')
            return .01
        def optimize(frame: object, **kwargs: object) -> object:
            config = yaml.safe_load(frame.to_yaml())
            path = Path(kwargs['save_dir']) / 'invalid.yaml'
            config['pipeline']['output']['path'] = str(path.with_suffix('.json'))
            path.write_text(yaml.safe_dump(config))
            Node(str(path)).execute_plan()
            with self.assertRaises(ValueError):
                kwargs['eval_fn'](config['pipeline']['output']['path'])
            return SimpleNamespace(search_results=SimpleNamespace(total_search_cost=.01))
        with patch.object(Node, 'execute_plan', execute), patch.object(Frame, 'optimize', optimize), \
                self.assertRaisesRegex(RuntimeError, 'no valid trial'):
            self.run_search()
        record = json.loads(next(self.folder.rglob('search.json')).read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertIsNone(record['selected_id'])

    def test_legacy_and_display_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, 'old extraction'):
            support.selected_rows({'format': 'docetl-document-search-v1'})
        plans = [{'id': 'original', 'rate': 1, 'changed': False},
                 {'id': 'trial', 'rate': 1, 'changed': True}]
        self.assertEqual(support.choose_candidate(plans)['id'], 'original')
        plans[0]['rate'] = .5
        self.assertEqual(support.choose_candidate(plans)['id'], 'trial')
        plans[1]['error'] = 'Invalid output'
        self.assertEqual(support.choose_candidate(plans)['id'], 'original')
        with redirect_stdout(io.StringIO()), patch.object(support, 'show_rows') as show:
            support.show_sector_result([], self.reference)
            self.assertEqual(show.call_args_list[0].args[0][0]['actual'], 'Excluded')
            support.show_search(None, self.docs)
        self.assertIsNone(support.plot_search(None))

    def test_search_display_order_failures_and_unchanged_answers(self) -> None:
        original, _ = support._freeze_query(self.query, self.docs, self.folder, 'offline/model')
        def plan(name: str, rows: list[dict], changed: bool) -> dict:
            config = json.loads(json.dumps(original))
            if changed:
                config['operations'][0]['prompt'] += '\n<em>Offline change</em>'
            score = support.score_sectors(rows, self.reference)
            return {'id': name, 'config': config, 'rows': rows, 'score': score,
                    'rate': score['accuracy'], 'changed': changed}
        baseline = plan('original', self.rows, False)
        first = plan('first', self.rows[:-1], True)
        later = plan('later', self.rows, True)
        failed = {'id': 'failed', 'changed': True, 'error': '<script>bad</script>', 'rate': None}
        record = {'format': support.SEARCH_FORMAT, 'kind': 'moar_search', 'status': 'completed',
                  'selected_id': 'original', 'plans': [baseline, first, later, failed]}
        with patch.object(support, 'show_rows') as tables, patch.object(support, 'show_document') as doc, \
                patch('IPython.display.display') as displays, redirect_stdout(io.StringIO()):
            support.show_search(record, self.docs)
        doc.assert_called_once_with(self.docs, self.rows[-1]['doc_id'])
        quality = next(c.args[0] for c in tables.call_args_list if 'correct_documents' in c.args[1])
        self.assertEqual([r['correct_documents'] for r in quality], ['6/6', '5/6', '6/6', 'Unavailable'])
        self.assertEqual(quality[-1]['status'], 'Failed')
        costs = next(c.args[0] for c in tables.call_args_list if 'execution_seconds' in c.args[1])
        self.assertTrue(all(r['execution_cost_estimate_usd'] == 'Unknown' for r in costs))
        html = '\n'.join(c.args[0].data for c in displays.call_args_list)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>', html)
        record['plans'] = [baseline, later]
        output = io.StringIO()
        with patch.object(support, 'show_rows'), patch.object(support, 'show_document') as doc, \
                patch('IPython.display.display'), redirect_stdout(output):
            support.show_search(record, self.docs)
        doc.assert_not_called()
        self.assertIn('stayed the same', output.getvalue())
        record.update(status='failed', selected_id=None, plans=[baseline, failed])
        output = io.StringIO()
        with patch.object(support, 'show_rows'), patch('IPython.display.display'), redirect_stdout(output):
            support.show_search(record, self.docs)
        self.assertIn('status: failed', output.getvalue())
        self.assertIn('No changed candidate', output.getvalue())


if __name__ == '__main__':
    unittest.main()

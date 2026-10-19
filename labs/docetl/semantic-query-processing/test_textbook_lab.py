"""Offline checks for the active restaurant-review lab; no model calls."""
import ast
import hashlib
import json
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
        exec(compile(saved['query_code'], '<saved query>', 'exec'), namespace)
        actual = namespace['moar_query'](Frame()).operations
        self.assertEqual(actual, saved['operations'])


if __name__ == '__main__':
    unittest.main()

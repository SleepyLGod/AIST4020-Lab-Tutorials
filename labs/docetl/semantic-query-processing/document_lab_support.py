"""Loading and display helpers for the document-query lab; no import-time requests."""
from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import copy
import math
import time
import uuid
from pathlib import Path
from typing import Any


def load_data(directory: Path) -> tuple[list[dict], list[dict]]:
    """Load the fixed inputs and verify the text against its provenance."""
    documents = json.loads((directory / 'documents.json').read_text())
    sectors = json.loads((directory / 'sectors.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    ids = [row['doc_id'] for row in documents]
    if ids != ['R01', 'R02', 'R05', 'R06', 'R08', 'R10']:
        raise ValueError('The six-document input does not match this lab.')
    hashes = {row['doc_id']: row['excerpt_sha256'] for row in provenance}
    for row in documents:
        if set(row) != {'doc_id', 'text'} or not row['text'].strip():
            raise ValueError('Model inputs must contain only doc_id and nonempty text.')
        if hashlib.sha256(row['text'].encode()).hexdigest() != hashes[row['doc_id']]:
            raise ValueError(f"Changed excerpt: {row['doc_id']}")
        if (directory / 'reports' / f"{row['doc_id']}.txt").read_text() != row['text']:
            raise ValueError('The displayed document differs from the query input.')
    if len(sectors) != 2 or len({row['sector'] for row in sectors}) != 2:
        raise ValueError('Expected two distinct teaching sectors.')
    return documents, sectors


def table_html(rows: list[dict], columns: list[str]) -> str:
    """Render values as escaped text, including model-generated content."""
    heading = ''.join(f'<th>{html.escape(column)}</th>' for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, '')
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            cells.append(f'<td style="white-space:pre-wrap;overflow-wrap:anywhere;vertical-align:top">{html.escape(str(value))}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table style="table-layout:fixed;width:100%"><thead><tr>' + heading + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'


def show_rows(rows: list[dict], columns: list[str]) -> None:
    """Display a compact result table without truncating its text."""
    from IPython.display import HTML, display
    print(f'{len(rows)} record(s)')
    display(HTML(table_html(rows, columns)))


def show_document(rows: list[dict], doc_id: str) -> None:
    """Display one full excerpt as plain text."""
    from IPython.display import HTML, display
    row = next((row for row in rows if row['doc_id'] == doc_id), None)
    if row is None:
        raise ValueError(f'Unknown document: {doc_id}')
    display(HTML(f'<h4>{html.escape(doc_id)}</h4><pre style="white-space:pre-wrap">{html.escape(row["text"])}</pre>'))


def trace_document(documents: list[dict], kept: list[dict], facts: list[dict],
                   matches: list[dict], summaries: list[dict], doc_id: str = 'R05') -> list[dict]:
    """Trace recorded outputs without inferring missing intermediate results."""
    source = next(row for row in documents if row['doc_id'] == doc_id)
    retained = [row for row in kept if row['doc_id'] == doc_id]
    extracted = [{key: row.get(key) for key in ['company', 'business', 'actions']}
                 for row in facts if row['doc_id'] == doc_id]
    joined = [row['sector'] for row in matches if row['doc_id'] == doc_id]
    grouped = [{'sector': row['sector'], 'summary': row['summary']} for row in summaries
               if any(isinstance(item, dict) and item.get('doc_id') == doc_id
                      for item in (row.get('summarize_by_sector_lineage') or []))]
    return [
        {'step': 'Source', 'observed': source['text']},
        {'step': 'Filter', 'observed': 'Kept' if retained else 'Excluded; no row was retained'},
        {'step': 'Map', 'observed': extracted or 'No extraction returned for this document'},
        {'step': 'Join', 'observed': joined or 'No sector matched',
         'note': 'Multiple matches: this document can enter more than one group.' if len(joined) > 1 else ''},
        {'step': 'Reduce', 'observed': grouped or 'No summary records this document in its input IDs'},
    ]


def show_trace(documents: list[dict], kept: list[dict], facts: list[dict],
               matches: list[dict], summaries: list[dict], doc_id: str = 'R05') -> None:
    """Follow one document through the existing run; make no model calls."""
    from IPython.display import HTML, display
    from lab_support import text_details
    rows = trace_document(documents, kept, facts, matches, summaries, doc_id)
    print(f'Follow {doc_id} through this run')
    display(HTML(text_details('Read the source excerpt', rows[0]['observed'])))
    show_rows(rows[1:], ['step', 'observed', 'note'])


def setup_model(backend: str, backends: dict, in_colab: bool, runs: Path) -> tuple[str, dict]:
    """Reuse the existing credential and provider setup without running a query."""
    import lab_support
    runs.mkdir(parents=True, exist_ok=True)
    # Retain getpass prompts while suppressing movie-specific status messages.
    with contextlib.redirect_stdout(io.StringIO()):
        model, options = lab_support.configure_backend(backend, backends, in_colab)
    if backend == 'ollama':
        print('Preparing the selected Ollama model; the download may take several minutes.')
        with contextlib.redirect_stdout(io.StringIO()):
            lab_support.start_ollama(backend, model, in_colab, runs)
    print(f'Model: {model}. Setup has not sent any document queries.')
    return model, {**options, 'max_tokens': 1536}


def require_enabled(enabled: bool) -> None:
    """Stop before a model request unless the student enables calls."""
    if not enabled:
        raise RuntimeError('Set ENABLE_MODEL_CALLS = True in the backend cell, then rerun it.')


def check_rows(inputs: list[dict], outputs: list[dict], *, complete: bool) -> None:
    """Reject unknown/duplicate documents and, for map, missing results."""
    expected = {row['doc_id'] for row in inputs}
    seen = [row['doc_id'] for row in outputs]
    if len(set(seen)) != len(seen) or not set(seen) <= expected:
        raise ValueError('The output contains duplicate or unknown document IDs.')
    if complete and set(seen) != expected:
        raise ValueError(f'Missing document results: {sorted(expected - set(seen))}')
    original = {row['doc_id']: row['text'] for row in inputs}
    if any(row.get('text') != original[row['doc_id']] for row in outputs):
        raise ValueError('The original text was lost or changed in the output.')
    if complete:
        for row in outputs:
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ('company', 'business')):
                raise ValueError('Each map result needs a company and business string.')
            if not isinstance(row.get('actions'), list) or not all(isinstance(item, str) for item in row['actions']):
                raise ValueError('Each map result needs a list of action strings.')


def inspect_matches(facts: list[dict], sectors: list[dict], matches: list[dict]) -> None:
    """Report missing and multiple sector matches without dropping results."""
    ids = {row['doc_id'] for row in facts}
    names = {row['sector'] for row in sectors}
    pairs = [(row['doc_id'], row['sector']) for row in matches]
    if len(set(pairs)) != len(pairs) or any(doc not in ids or sector not in names for doc, sector in pairs):
        raise ValueError('Join output contains duplicate or unknown pairs.')
    counts = {doc: sum(left == doc for left, _ in pairs) for doc in sorted(ids)}
    unusual = {doc: count for doc, count in counts.items() if count != 1}
    print(f'Matches per document needing a closer look: {unusual}' if unusual else 'Each input document matched one sector.')


def check_summaries(matches: list[dict], summaries: list[dict]) -> None:
    """Check group coverage, not the factual quality of the generated prose."""
    expected = {row['sector'] for row in matches}
    actual = [row['sector'] for row in summaries]
    if set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError('Missing, duplicate, or unknown summary groups.')
    if any(not isinstance(row.get('summary'), str) or not row['summary'].strip() for row in summaries):
        raise ValueError('A group has no summary text.')


def add_company_counts(summaries: list[dict]) -> list[dict]:
    """Count this run's source documents, one per company, without a model call."""
    rows = []
    seen_sectors = set()
    for summary in summaries:
        sector = summary.get('sector')
        if not isinstance(sector, str) or not sector.strip() or sector in seen_sectors:
            raise ValueError('Missing or duplicate summary sector.')
        if not isinstance(summary.get('summary'), str) or not summary['summary'].strip():
            raise ValueError('A group has no summary text.')
        lineage = summary.get('summarize_by_sector_lineage')
        if not isinstance(lineage, list) or not lineage:
            raise ValueError('Missing source document IDs; rerun the query with output lineage enabled.')
        ids = [row.get('doc_id') if isinstance(row, dict) else None for row in lineage]
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in ids):
            raise ValueError('Invalid source document ID in summary.')
        if len(set(ids)) != len(ids):
            raise ValueError('A company document appears more than once in a sector; check the join.')
        seen_sectors.add(sector)
        rows.append({**summary, 'company_count': len(ids)})
    return rows


def plot_plan() -> Any:
    """Draw the query structure, without invented result counts or timings."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set(xlim=(-0.8, 4.8), ylim=(-0.6, 1.7))
    ax.axis('off')
    labels = ['Document\nrecords', 'Filter\nkeep reports', 'Map\nextract facts', 'Join\nmatch sectors', 'Reduce\nsummarize groups']
    for i, label in enumerate(labels):
        ax.text(i, 0, label, ha='center', va='center', fontsize=10,
                bbox={'boxstyle': 'round,pad=0.4', 'fc': '#edf5f2' if i % 2 else '#f4f1f9', 'ec': '#65716c'})
        if i:
            ax.annotate('', xy=(i - 0.39, 0), xytext=(i - 0.62, 0), arrowprops={'arrowstyle': '->'})
    ax.text(3, 1.1, 'Sector definitions', ha='center', fontsize=10)
    ax.annotate('', xy=(3, 0.36), xytext=(3, 0.91), arrowprops={'arrowstyle': '->'})
    fig.tight_layout()
    return fig


def load_reference(directory: Path) -> dict:
    """Verify the teaching reference against the exact supplied excerpts."""
    reference = json.loads((directory / 'golden_answers.json').read_text())
    for name, expected in reference['input_files'].items():
        if name not in {'documents.json', 'sectors.json'}:
            raise ValueError('Unexpected reference input file.')
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('The reference does not match this data bundle.')
    docs, _ = load_data(directory)
    texts = {row['doc_id']: row['text'] for row in docs}
    entries = reference['documents']
    if len(entries) != len(texts) or {row['doc_id'] for row in entries} != set(texts):
        raise ValueError('The reference must cover every document exactly once.')
    for row in entries:
        quotes = [row.get('filter_evidence'), row.get('business_evidence')]
        quotes += [action['evidence'] for action in row['acceptable_actions']]
        if any(quote not in texts[row['doc_id']] for quote in quotes if quote):
            raise ValueError('Reference evidence is absent from its source excerpt.')
    keep = {row['doc_id'] for row in entries if row['keep']}
    pairs = {(row['doc_id'], row['sector']) for row in entries if row['keep']}
    if keep != set(reference['expected_filter_ids']) or pairs != {
        (row['doc_id'], row['sector']) for row in reference['expected_join_pairs']
    }:
        raise ValueError('Inconsistent reference decisions.')
    return reference


def _index(rows: list[dict], key: str, allowed: set[str]) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError('Expected a list of output records.')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise ValueError(f'Missing or invalid {key}.')
        value = row[key]
        if value not in allowed or value in result:
            raise ValueError(f'Duplicate or unknown {key}.')
        result[value] = row
    return result


def _save_record(path: Path, record: dict) -> None:
    from lab_support import check_record_secrets
    check_record_secrets(record)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))


def _positive_cost(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    return None


LOCAL_EMBEDDING_MODEL = 'ollama/nomic-embed-text:v1.5'
LOCAL_EMBEDDING_API = 'http://127.0.0.1:11434'


class CandidateNotExecuted(ValueError):
    """A candidate requires a model outside the explicitly enabled configuration."""


def _local_topk_operators(operators: list[dict]) -> list[dict]:
    """Bind only the chunk-retrieval directive's default embedding to the local model."""
    result = copy.deepcopy(operators)
    for op in result:
        if op.get('type') == 'topk' and op.get('method') == 'embedding':
            if op.get('embedding_model') in (None, 'text-embedding-3-small', LOCAL_EMBEDDING_MODEL):
                op['embedding_model'] = LOCAL_EMBEDDING_MODEL
    return result


def embedding_device() -> str:
    """Read Ollama's actual placement after loading, rather than infer it from GPU presence."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f'{LOCAL_EMBEDDING_API}/api/ps', timeout=3) as response:
            models = json.load(response)['models']
        for row in models:
            if row.get('name') == LOCAL_EMBEDDING_MODEL.split('/', 1)[1]:
                size, vram = row.get('size'), row.get('size_vram')
                if isinstance(size, int) and size > 0 and isinstance(vram, int) and vram >= 0:
                    return 'CPU' if vram == 0 else 'GPU' if vram >= size else 'CPU + GPU'
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError):
        return 'Unknown (device status unavailable)'
    return 'Unknown (model not listed)'


def _local_embedding_call(original: Any) -> Any:
    """Let Ollama place approved embeddings on available hardware; never truncate silently."""
    reported = False
    def embed(*args: Any, **kwargs: Any) -> Any:
        nonlocal reported
        if kwargs.get('model') != LOCAL_EMBEDDING_MODEL or args:
            raise CandidateNotExecuted('Embedding model is not enabled; candidate was not executed.')
        response = original(model=LOCAL_EMBEDDING_MODEL, input=kwargs['input'],
                        api_base=LOCAL_EMBEDDING_API, truncate=False,
                        timeout=120)
        if not reported:
            print(f'Local embeddings: {embedding_device()}')
            reported = True
        return response
    return embed


def _bounded_retrieval(original: Any) -> Any:
    """Reject inputs that DocETL 0.3.0 would silently shorten before embedding."""
    def retrieve(items: list[dict], config: dict, api: Any) -> Any:
        if config.get('embedding_model') == LOCAL_EMBEDDING_MODEL and items:
            from docetl.operations.utils.validation import lookup_field
            keys = config.get('embedding_keys') or list(items[0])
            for item in items:
                text = ' '.join(str(lookup_field(item, key)) for key in keys)
                if len(text) > 1000:
                    raise ValueError('Retrieval text exceeds DocETL 0.3.0\'s 1000-character limit; use smaller chunks.')
        return original(items, config, api)
    return retrieve


@contextlib.contextmanager
def local_embedding_support(enabled: bool, in_colab: bool, runs: Path) -> Any:
    """Enable one local embedding model only for the duration of an explicit search."""
    if not enabled:
        yield
        return
    import lab_support
    from unittest.mock import patch
    from docetl.operations import sample
    from docetl.operations.utils import api
    from docetl.reasoning_optimizer.directives.doc_chunking_topk import DocumentChunkingTopKDirective
    print('Preparing optional local embeddings; no additional API key is used.')
    with contextlib.redirect_stdout(io.StringIO()):
        lab_support.start_ollama('ollama', LOCAL_EMBEDDING_MODEL, in_colab, runs)
    apply = DocumentChunkingTopKDirective.apply

    def apply_local(*args: Any, **kwargs: Any) -> list[dict]:
        return _local_topk_operators(apply(*args, **kwargs))

    with patch.object(DocumentChunkingTopKDirective, 'apply', apply_local), \
            patch.object(api, 'embedding', _local_embedding_call(api.embedding)), \
            patch.object(sample, 'get_embeddings_for_clustering',
                         _bounded_retrieval(sample.get_embeddings_for_clustering)):
        yield


def _check_candidate(config: dict, original: dict, folder: Path, model: str,
                     *, allow_local_embeddings: bool = False) -> None:
    from lab_support import check_record_secrets
    check_record_secrets(config)
    if config.get('datasets') != original['datasets']:
        raise ValueError('Candidate changed the frozen input dataset.')
    output = config.get('pipeline', {}).get('output', {})
    if output.get('type') != 'file' or not Path(output.get('path', '')).resolve().is_relative_to(folder.resolve()):
        raise ValueError('Candidate output must stay in this search folder.')
    if config.get('default_model') != model or config.get('fallbacks'):
        raise ValueError('Candidate changed the selected provider/model.')
    if config.get('bypass_cache') is not True:
        raise ValueError('Candidate changed the trial cache setting.')
    original_options = original['operations'][0].get('litellm_completion_kwargs', {})
    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'embedding_model':
                    allowed = (None, LOCAL_EMBEDDING_MODEL) if allow_local_embeddings else (None,)
                else:
                    allowed = (None, model)
                if (key == 'model' or key.endswith('_model')) and child not in allowed:
                    raise CandidateNotExecuted(f'Model {child!r} is not enabled; candidate was not executed.')
                if key in {'api_base', 'base_url'} and child != original_options.get('api_base'):
                    raise ValueError('Candidate introduced another API endpoint.')
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(config['operations'])


def _moar_tool_options(original: Any) -> Any:
    """Keep forced DeepSeek tool calls compatible with this lab's non-thinking route."""
    def complete(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get('model', args[0] if args else '')
        choice = kwargs.get('tool_choice')
        if str(model).startswith('deepseek/') and (choice == 'required' or isinstance(choice, dict)):
            kwargs = {**kwargs, 'extra_body': {
                **(kwargs.get('extra_body') or {}), 'thinking': {'type': 'disabled'},
            }}
        return original(*args, **kwargs)
    return complete


SEARCH_FORMAT = 'docetl-sector-search-v2'
DISABLED_SEARCH_DIRECTIVES = ('cascade_filtering',)
OLLAMA_OPTIMIZER_CONTEXT = 16384


@contextlib.contextmanager
def optimizer_context(model: str, run_dir: Path) -> Any:
    """Set Ollama rewrite context and reject responses produced from truncated input."""
    if not model.startswith('ollama_chat/'):
        yield
        return
    import sys
    import litellm
    original = litellm.completion
    log_path = run_dir / 'ollama-server.log'
    truncated = False

    def complete(*args: Any, **kwargs: Any) -> Any:
        nonlocal truncated
        if truncated:
            raise RuntimeError('Ollama truncated optimizer input; stop this run and inspect its log.')
        target = kwargs.get('model', args[0] if args else '')
        if target != model:
            return original(*args, **kwargs)
        options = {**kwargs, 'num_ctx': OLLAMA_OPTIMIZER_CONTEXT, 'think': False,
                   'max_tokens': 2048}
        offset = log_path.stat().st_size if log_path.exists() else 0
        try:
            return original(*args, **options)
        finally:
            if log_path.exists():
                with log_path.open('rb') as log:
                    log.seek(offset)
                    new_log = log.read()
                if b'truncating input prompt' in new_log:
                    truncated = True
                    raise RuntimeError('Ollama truncated optimizer input; this response is invalid.')

    litellm.completion = complete
    for name, module in tuple(sys.modules.items()):
        if name.startswith('docetl.') and getattr(module, 'completion', None) is original:
            module.completion = complete
    try:
        yield
        if truncated:
            raise RuntimeError('Ollama truncated optimizer input; search did not complete safely.')
    finally:
        if litellm.completion is complete:
            litellm.completion = original
        for name, module in tuple(sys.modules.items()):
            if name.startswith('docetl.') and getattr(module, 'completion', None) is complete:
                module.completion = original


@contextlib.contextmanager
def classroom_search_actions() -> Any:
    """Exclude the GPT-dependent cascade only during this lab's optimizer run."""
    from unittest.mock import patch
    from docetl.moar import optimizer
    actions = [action for action in optimizer.ALL_DIRECTIVES
               if action.name not in DISABLED_SEARCH_DIRECTIVES]
    with patch.object(optimizer, 'ALL_DIRECTIVES', actions):
        yield


def score_sectors(rows: list[dict], reference: dict) -> dict:
    """Compare six document decisions; omitted records mean excluded."""
    expected = {row['doc_id']: row['sector'] if row['keep'] else None
                for row in reference['documents']}
    sectors = {row['sector'] for row in reference['expected_join_pairs']}
    indexed = _index(rows, 'doc_id', set(expected))
    for row in indexed.values():
        if not isinstance(row.get('sector'), str) or row['sector'] not in sectors:
            raise ValueError('Each returned document needs one valid sector.')
    decisions = [{'doc_id': key, 'expected': expected[key],
                  'actual': indexed[key]['sector'] if key in indexed else None,
                  'correct': expected[key] == (indexed[key]['sector'] if key in indexed else None)}
                 for key in sorted(expected)]
    correct = sum(row['correct'] for row in decisions)
    return {'accuracy': correct / len(decisions), 'correct': correct,
            'total': len(decisions), 'decisions': decisions}


def sector_counts(rows: list[dict], reference: dict) -> list[dict]:
    """Count validated document rows, including zero-count sectors."""
    score_sectors(rows, reference)
    return [{'sector': sector, 'company_count': sum(row['sector'] == sector for row in rows)}
            for sector in sorted({row['sector'] for row in reference['expected_join_pairs']})]


def show_sector_result(rows: list[dict], reference: dict) -> None:
    """Display the prediction for each document before totals and accuracy."""
    score = score_sectors(rows, reference)
    show_rows([{**row, 'expected': row['expected'] or 'Excluded',
                'actual': row['actual'] or 'Excluded'} for row in score['decisions']],
              ['doc_id', 'expected', 'actual', 'correct'])
    print(f"Accuracy: {score['correct']}/{score['total']} = {score['accuracy']:.1%}")
    show_rows(sector_counts(rows, reference), ['sector', 'company_count'])


def collect_measured(frame: Any) -> tuple[list[dict], dict]:
    """Collect once and retain SDK usage, distinguishing Frame result reuse."""
    reused = getattr(frame, '_memo', None) is not None
    started = time.perf_counter()
    rows = frame.collect()
    metrics = {
        'execution_seconds': time.perf_counter() - started,
        'execution_cost_estimate_usd': None if reused else _positive_cost(frame.total_cost),
        'token_usage': None if reused else copy.deepcopy(frame.token_usage),
        'cache': 'Frame result reused; no new usage measured' if reused else 'Query cache allowed',
    }
    return rows, metrics


def show_query_work(plans: list[dict]) -> None:
    """Show recorded execution work without inventing missing usage or prices."""
    from lab_support import measured_tokens
    def known(value: Any) -> Any:
        return value if value is not None else 'Unknown'
    show_rows([{
        'plan': row['id'],
        'execution_seconds': known(row.get('execution_seconds')),
        'reported_input_tokens': known(measured_tokens(row.get('token_usage'), 'prompt_tokens')),
        'reported_output_tokens': known(measured_tokens(row.get('token_usage'), 'completion_tokens')),
        'execution_cost_estimate_usd': known(row.get('execution_cost_estimate_usd')),
        'cache': row.get('cache', 'Unknown'),
    } for row in plans], ['plan', 'execution_seconds', 'reported_input_tokens',
                         'reported_output_tokens', 'execution_cost_estimate_usd', 'cache'])


def show_scoring_reference(course_dir: Path) -> None:
    """Open bundled scoring notes and reference answers inside Colab."""
    from IPython.display import HTML, display
    from lab_support import text_details
    for title, path in [
        ('Scoring reference', course_dir / 'document_scoring.md'),
        ('Reference answers and source quotes', course_dir / 'document_data' / 'golden_answers.json'),
    ]:
        display(HTML(text_details(title, path.read_text())))


def choose_candidate(plans: list[dict]) -> dict:
    """Keep the original on ties; accept only a strictly higher valid accuracy."""
    baseline = next(row for row in plans if row['id'] == 'original')
    if baseline.get('rate') is None or baseline.get('error'):
        raise ValueError('The original query has no valid score.')
    best = baseline
    for row in plans:
        rate = row.get('rate')
        if (not row.get('error') and row.get('changed', False)
                and isinstance(rate, (float, int)) and not isinstance(rate, bool)
                and math.isfinite(rate) and 0 <= rate <= 1 and rate > best['rate']):
            best = row
    return best


def _task_config(config: dict) -> dict:
    return {'default_model': config.get('default_model'), 'operations': config['operations'],
            'steps': config['pipeline']['steps']}


def _freeze_query(query: Any, documents: list[dict], folder: Path, model: str) -> tuple[dict, Path]:
    """Freeze only source records, never reference labels, for all trials."""
    import yaml
    config = yaml.safe_load(query.to_yaml())
    if len(config['datasets']) != 1 or [op['type'] for op in config['operations']] != ['filter', 'map']:
        raise ValueError('Optimize the filter-plus-sector-map query.')
    if len(documents) != 6 or any(set(row) != {'doc_id', 'text'} for row in documents):
        raise ValueError('Optimization inputs must be the six original document records.')
    _index(documents, 'doc_id', {row['doc_id'] for row in documents})
    data_path = folder / 'inputs.json'
    data_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2) + '\n')
    name = next(iter(config['datasets']))
    config['datasets'][name] = {'type': 'file', 'path': str(data_path)}
    config['default_model'] = model
    config['bypass_cache'] = True
    config['pipeline']['output'] = {'type': 'file', 'path': str(folder / 'output.json')}
    config['pipeline'].pop('intermediate_dir', None)
    return config, data_path


def run_optimization(query: Any, documents: list[dict], baseline_rows: list[dict],
                     reference: dict, model: str, run_dir: Path, *,
                     enabled: bool, in_colab: bool, prompt_only: bool = False,
                     allow_local_embeddings: bool = False,
                     baseline_work: dict | None = None) -> dict | None:
    """Search using deterministic document accuracy, with no judge calls."""
    if not enabled:
        return None
    if not in_colab:
        raise RuntimeError('Use a fresh Colab runtime; generated candidate code may execute.')
    input_hash = hashlib.sha256((json.dumps(documents, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
    if input_hash != reference['input_files']['documents.json']:
        raise ValueError('Optimization inputs differ from the source-checked document set.')
    ids = {row['doc_id'] for row in reference['documents']}
    if set(_index(documents, 'doc_id', ids)) != ids:
        raise ValueError('Supply all six source documents, not just the filtered rows.')
    import docetl
    import yaml
    from unittest.mock import patch
    from docetl.moar.Node import Node
    from lab_support import make_json_adapter, moar_response_format
    folder = (run_dir / ('sector-prompt-case' if prompt_only else 'sector-moar') / uuid.uuid4().hex[:8]).resolve()
    folder.mkdir(parents=True)
    original, data_path = _freeze_query(query, documents, folder, model)
    baseline_score = score_sectors(baseline_rows, reference)
    plans = [{'id': 'original', 'config': original, 'rows': copy.deepcopy(baseline_rows),
              'score': baseline_score, 'rate': baseline_score['accuracy'], 'changed': False,
              'execution_seconds': None, 'execution_cost_estimate_usd': None}]
    for key in ('execution_seconds', 'execution_cost_estimate_usd', 'token_usage', 'cache'):
        if baseline_work is not None and key in baseline_work:
            plans[0][key] = copy.deepcopy(baseline_work[key])
    record = {'format': SEARCH_FORMAT,
              'kind': 'separate_prompt_case' if prompt_only else 'moar_search',
              'model': model, 'input_sha256': input_hash, 'metric': 'accuracy',
              'plans': plans, 'status': 'running', 'selected_id': None,
              'cache': 'bypassed for trial executions', 'search_seconds': None,
              'sdk_search_cost_estimate_usd': None, 'pricing_verified': False}
    record['local_embedding_model'] = LOCAL_EMBEDDING_MODEL if allow_local_embeddings else None
    record['disabled_search_directives'] = list(DISABLED_SEARCH_DIRECTIVES)
    config_path = folder / 'original.yaml'
    config_path.write_text(yaml.safe_dump(original, sort_keys=False))
    executions: list[dict] = []
    by_output: dict[str, dict] = {}
    execute = Node.execute_plan

    def guarded_execute(node: Any, max_threads: int | None = None) -> float:
        config = node.parsed_yaml
        entry = {'id': f'trial-{len(executions) + 1}', 'config': copy.deepcopy(config),
                 'changed': _task_config(config) != _task_config(original),
                 'execution_seconds': None, 'execution_cost_estimate_usd': None,
                 'cache': 'Query cache bypassed'}
        executions.append(entry)
        started = None
        try:
            _check_candidate(config, original, folder, model,
                             allow_local_embeddings=allow_local_embeddings)
            output_path = str(Path(config['pipeline']['output']['path']).resolve())
            by_output[output_path] = entry
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != input_hash:
                raise ValueError('The frozen trial data changed.')
            started = time.perf_counter()
            cost = execute(node, max_threads=1)
            if cost == -1:
                raise RuntimeError('Candidate execution failed; no score is available.')
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != input_hash:
                raise ValueError('A candidate changed the frozen trial data.')
            entry['execution_cost_estimate_usd'] = _positive_cost(cost)
            return cost
        except Exception as exc:
            entry.update(error=type(exc).__name__, rate=None)
            if isinstance(exc, CandidateNotExecuted):
                entry.update(status='not_executed' if started is None else 'failed', reason=str(exc))
            raise
        finally:
            if started is not None:
                entry['execution_seconds'] = time.perf_counter() - started

    def evaluate_candidate(results_path: str) -> dict:
        key = str(Path(results_path).resolve())
        if key not in by_output or by_output[key].get('error'):
            raise ValueError('Candidate did not complete a checked execution.')
        entry = by_output[key]
        try:
            rows = json.loads(Path(key).read_text())
            entry['rows'] = rows
            score = score_sectors(rows, reference)
            entry.update(score=score, rate=score['accuracy'])
            return {'accuracy': score['accuracy']}
        except (ValueError, OSError) as exc:
            entry.update(error=type(exc).__name__, rate=None)
            raise

    started = time.perf_counter()
    try:
        def adapter_factory(original: Any, base_model: type, stats: dict) -> Any:
            return make_json_adapter(_moar_tool_options(original), base_model, stats)
        with classroom_search_actions(), \
                local_embedding_support(allow_local_embeddings, in_colab, folder), \
                optimizer_context(model, run_dir), \
                patch('lab_support.make_json_adapter', adapter_factory), \
                moar_response_format(model) as adaptation, \
                patch.object(Node, 'execute_plan', guarded_execute):
            if prompt_only:
                from docetl.reasoning_optimizer.directives.clarify_instructions import ClarifyInstructionsDirective
                directive = ClarifyInstructionsDirective()
                changed_ops, _, rewrite_cost = directive.instantiate(
                    operators=copy.deepcopy(original['operations']),
                    target_ops=[original['operations'][0]['name']], agent_llm=model,
                    message_history=[], global_default_model=model,
                    input_file_path=str(data_path), pipeline_code=original)
                candidate = copy.deepcopy(original)
                candidate['operations'] = changed_ops
                candidate['pipeline']['output']['path'] = str(folder / 'prompt-output.json')
                candidate_path = folder / 'prompt-rewritten.yaml'
                candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
                node = Node(str(candidate_path))
                node.execute_plan(max_threads=1)
                evaluate_candidate(candidate['pipeline']['output']['path'])
                record['rewrite_cost_estimate_usd'] = _positive_cost(rewrite_cost)
            else:
                optimized = docetl.Frame.from_yaml(str(config_path)).optimize(
                    eval_fn=evaluate_candidate, metric_key='accuracy',
                    models=[model], agent_model=model,
                    max_iterations=2, max_concurrent_agents=1, max_threads=1,
                    dataset_path=str(data_path), save_dir=str(folder))
                record['sdk_search_cost_estimate_usd'] = _positive_cost(
                    optimized.search_results.total_search_cost)
            record['provider_adaptation'] = adaptation
        if not any(row.get('rate') is not None and not row.get('error') for row in executions):
            raise RuntimeError('Search produced no valid trial output. Inspect search.json.')
        plans.extend(executions)
        selected = choose_candidate(plans)
        record.update(status='completed', selected_id=selected['id'])
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        if len(plans) == 1:
            plans.extend(executions)
        record['search_seconds'] = time.perf_counter() - started
        _save_record(folder / 'search.json', record)
    return record


def _check_search_record(record: dict) -> None:
    if record.get('format') != SEARCH_FORMAT:
        raise ValueError('This is not a sector-query search record; old extraction scores cannot be reused.')


def selected_rows(record: dict) -> list[dict]:
    """Reuse the selected sector rows without making more model calls."""
    _check_search_record(record)
    if record.get('status') != 'completed':
        raise ValueError('There is no completed selection to display.')
    return copy.deepcopy(next(row['rows'] for row in record['plans'] if row['id'] == record['selected_id']))


def show_search(record: dict | None, documents: list[dict]) -> None:
    """Show one actual change and its document decisions before the score table."""
    if record is None:
        print('No search run yet. Enable RUN_MOAR to try the optimizer.')
        return
    _check_search_record(record)
    from IPython.display import HTML, display
    from lab_support import operation_changes, text_details
    import difflib
    import yaml
    print(f"Run: {record['kind']}; status: {record['status']}")
    original = record['plans'][0]
    candidates = [row for row in record['plans'][1:] if row.get('changed') and row.get('rate') is not None
                  and not row.get('error')]
    if candidates:
        candidate = candidates[0]
        print(f"First comparable change: {candidate['id']} (execution order, not score order).")
        change = operation_changes(original['config'], candidate['config'])
        if change:
            display(HTML(text_details('Operation change', change, expanded=True)))
        else:
            show_rows([{'version': plan['id'], 'operations': ' -> '.join(
                op['type'] for op in plan['config']['operations'])}
                for plan in [original, candidate]], ['version', 'operations'])
        diff = ''.join(difflib.unified_diff(
            yaml.safe_dump(original['config'], sort_keys=False).splitlines(True),
            yaml.safe_dump(candidate['config'], sort_keys=False).splitlines(True),
            fromfile='original', tofile=candidate['id']))
        display(HTML(text_details('Full configuration diff', diff)))
        before = {row['doc_id']: row['sector'] for row in original['rows']}
        after = {row['doc_id']: row['sector'] for row in candidate['rows']}
        changed = [row['doc_id'] for row in documents if before.get(row['doc_id']) != after.get(row['doc_id'])]
        if changed:
            key = changed[0]
            show_document(documents, key)
            show_rows([{'version': plan['id'], **decision,
                        'expected': decision['expected'] or 'Excluded',
                        'actual': decision['actual'] or 'Excluded'}
                       for plan in [original, candidate] for decision in plan['score']['decisions']
                       if decision['doc_id'] == key], ['version', 'doc_id', 'expected', 'actual', 'correct'])
        else:
            print('The query changed, but all six document decisions stayed the same.')
    else:
        print('No changed candidate with a valid score was recorded.')
    show_rows([{'plan': row['id'],
                'correct_documents': f"{row['score']['correct']}/{row['score']['total']}" if row.get('score') and not row.get('error') else 'Unavailable',
                'accuracy': row.get('rate') if row.get('rate') is not None else 'Unavailable',
                'selected': row['id'] == record['selected_id'],
                'status': 'Not executed' if row.get('status') == 'not_executed' else 'Failed' if row.get('error') else 'Scored' if row.get('rate') is not None else 'Not scored'}
               for row in record['plans']], ['plan', 'correct_documents', 'accuracy', 'selected', 'status'])
    print('Work recorded for each query execution')
    show_query_work(record['plans'])
    details = [{'plan': row['id'], 'change': 'original' if row['id'] == 'original' else
                'changed query' if row.get('changed') else 'unchanged query rerun',
                'error': row.get('error', ''), 'reason': row.get('reason', '')} for row in record['plans']]
    display(HTML(text_details('Run details and errors', json.dumps(
        {'status': record['status'], 'error': record.get('error'), 'plans': details}, ensure_ascii=False, indent=2))))
    print('Search overhead is separate from the cost of executing one query.')


def plot_search(record: dict | None) -> Any:
    """Plot measured document accuracy, with no invented frontier or costs."""
    if record is None:
        return None
    _check_search_record(record)
    import matplotlib.pyplot as plt
    rows = [row for row in record['plans'] if row.get('rate') is not None and not row.get('error')]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([row['id'] for row in rows], [row['rate'] for row in rows], color='#247d78')
    ax.set(ylabel='Document accuracy', ylim=(0, 1.1))
    for i, row in enumerate(rows):
        ax.text(i, row['rate'] + .025, f'{row["rate"]:.0%}', ha='center')
    fig.tight_layout()
    return fig

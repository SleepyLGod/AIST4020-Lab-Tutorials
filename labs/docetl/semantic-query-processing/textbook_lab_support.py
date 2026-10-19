"""Display, scoring and optional MOAR execution for the six-review example."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch
import uuid


def score(rows: list[dict], reference: dict[str, str | None]) -> dict:
    """Score each input once; omitted rows mean the filter excluded them."""
    if not isinstance(rows, list):
        raise ValueError('Query output must be a list.')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('doc_id') not in reference:
            raise ValueError('Unknown or missing review ID.')
        key = row['doc_id']
        if key in indexed:
            raise ValueError(f'Duplicate review ID: {key}')
        if row.get('topic') not in {'Food', 'Service'}:
            raise ValueError(f'Invalid or missing topic for {key}.')
        indexed[key] = row['topic']
    decisions = [{'doc_id': key, 'expected': value, 'actual': indexed.get(key),
                  'correct': value == indexed.get(key)} for key, value in reference.items()]
    correct = sum(row['correct'] for row in decisions)
    return {'accuracy': correct / len(reference), 'correct': correct,
            'total': len(reference), 'decisions': decisions}


def show_decisions(rows: list[dict], reference: dict[str, str | None]) -> None:
    """Display excluded reviews explicitly without changing the score."""
    from document_lab_support import show_rows
    decisions = score(rows, reference)['decisions']
    display_rows = [dict(row, expected='Excluded' if row['expected'] is None else row['expected'],
                         actual='Excluded' if row['actual'] is None else row['actual'])
                    for row in decisions]
    show_rows(display_rows, ['doc_id', 'expected', 'actual', 'correct'])


def compare(plans: list[dict], reference: dict) -> Any:
    """Show current query results and measured work, without new requests."""
    import document_lab_support as support
    import matplotlib.pyplot as plt
    table = []
    for plan in plans:
        result = score(plan['rows'], reference)
        table.append({'query': plan['id'], 'correct': f"{result['correct']}/{result['total']}",
                      'accuracy': result['accuracy']})
    support.show_rows(table, ['query', 'correct', 'accuracy'])
    support.show_query_work(plans)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.bar([r['query'] for r in table], [r['accuracy'] for r in table], color=['#247d78', '#b05c40'])
    ax.set(ylabel='Accuracy', ylim=(0, 1.12))
    for i, row in enumerate(table):
        ax.text(i, row['accuracy'] + .025, row['correct'], ha='center')
    fig.tight_layout()
    return fig


def freeze(query: Any, reviews: list[dict], model: str, directory: Path) -> tuple[dict, Path]:
    """Write only inputs and the selected query, never reference labels."""
    import yaml
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / 'inputs.json'
    data_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n')
    config = yaml.safe_load(query.to_yaml())
    if len(config['datasets']) != 1:
        raise ValueError('Expected one input dataset.')
    name = next(iter(config['datasets']))
    config['datasets'][name] = {'type': 'file', 'path': str(data_path)}
    config['default_model'] = model
    config['bypass_cache'] = True
    config['pipeline'].pop('intermediate_dir', None)
    config['pipeline']['output'] = {'type': 'file', 'path': str(directory / 'output.json')}
    return config, data_path


def collect_fresh(query: Any, reviews: list[dict], model: str, directory: Path) -> tuple[list[dict], dict]:
    """Execute a comparison query without reusing cached model answers."""
    import docetl
    import yaml
    from document_lab_support import collect_measured
    config, _ = freeze(query, reviews, model, directory.resolve())
    path = directory / 'query.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    rows, work = collect_measured(docetl.Frame.from_yaml(str(path)))
    work['cache'] = 'Query cache bypassed'
    return rows, work


def run_search(query: Any, reviews: list[dict], reference: dict, model: str,
               run_dir: Path, *, enabled: bool = False, prompt_only: bool = False) -> dict | None:
    """Run a small real MOAR search in Colab; retain failed candidates as failures."""
    if not enabled:
        return None
    import sys
    if 'google.colab' not in sys.modules:
        raise RuntimeError('Run optional optimization in a dedicated Colab runtime.')
    import docetl
    import yaml
    import docetl.moar.optimizer as optimizer
    from docetl.moar.Node import Node
    from lab_support import make_json_adapter, moar_response_format, check_record_secrets
    from document_lab_support import _check_candidate, _moar_tool_options, optimizer_context

    folder = (run_dir / ('prompt' if prompt_only else 'search') / uuid.uuid4().hex[:8]).resolve()
    original, data_path = freeze(query, reviews, model, folder)
    input_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    config_path = folder / 'original.yaml'
    config_path.write_text(yaml.safe_dump(original, sort_keys=False))
    record = {'format': 'docetl-textbook-search-v1', 'kind': 'prompt_rewriting' if prompt_only else 'moar_search',
              'model': model, 'input_sha256': input_hash, 'plans': [], 'status': 'running',
              'strategies': ['clarify_instructions', 'operator_fusion'], 'search_seconds': None}
    by_output = {}
    execute = Node.execute_plan

    def guarded(node: Any, max_threads: int | None = None) -> float:
        config = node.parsed_yaml
        entry = {'id': f"trial-{len(record['plans'])+1}", 'config': copy.deepcopy(config)}
        record['plans'].append(entry)
        start = time.perf_counter()
        try:
            _check_candidate(config, original, folder, model)
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != input_hash:
                raise ValueError('Input records changed.')
            output_path = str(Path(config['pipeline']['output']['path']).resolve())
            by_output[output_path] = entry
            cost = execute(node, max_threads=1)
            if cost == -1:
                raise RuntimeError('Candidate failed to execute.')
            entry['execution_cost_estimate_usd'] = cost if cost > 0 else None
            return cost
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            entry['execution_seconds'] = time.perf_counter() - start

    def evaluate(path: str) -> dict:
        entry = by_output[str(Path(path).resolve())]
        if entry.get('error'):
            raise ValueError('Failed execution is not an empty result.')
        rows = json.loads(Path(path).read_text())
        result = score(rows, reference)
        entry.update(rows=rows, score=result)
        return {'accuracy': result['accuracy']}

    def adapter(original_call: Any, base_model: type, stats: dict) -> Any:
        return make_json_adapter(_moar_tool_options(original_call), base_model, stats)

    allowed = [d for d in optimizer.ALL_DIRECTIVES if d.name in record['strategies']]
    if len(allowed) != 2:
        raise ValueError('Pinned MOAR strategies are unavailable.')
    started = time.perf_counter()
    try:
        with patch.object(optimizer, 'ALL_DIRECTIVES', allowed), optimizer_context(model, run_dir), \
                patch('lab_support.make_json_adapter', adapter), moar_response_format(model), \
                patch.object(Node, 'execute_plan', guarded):
            if prompt_only:
                from docetl.reasoning_optimizer.directives.clarify_instructions import ClarifyInstructionsDirective
                operations, _, cost = ClarifyInstructionsDirective().instantiate(
                    operators=copy.deepcopy(original['operations']), target_ops=[original['operations'][0]['name']],
                    agent_llm=model, message_history=[], global_default_model=model,
                    input_file_path=str(data_path), pipeline_code=original)
                candidate = copy.deepcopy(original)
                candidate['operations'] = operations
                path = folder / 'rewritten.yaml'
                path.write_text(yaml.safe_dump(candidate, sort_keys=False))
                Node(str(path)).execute_plan(max_threads=1)
                evaluate(candidate['pipeline']['output']['path'])
                record['rewrite_cost_estimate_usd'] = cost if cost > 0 else None
            else:
                result = docetl.Frame.from_yaml(str(config_path)).optimize(
                    eval_fn=evaluate, metric_key='accuracy', models=[model], agent_model=model,
                    max_iterations=2, max_concurrent_agents=1, max_threads=1,
                    dataset_path=str(data_path), save_dir=str(folder))
                record['sdk_search_cost_estimate_usd'] = result.search_results.total_search_cost
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['search_seconds'] = time.perf_counter() - started
        check_record_secrets(record)
        (folder / 'record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record

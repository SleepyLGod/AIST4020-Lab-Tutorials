"""Display, scoring and optional MOAR execution for the six-review example."""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
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


def show_query_plans() -> Any:
    """Draw the two query structures, not measured execution results."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 11), ylim=(0, 4.4))
    ax.axis('off')
    rows = [
        (2.45, 'Original query',
         [('filter', 'Is it a complaint?', 'Model'),
          ('map', 'Food or Service?', 'Model')]),
        (0.35, 'Saved MOAR rewrite',
         [('map', 'Return keep and topic', 'Model'),
          ('code_filter', 'Keep rows with keep=True', 'Python')]),
    ]
    for y, title, operations in rows:
        ax.text(0.15, y + 1.5, title, fontsize=13, weight='bold', color='#222222')
        ax.text(0.75, y + 0.65, 'Reviews', ha='center', va='center', fontsize=12)
        for x, (operator, question, executor) in zip((2.2, 6.65), operations):
            color = '#e4f2f0' if executor == 'Model' else '#fff0d6'
            ax.add_patch(Rectangle((x, y), 3.65, 1.3,
                                   facecolor=color, edgecolor='#777777', linewidth=1))
            ax.text(x + 1.825, y + 1.02, operator, ha='center',
                    fontsize=12, weight='bold')
            ax.text(x + 1.825, y + 0.62, question, ha='center', fontsize=11)
            ax.text(x + 1.825, y + 0.22, executor, ha='center',
                    fontsize=10, color='#444444')
        for left, right in ((1.45, 2.15), (5.9, 6.6)):
            ax.add_patch(FancyArrowPatch((left, y + 0.65), (right, y + 0.65),
                                        arrowstyle='-|>', mutation_scale=15,
                                        color='#444444', linewidth=1.5))
    fig.tight_layout(pad=0.5)
    return fig


def filter_prompt(query: Any) -> str:
    """Read the filter prompt from the query that will actually run."""
    import yaml
    config = yaml.safe_load(query.to_yaml())
    filters = [op for op in config['operations'] if op['type'] == 'filter']
    if len(filters) != 1 or not isinstance(filters[0].get('prompt'), str):
        raise ValueError('Expected one filter with a text prompt.')
    return filters[0]['prompt']


def compare(plans: list[dict], reference: dict) -> Any:
    """Compare two current executions, without requesting model answers."""
    import document_lab_support as support
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    from lab_support import measured_tokens

    if len(plans) != 2:
        raise ValueError('Compare the original query with one rewrite.')
    results = [score(plan['rows'], reference) for plan in plans]
    names = [str(plan['id']) for plan in plans]

    def label(value: str | None) -> str:
        return 'Excluded' if value is None else value

    body = []
    for left, right in zip(results[0]['decisions'], results[1]['decisions']):
        changed = left['actual'] != right['actual']
        cells = [left['doc_id'], label(left['expected']), label(left['actual']),
                 label(right['actual']), 'Changed' if changed else 'Same']
        style = ' style="background:#fff1c2;color:#202020"' if changed else ''
        body.append(f'<tr{style}>' + ''.join(
            f'<td style="padding:4px 10px">{html.escape(str(value))}</td>' for value in cells
        ) + '</tr>')
    headings = ['Review', 'Reference', *names, 'Decision']
    display(HTML('<table><thead><tr>' + ''.join(
        f'<th>{html.escape(value)}</th>' for value in headings
    ) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'))
    if all(a['actual'] == b['actual'] for a, b in
           zip(results[0]['decisions'], results[1]['decisions'])):
        print('Both queries made the same decisions.')

    def number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    inputs = [measured_tokens(p.get('token_usage'), 'prompt_tokens') for p in plans]
    outputs = [measured_tokens(p.get('token_usage'), 'completion_tokens') for p in plans]
    times = [number(p.get('execution_seconds')) for p in plans]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    colors = ['#247d78', '#b05c40']
    for i, result in enumerate(results):
        axes[0].bar(i, result['accuracy'], color=colors[i])
        axes[0].text(i, result['accuracy'] + .025,
                     f"{result['correct']}/{result['total']}", ha='center')
    axes[0].set(title='Accuracy', ylabel='Correct / all reviews', ylim=(0, 1.15))

    for values, offset, color, caption in [
        (inputs, -.2, '#247d78', 'Input'), (outputs, .2, '#b05c40', 'Output')
    ]:
        for i, value in enumerate(values):
            if value is None:
                axes[1].text(i + offset, .04, 'Unknown', rotation=90, ha='center',
                             va='bottom', transform=axes[1].get_xaxis_transform())
            else:
                axes[1].bar(i + offset, value, width=.36, color=color)
                axes[1].annotate(f'{value:g}', (i + offset, value),
                                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        axes[1].plot([], [], color=color, linewidth=6, label=caption)
    axes[1].set(title='Model tokens', ylabel='Tokens')
    axes[1].legend(fontsize=8)

    for i, value in enumerate(times):
        if value is None:
            axes[2].text(i, .04, 'Unknown', ha='center',
                         transform=axes[2].get_xaxis_transform())
        else:
            axes[2].bar(i, value, color=colors[i])
            axes[2].annotate(f'{value:.2f}', (i, value), xytext=(0, 3),
                             textcoords='offset points', ha='center')
    axes[2].set(title='Query time', ylabel='Seconds')
    for ax in axes:
        ax.set_xticks([0, 1], names)
        ax.set_xlim(-.6, 1.6)
    for ax in axes[1:]:
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1) * 1.2)
    fig.tight_layout()

    details = [{
        'Query': names[i], 'Seconds': times[i], 'Input tokens': inputs[i],
        'Output tokens': outputs[i],
        'Estimated USD': number(plan.get('execution_cost_estimate_usd')),
        'Cache': plan.get('cache'),
    } for i, plan in enumerate(plans)]
    details = [{key: 'Unknown' if value is None else value for key, value in row.items()}
               for row in details]
    display(HTML(
        '<details><summary>Measurement details</summary>'
        '<p>Tokens measure model usage, not dollars. Prices depend on the model; '
        'USD values are SDK estimates. Query time includes provider response time. '
        'These measurements cover query execution, not the earlier MOAR search. '
        'Unknown values are left unplotted.</p>'
        + support.table_html(details, list(details[0])) + '</details>'
    ))
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

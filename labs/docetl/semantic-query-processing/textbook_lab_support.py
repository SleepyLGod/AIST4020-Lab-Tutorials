"""Plotting and optional runtime compatibility for the restaurant-review lab."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
import subprocess
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch
import uuid


def require_colab_gpu(backend: str, in_colab: bool) -> None:
    """Check the default Colab setup before downloading a local chat model."""
    if backend != 'ollama' or not in_colab:
        return
    message = (
        'For Ollama in Colab, select Runtime > Change runtime type > T4 GPU, '
        'then rerun the notebook from the top.'
    )
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(message) from exc
    if not result.stdout.strip():
        raise RuntimeError(message)


def plot_comparison(metrics: list[dict]) -> Any:
    """Plot already-computed measurements without running or scoring queries."""
    import matplotlib.pyplot as plt

    names = [row['query'] for row in metrics]
    colors = ['#247d78', '#b05c40']
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    def bar(axis: Any, x: float, value: Any, color: str, width: float = .55) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            axis.text(x, .04, 'Unknown', ha='center',
                      transform=axis.get_xaxis_transform(), fontsize=8)
            return
        axis.bar(x, value, color=color, width=width)
        axis.annotate(f'{value:.3g}', (x, value), xytext=(0, 4),
                      textcoords='offset points', ha='center', fontsize=9)

    for i, row in enumerate(metrics):
        bar(axes[0], i, row['accuracy'], colors[i % len(colors)])
        bar(axes[1], i - .2, row.get('input_tokens'), colors[0], .36)
        bar(axes[1], i + .2, row.get('output_tokens'), colors[1], .36)
        bar(axes[2], i, row.get('seconds'), colors[i % len(colors)])
    axes[0].set(title='Accuracy', ylabel='Correct / all reviews', ylim=(0, 1.15))
    axes[1].set(title='Model tokens', ylabel='Tokens')
    for color, caption in zip(colors, ['Input', 'Output']):
        axes[1].plot([], [], color=color, linewidth=6, label=caption)
    axes[1].legend(fontsize=8)
    axes[2].set(title='Query time', ylabel='Seconds')
    for axis in axes:
        axis.set_xticks(range(len(names)), names)
        axis.set_xlim(-.6, len(names) - .4)
    for axis in axes[1:]:
        axis.set_ylim(bottom=0, top=max(axis.get_ylim()[1], 1) * 1.2)
    figure.tight_layout()
    return figure


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


@contextmanager
def optimization_environment(query: Any, reviews: list[dict], model: str,
                             run_dir: Path) -> Iterator[dict[str, str]]:
    """Prepare search files and provider guards; the caller runs optimize()."""
    import sys
    if 'google.colab' not in sys.modules:
        raise RuntimeError('Run optional optimization in a dedicated Colab runtime.')
    import yaml
    import docetl.moar.optimizer as optimizer
    from docetl.moar.Node import Node
    from lab_support import make_json_adapter, moar_response_format, check_record_secrets
    from document_lab_support import _check_candidate, _moar_tool_options, optimizer_context

    folder = (run_dir / 'search' / uuid.uuid4().hex[:8]).resolve()
    original, data_path = freeze(query, reviews, model, folder)
    input_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    config_path = folder / 'original.yaml'
    config_path.write_text(yaml.safe_dump(original, sort_keys=False))
    record = {'format': 'docetl-textbook-search-v3', 'kind': 'moar_search',
              'model': model, 'input_sha256': input_hash, 'plans': [], 'status': 'running',
              'strategies': ['clarify_instructions', 'operator_fusion'], 'search_seconds': None}
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
            yield {'query_path': str(config_path), 'dataset_path': str(data_path),
                   'save_dir': str(folder)}
        record['status'] = 'finished'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['search_seconds'] = time.perf_counter() - started
        check_record_secrets(record)
        (folder / 'record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))

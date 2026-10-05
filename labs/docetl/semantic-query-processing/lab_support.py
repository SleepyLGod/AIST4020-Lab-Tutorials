"""Setup, validation, and recording for the movie-review teaching notebook.

Importing this module does not install packages, read keys, or call a model.
Queries, prompts, pairing, and scoring remain in the notebook.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import getpass
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterator
import urllib.error
import urllib.request
import uuid
import zipfile

INPUT_FIELDS = {'id', 'reviewId', 'reviewText'}
LABELS = ('POSITIVE', 'NEGATIVE')
DATA_FILES = {f'{split}{suffix}.json' for split in ('demo', 'optimization', 'test')
              for suffix in ('', '_labels')}
MOAR_RECORD_FORMAT = 'docetl-movie-lab-moar-v1'



EXPECTED_MANIFEST_SHA256 = 'fd7271541c3228473f397fa4c941297de4ae5506110426a97791a0a84efd4ced'


def unpack_course_data(blob: bytes, directory: Path) -> None:
    """Check the fixed course bundle before writing its seven named files."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        if len(names) != 7 or set(names) != DATA_FILES | {'manifest.json'}:
            raise ValueError('This is not the expected course ZIP.')
        if sum(item.file_size for item in archive.infolist()) > 5_000_000:
            raise ValueError('The course ZIP is unexpectedly large.')
        payloads = {name: archive.read(name) for name in names}
    if hashlib.sha256(payloads['manifest.json']).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError('The data manifest differs from the version used by this notebook.')
    manifest = json.loads(payloads['manifest.json'])
    for name in DATA_FILES:
        if hashlib.sha256(payloads[name]).hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'Data checksum mismatch: {name}')
    for name, payload in payloads.items():
        path = directory / name
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f'Different existing file; not overwritten: {path}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(payload)


def review_key(row: dict) -> tuple[str, str]:
    """Identify a review without relying on output order."""
    values = (row.get('id'), row.get('reviewId'))
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError('Every review needs a nonempty movie ID and review ID.')
    return values


def index_inputs(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Require only the three model-input fields and unique review IDs."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of reviews.')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != INPUT_FIELDS:
            raise ValueError('Model input must contain only id, reviewId, and reviewText.')
        if not isinstance(row['reviewText'], str) or not row['reviewText'].strip():
            raise ValueError('Empty review text.')
        key = review_key(row)
        if key in indexed:
            raise ValueError(f'Duplicate input ID: {key}')
        indexed[key] = row
    return indexed


def validate_returned_rows(inputs: list[dict], outputs: list[dict]) -> tuple[dict, list]:
    """Reject duplicate, unknown, or changed rows and report missing input IDs."""
    expected = index_inputs(inputs)
    if not isinstance(outputs, list):
        raise ValueError('The query did not return a list of rows.')
    returned = {}
    for row in outputs:
        if not isinstance(row, dict):
            raise ValueError('A returned row is not a dictionary.')
        key = review_key(row)
        if key not in expected:
            raise ValueError(f'Unknown output ID: {key}')
        if key in returned:
            raise ValueError(f'Duplicate output ID: {key}')
        if row.get('reviewText') != expected[key]['reviewText']:
            raise ValueError(f'The review text changed: {key}')
        returned[key] = row
    return returned, sorted(set(expected) - set(returned))


def validate_labels(inputs: list[dict], outputs: list[dict]) -> dict:
    """Check that every original review has one allowed sentiment label."""
    returned, missing = validate_returned_rows(inputs, outputs)
    invalid = [key for key, row in returned.items() if row.get('sentiment') not in LABELS]
    return {'returned': returned, 'missing': missing, 'invalid': invalid,
            'complete': not missing and not invalid}


def canonical_label(value: Any) -> Any:
    """Normalize case and whitespace only; leave unsupported answers invalid."""
    return value.strip().upper() if isinstance(value, str) else value


def normalize_labels(rows: list[dict]) -> list[dict]:
    """Return label-normalized copies without changing saved model output."""
    return [{**row, 'sentiment': canonical_label(row.get('sentiment'))} for row in rows]


def project_candidate_labels(inputs: list[dict], outputs: list[dict]) -> list[dict]:
    """Join complete candidate labels to original reviews, including rewritten inputs."""
    expected = index_inputs(inputs)
    labels = {}
    if not isinstance(outputs, list):
        raise ValueError('Candidate output must be a list.')
    for row in outputs:
        if not isinstance(row, dict):
            raise ValueError('Candidate output contains a non-record value.')
        key = review_key(row)
        label = canonical_label(row.get('sentiment'))
        if key not in expected or key in labels or label not in LABELS:
            raise ValueError('Candidate has duplicate/unknown IDs or invalid labels.')
        labels[key] = label
    if set(labels) != set(expected):
        raise ValueError('Candidate is missing reviews; no partial pairing will be scored.')
    return [{**row, 'sentiment': labels[key]} for key, row in expected.items()]


def inspect_pair_call(record: dict, inputs: list[dict], returned_pairs: set) -> tuple:
    """Check that the recorded response belongs to one actual compared pair."""
    call = record.get('example_call')
    if not call or call.get('response') is None or call.get('parsed_output') is None:
        raise ValueError('No complete pair-call record. Inspect the query log.')
    index = index_inputs(inputs)
    keys = call.get('pair_keys', [])
    if len(keys) != 2 or any(tuple(key) not in index for key in keys):
        raise ValueError('The saved call does not identify its two input reviews.')
    pair = tuple(tuple(key) for key in keys)
    if pair[0] == pair[1] or pair[0][0] != pair[1][0]:
        raise ValueError('The recorded pair violates the candidate condition.')
    parsed = call['parsed_output']
    if len(parsed) != 1 or type(parsed[0].get('is_match')) is not bool:
        raise ValueError('Expected one Boolean pair judgment.')
    matched = parsed[0]['is_match']
    if (pair in returned_pairs) != matched:
        raise ValueError('Recorded judgment and returned pairs disagree.')
    texts = [m['content'] for m in call['request']['messages']
             if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    if any(not any(index[key]['reviewText'] in text for text in texts) for key in pair):
        raise ValueError('The saved question does not contain both original reviews.')
    return call, [index[key] for key in pair], texts, matched


def text_details(title: str, text: str, expanded: bool = False) -> str:
    """Wrap escaped text in an expandable notebook output."""
    import html
    state = ' open' if expanded else ''
    return (f'<details{state}><summary>{html.escape(title)}</summary>'
            f'<pre style="white-space:pre-wrap">{html.escape(text)}</pre></details>')


def cost_for_display(value: object) -> object:
    """Keep usable estimates visible and label unknown prices explicitly."""
    import math
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return value
    return 'Unknown'


def operation_changes(before: dict, after: dict) -> str | None:
    """Describe exact field changes only when operation identities still align."""
    old, new = before.get('operations', []), after.get('operations', [])
    names = [op.get('name') for op in old]
    if (not names or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or names != [op.get('name') for op in new]
            or any(a.get('type') != b.get('type') for a, b in zip(old, new))):
        return None
    changes = []
    for a, b in zip(old, new):
        for field in sorted(set(a) | set(b)):
            if (field in a) != (field in b) or a.get(field) != b.get(field):
                previous = (a[field] if isinstance(a[field], str) else json.dumps(a[field], ensure_ascii=False, indent=2)) if field in a else '[not present]'
                current = (b[field] if isinstance(b[field], str) else json.dumps(b[field], ensure_ascii=False, indent=2)) if field in b else '[not present]'
                changes.append(f"{a['name']} / {field}\nBefore: {previous}\nAfter: {current}")
    return '\n\n'.join(changes) or 'Operation definitions are unchanged. Check the full configuration below.'


def validate_pair_rows(inputs: list[dict], outputs: list[dict]) -> set:
    """Require original rows, distinct same-movie endpoints, and unique ordered pairs."""
    expected = index_inputs(inputs)
    if not isinstance(outputs, list):
        raise ValueError('Expected a list of joined rows.')
    pairs = set()
    for row in outputs:
        if not isinstance(row, dict):
            raise ValueError('A joined row is not a dictionary.')
        endpoints = []
        for side in ('left', 'right'):
            original = {field: row.get(f'{field}_{side}') for field in INPUT_FIELDS}
            key = review_key(original)
            if key not in expected or original != expected[key]:
                raise ValueError('A joined review is unknown or its text changed.')
            endpoints.append(key)
        left, right = endpoints
        if left == right or left[0] != right[0]:
            raise ValueError('A pair compares itself or contains different movies.')
        if (left, right) in pairs:
            raise ValueError('Duplicate ordered pair.')
        pairs.add((left, right))
    return pairs


def check_pair_judgments(candidates: set, judgments: list[dict], returned: set) -> None:
    """Check coverage and compare saved Boolean decisions with returned pairs."""
    seen = {}
    for item in judgments:
        pair = (tuple(item['left']), tuple(item['right']))
        if pair not in candidates or pair in seen or type(item['is_match']) is not bool:
            raise ValueError('Unknown, duplicate, or invalid pair judgment.')
        seen[pair] = item['is_match']
    if set(seen) != candidates:
        raise ValueError('Some candidate pairs have no saved judgment. Do not score this run.')
    if {pair for pair, match in seen.items() if match} != returned:
        raise ValueError('Saved judgments and returned pairs disagree.')


def checked_reference(inputs: list[dict], label_rows: list[dict]) -> dict:
    """Require exactly one allowed reference label for each input review."""
    expected = index_inputs(inputs)
    labels = {}
    for row in label_rows:
        key = review_key(row)
        if key in labels or key not in expected or row.get('scoreSentiment') not in LABELS:
            raise ValueError('Duplicate, unknown, or invalid reference label.')
        labels[key] = row['scoreSentiment']
    if set(labels) != set(expected):
        raise ValueError('Reference labels do not cover exactly the input reviews.')
    return labels


def checked_reported_cost(value: object) -> float | None:
    """Treat absent, zero, negative, or non-finite pricing as unknown."""
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def check_record_secrets(record: dict) -> None:
    """Refuse to save known provider keys or explicit credential fields."""
    encoded = json.dumps(record, ensure_ascii=False)
    for name in ('DEEPSEEK_API_KEY', 'NVIDIA_NIM_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        secret = os.environ.get(name, '')
        if secret and secret in encoded:
            raise ValueError('A provider key appeared in the record; it was not saved.')
    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {'api_key', 'authorization', 'access_token', 'extra_headers'} and child:
                    raise ValueError('A credential field appeared in a candidate configuration.')
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(record)


def rebind_single_input(config: dict, data_path: Path, output_path: Path) -> dict:
    """Change only file locations and disable old checkpoints for held-out execution."""
    rebound = copy.deepcopy(config)
    datasets = rebound.get('datasets', {})
    if len(datasets) != 1:
        raise ValueError('This helper requires one file dataset. Inspect this rewrite before replaying it.')
    dataset = next(iter(datasets.values()))
    if dataset.get('type') != 'file' or not isinstance(dataset.get('path'), str):
        raise ValueError('Expected one file-backed dataset, not embedded or multiple inputs.')
    old_path = dataset['path']
    dataset['path'] = str(data_path)
    rebound['pipeline']['output'] = {'type': 'file', 'path': str(output_path)}
    rebound['bypass_cache'] = True
    remainder = json.dumps(rebound, ensure_ascii=False)
    if old_path in remainder or Path(old_path).name in remainder:
        raise ValueError('The plan still refers to its old input elsewhere. Inspect before replaying.')
    return rebound


def load_course_data(directory: Path, bundle: Path | bytes) -> dict:
    """Load the supplied data, refusing changed or mismatched files."""
    if not (directory / 'manifest.json').exists():
        if isinstance(bundle, Path):
            if not bundle.is_file():
                raise FileNotFoundError('Place movie_lab_data.zip beside lab_support.py.')
            bundle = bundle.read_bytes()
        unpack_course_data(bundle, directory)
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError('Unexpected data manifest.')
    manifest = json.loads(manifest_bytes)
    for filename in DATA_FILES:
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != manifest['sha256'][filename]:
            raise ValueError(f'Data checksum mismatch: {filename}')
    return manifest


def prepare_course(directory: Path, course_bundle: Path) -> dict:
    """Read the data from the already verified course archive."""
    with zipfile.ZipFile(course_bundle) as archive:
        if sorted(archive.namelist()) != ['lab_support.py', 'movie_lab_data.zip']:
            raise ValueError('Unexpected course bundle contents.')
        if sum(item.file_size for item in archive.infolist()) > 6_000_000:
            raise ValueError('Course bundle is unexpectedly large.')
        return load_course_data(directory, archive.read('movie_lab_data.zip'))


def build_course_bundle(destination: Path, data_bundle: Path) -> str:
    """Package matching support code and data; return the notebook's SHA-256 pin.

    Maintainer use only. Update COURSE_BUNDLE_SHA256 in the notebook after rebuilding.
    No model calls are made and the input data archive is not modified.
    """
    source = Path(__file__)
    if not source.is_file():
        raise ValueError('Build from the source file, not from inside the course ZIP.')
    if destination.resolve() in {source.resolve(), data_bundle.resolve()}:
        raise ValueError('The output must not overwrite a source file.')
    if destination.exists():
        raise FileExistsError('Choose a new destination; existing bundles are not overwritten.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in [('lab_support.py', source), ('movie_lab_data.zip', data_bundle)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(destination.read_bytes()).hexdigest()



def show_reviews(rows: list[dict], titles: dict) -> None:
    """Display complete review text with escaped HTML and readable IDs."""
    from IPython.display import HTML, display
    for row in rows:
        title = titles[row['id']]
        display(HTML(f"<p><b>{html.escape(title)}</b> | reviewId: {html.escape(row['reviewId'])}</p>"
                     f"<blockquote>{html.escape(row['reviewText'])}</blockquote>"))


def show_pairs(pairs: set, index: dict, titles: dict, limit: int = 3) -> None:
    """Show complete review text for a few already-computed pairs."""
    print(f'{len(pairs)} ordered pairs; displaying up to {limit}.')
    for left, right in sorted(pairs)[:limit]:
        print(f'Left {left[1]} / Right {right[1]}')
        show_reviews([index[left], index[right]], titles)


def choose_example_pair(pairs_a: set, pairs_b: set) -> tuple | None:
    """Prefer a disagreement, then a shared pair; never manufacture an example."""
    choices = sorted(pairs_a ^ pairs_b) or sorted(pairs_a & pairs_b)
    return choices[0] if choices else None


def explain_pair(inputs: list[dict], labels: list[dict], pairs_a: set,
                 pairs_b: set, reference: dict, titles: dict) -> None:
    """Read one pair before introducing aggregate quality scores."""
    pair = choose_example_pair(pairs_a, pairs_b)
    if pair is None:
        print('Both methods returned no pairs. Next, check whether the reference contains any.')
        return
    indexed, validation = index_inputs(inputs), validate_labels(inputs, labels)
    if not validation['complete']:
        raise ValueError('Complete valid Method B labels are required to explain a pair.')
    predicted = validation['returned']
    print('A pair on which the methods disagree:' if pairs_a != pairs_b
          else 'The methods agree. Read one shared pair:')
    show_reviews([indexed[key] for key in pair], titles)
    for side, key in zip(('Left', 'Right'), pair):
        print(f'{side}: Method B says {predicted[key]["sentiment"]}; reference says {reference[key]}.')
    different = predicted[pair[0]]['sentiment'] != predicted[pair[1]]['sentiment']
    print(f'Method A {"kept" if pair in pairs_a else "left out"} this pair.')
    print(f'Method B {"kept" if pair in pairs_b else "left out"} it because its labels are '
          f'{"different" if different else "the same"}.')
    print('The reference labels describe opposite opinions.' if reference[pair[0]] != reference[pair[1]]
          else 'The reference labels describe the same overall attitude.')


def plot_execution_plans(review_count: int) -> Any:
    """Draw the two teaching plans, separating model work from ordinary code."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(11, 3.7), layout='constrained')
    ax.set(xlim=(-0.6, 3.6), ylim=(-0.65, 1.55))
    ax.axis('off')
    rows = [
        ('A', [f'{review_count} reviews', 'Build eligible\nordered pairs',
               'Model judges\neach pair', 'Keep matching\npairs']),
        ('B', [f'{review_count} reviews', 'Model labels\neach review',
               'Python pairs\nopposite labels', 'Return matching\npairs']),
    ]
    for y, (name, steps) in zip((1, 0), rows):
        ax.text(-0.48, y, f'Method {name}', ha='center', va='center', fontsize=10)
        for x, label in enumerate(steps):
            model_step = (name == 'A' and x == 2) or (name == 'B' and x == 1)
            ax.text(x, y, label, ha='center', va='center', fontsize=10,
                    bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '#d8eee9' if model_step else '#eeeeee',
                          'edgecolor': '#46796d' if model_step else '#777777'})
            if x < 3:
                ax.annotate('', xy=(x + 0.66, y), xytext=(x + 0.34, y),
                            arrowprops={'arrowstyle': '->', 'color': '#444444'})
    ax.legend(handles=[Patch(facecolor='#d8eee9', label='Model judgment'),
                       Patch(facecolor='#eeeeee', label='Data / ordinary code')],
              loc='upper center', ncol=2, frameon=False)
    ax.set_title('One question, two ways to process the reviews', pad=12)
    return fig


def pair_matrix(inputs: list[dict], pairs: set) -> list[list[int]]:
    """Encode ordered pairs as 1, absent pairs as 0, and self-pairs as -1."""
    keys = list(index_inputs(inputs))
    eligible = {(a, b) for a in keys for b in keys if a != b and a[0] == b[0]}
    if not pairs <= eligible:
        raise ValueError('Matrix contains an unknown, cross-movie, or self pair.')
    return [[-1 if a == b else int((a, b) in pairs) for b in keys] for a in keys]


def plot_pair_matrices(inputs: list[dict], expected: set, pairs_a: set, pairs_b: set) -> Any:
    """Plot reference and measured pair sets on the same axes and color scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.3), layout='constrained')
    labels = [f'R{i + 1}' for i in range(len(inputs))]
    colors = ['#c9c9c9', '#ffffff', '#397b70']
    for ax, name, pairs in zip(axes, ('Reference', 'Method A', 'Method B'), (expected, pairs_a, pairs_b)):
        ax.imshow(pair_matrix(inputs, pairs), cmap=ListedColormap(colors),
                  norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3))
        ax.set(xticks=range(len(inputs)), yticks=range(len(inputs)), xticklabels=labels,
               yticklabels=labels, xlabel='Right review', ylabel='Left review', title=name)
        ax.tick_params(labelsize=8)
    fig.suptitle('Each colored square is one returned ordered pair')
    axes[1].legend(handles=[Patch(facecolor=c, edgecolor='#777777', label=t)
                           for c, t in zip(colors, ('Self-pair excluded', 'Not returned', 'Returned'))],
                   loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    return fig


def measured_tokens(usage: object, field: str) -> float | None:
    """Sum one DocETL per-model token field without adding totals or cached subsets."""
    import math
    if not isinstance(usage, dict) or not usage:
        return None
    values = [item.get(field) if isinstance(item, dict) else None for item in usage.values()]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in values):
        return None
    return sum(values)


def plot_work_comparison(method_a: dict, method_b: dict) -> Any:
    """Display measured query time and input/output tokens; leave unknowns unplotted."""
    import math
    import matplotlib.pyplot as plt
    records = (method_a, method_b)
    times = [r.get('elapsed_seconds') for r in records]
    times = [t + r.get('python_pairing_seconds', 0.0) if isinstance(t, (int, float)) else None
             for t, r in zip(times, records)]
    columns = [('Query + pairing time', 'Seconds', times),
               ('Input tokens', 'Tokens', [measured_tokens(r.get('token_usage'), 'prompt_tokens') for r in records]),
               ('Output tokens', 'Tokens', [measured_tokens(r.get('token_usage'), 'completion_tokens') for r in records])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained')
    for ax, (title, unit, values) in zip(axes, columns):
        valid = [(i, value) for i, value in enumerate(values)
                 if not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0]
        for i, value in valid:
            ax.bar(i, value, width=0.55, color=('#397b70', '#be7242')[i])
            ax.annotate(f'{value:,.2f}' if unit == 'Seconds' else f'{value:,.0f}',
                        (i, value), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        for i in set(range(2)) - {i for i, _ in valid}:
            ax.text(i, 0.08, 'Unknown', transform=ax.get_xaxis_transform(), ha='center')
        high = max((value for _, value in valid), default=0)
        ax.set(xticks=[0, 1], xticklabels=['A', 'B'], xlim=(-0.6, 1.6),
               ylim=(0, high * 1.25 if high else 1), title=title, ylabel=unit)
    return fig


def first_comparable_candidate(record: dict | None, inputs: list[dict], expected: dict) -> str | None:
    """Select the first comparable non-baseline in record order, never by score."""
    if record is None:
        return None
    baseline = find_baseline(record)
    project_candidate_labels(inputs, baseline['outputs'])
    for plan in record['plans']:
        if plan['id'] == baseline['id']:
            continue
        try:
            prediction_changes(inputs, baseline['outputs'], plan['outputs'], expected)
        except ValueError:
            continue
        return plan['id']
    return None


def show_candidate_change(record: dict, plan_id: str, inputs: list[dict], expected: dict,
                          titles: dict) -> None:
    """Show an actual rewrite and one changed review before aggregate scores."""
    import difflib
    import yaml
    from IPython.display import HTML, display
    lookup = {plan['id']: plan for plan in record['plans']}
    if plan_id not in lookup:
        raise ValueError('Choose a candidate ID from this run.')
    plan = lookup[plan_id]
    summary = operation_changes(record['initial_config'], plan['config'])
    print(f'Inspecting {plan_id}. This display is not a recommendation to adopt it.')
    print(summary if summary is not None else 'The operation structure changed; read the full configuration difference.')
    before = yaml.safe_dump(record['initial_config'], sort_keys=False).splitlines()
    after = yaml.safe_dump(plan['config'], sort_keys=False).splitlines()
    diff = '\n'.join(difflib.unified_diff(before, after, fromfile='Initial query', tofile=plan_id, lineterm=''))
    display(HTML(text_details('Full query configuration difference', diff or 'No differences.', expanded=summary is None)))
    try:
        baseline = find_baseline(record)
        changes = prediction_changes(inputs, baseline['outputs'], plan['outputs'], expected)
    except ValueError as error:
        print(f'Cannot compare complete predictions: {error}')
        return
    print(f'{len(changes)} labels changed from the recorded baseline.')
    if changes:
        change = changes[0]
        row = index_inputs(inputs)[(change['Movie ID'], change['reviewId'])]
        show_reviews([row], titles)
        print(f'Before: {change["Before"]}; after: {change["After"]}; reference: {change["Reference"]}.')
        display(HTML(text_details('All changed predictions', json.dumps(changes, ensure_ascii=False, indent=2))))
    else:
        print('The query changed, but none of these sentiment labels changed.')


def run_logged(command: list[str], log_path: Path, *, timeout: int,
               cwd: Path | None = None) -> None:
    """Keep routine output in a log; propagate failures with its location."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as log:
        try:
            subprocess.run(command, check=True, timeout=timeout, cwd=cwd,
                           stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'Command failed (exit {error.returncode}). Read {log_path}') from error
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f'Command exceeded {timeout} seconds. Read {log_path}') from error



def install_docetl(runs_dir: Path, version: str = '0.3.0') -> dict:
    """Install the teaching version and record dependency versions, without a query."""
    if version != '0.3.0':
        raise ValueError('This notebook and recorder require DocETL 0.3.0.')
    if sys.version_info < (3, 10):
        raise RuntimeError('DocETL 0.3.0 requires Python 3.10 or newer.')
    print(f'Installing DocETL {version}. Detailed output: {runs_dir / "install.log"}')
    run_logged([sys.executable, '-m', 'pip', 'install', f'docetl=={version}'],
               runs_dir / 'install.log', timeout=600)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name)
                for name in ('docetl', 'litellm', 'pandas', 'pydantic')}
    if versions['docetl'] != version:
        raise RuntimeError('Restart the kernel after installing DocETL 0.3.0.')
    (runs_dir / 'dependency_versions.json').write_text(json.dumps(versions, indent=2))
    return versions



def load_key(name: str, in_colab: bool) -> None:
    """Load a key without displaying it or saving it in this notebook."""
    value = os.environ.get(name, '').strip()
    if not value and in_colab:
        from google.colab import userdata
        try:
            value = userdata.get(name).strip()
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError, TimeoutError):
            value = ''
    if not value:
        value = getpass.getpass(f'{name}: ').strip()
    if not value:
        raise ValueError(f'No key provided for {name}.')
    os.environ[name] = value


def configure_provider(provider: str, nvidia_model: str, options: dict,
                       in_colab: bool) -> dict:
    """Configure only the selected provider; never print or persist its key."""
    if provider not in {'deepseek', 'nvidia', 'ollama'}:
        raise ValueError('Unknown provider.')
    options = dict(options)
    if provider == 'deepseek':
        load_key('DEEPSEEK_API_KEY', in_colab)
        options['api_base'] = 'https://api.deepseek.com'
        options['extra_body'] = {'thinking': {'type': 'disabled'}}
    elif provider == 'nvidia':
        load_key('NVIDIA_NIM_API_KEY', in_colab)
        options['api_base'] = 'https://integrate.api.nvidia.com/v1'
        if nvidia_model == 'nvidia/nemotron-3-super-120b-a12b':
            # This query only needs a short judgment, so disable extended thinking.
            options['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    else:
        options['api_base'] = 'http://127.0.0.1:11434'
        options['num_ctx'] = 8192
        options['think'] = False

    print(f'Configuration prepared for {provider}. No review query has run yet.')
    return options



def start_ollama(provider: str, model_ref: str, in_colab: bool, runs_dir: Path) -> None:
    """Install/start Ollama only when explicitly selected; save setup output."""
    setup_log = runs_dir / 'ollama-setup.log'
    if provider == 'ollama':
        print(f'Preparing Ollama and downloading the selected model. Log: {setup_log}')
        if shutil.which('ollama') is None:
            if not in_colab:
                raise RuntimeError('Install Ollama locally, then rerun this cell.')
            run_logged(['apt-get', 'update'], setup_log, timeout=180)
            run_logged(['apt-get', 'install', '-y', 'zstd', 'curl', 'ca-certificates'],
                           log_path=setup_log, timeout=180)
            installer = runs_dir / 'ollama-install.sh'
            run_logged(['curl', '-fsSL', 'https://ollama.com/install.sh', '-o', str(installer)],
                           log_path=setup_log, timeout=60)
            run_logged(['sh', str(installer)], log_path=setup_log, timeout=600)

        def ollama_ready() -> bool:
            """Check whether the local Ollama server responds, without generating text."""
            try:
                with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2) as response:
                    return response.status == 200
            except (urllib.error.URLError, TimeoutError):
                return False

        if not ollama_ready():
            with (runs_dir / 'ollama-server.log').open('ab') as log:
                ollama_process = subprocess.Popen(['ollama', 'serve'], stdout=log, stderr=log,
                                                  start_new_session=True)
            for _ in range(30):
                if ollama_ready():
                    break
                if ollama_process.poll() is not None:
                    raise RuntimeError('Ollama stopped. Read runs/ollama-server.log.')
                time.sleep(1)
            if not ollama_ready():
                raise TimeoutError('Ollama did not become ready. Read runs/ollama-server.log.')
        run_logged(['ollama', 'pull', model_ref.split('/', 1)[1]], log_path=setup_log, timeout=1200)
        print('Ollama is ready for the review-pair query.')
    else:
        print('Skipped: the selected provider does not use a local Ollama server.')


WORKER_CODE = r'''import json
from pathlib import Path
import sys
import time
from typing import Any
import docetl
from docetl.operations.utils import api as model_api

original_completion = model_api.completion
original_parse = model_api.APIWrapper.parse_llm_response
example_call = None
observed_response = None


def recorded_completion(*args: Any, **kwargs: Any) -> Any:
    """Record the first call's teaching fields and return the unchanged response."""
    global example_call, observed_response
    capture = example_call is None
    if capture:
        # Allowlist model inputs. Never copy credentials or request headers.
        fields = ('model', 'messages', 'tools', 'tool_choice', 'response_format')
        example_call = {
            'request': json.loads(json.dumps({k: kwargs[k] for k in fields if k in kwargs})),
            'response': None,
            'parsed_output': None,
        }
    response = original_completion(*args, **kwargs)
    if capture:
        observed_response = response
        example_call['response'] = [
            {'message': choice.message.model_dump(mode='json'),
             'finish_reason': choice.finish_reason}
            for choice in response.choices
        ]
    return response


def recorded_parse(self: Any, response: Any, *args: Any, **kwargs: Any) -> Any:
    """Save DocETL's parsed fields for that same response without changing them."""
    parsed = original_parse(self, response, *args, **kwargs)
    if (example_call is not None and response is observed_response
            and example_call['parsed_output'] is None):
        example_call['parsed_output'] = json.loads(json.dumps(parsed))
    return parsed


frame = docetl.Frame.from_yaml(sys.argv[1])
started = time.perf_counter()
model_api.completion = recorded_completion
model_api.APIWrapper.parse_llm_response = recorded_parse
try:
    rows = frame.collect(max_threads=1)
finally:
    model_api.completion = original_completion
    model_api.APIWrapper.parse_llm_response = original_parse
    if example_call is not None:
        Path(sys.argv[2]).with_name('example_call.json').write_text(
            json.dumps(example_call, ensure_ascii=False, indent=2))
record = {
    'rows': rows,
    'elapsed_seconds': time.perf_counter() - started,
    'token_usage': frame.token_usage,
    'reported_cost_usd': frame.total_cost,
    'example_call': example_call,
}
Path(sys.argv[2]).write_text(json.dumps(record, ensure_ascii=False, indent=2))
'''


PAIR_AUDIT_CODE = r"""
from docetl.operations.equijoin import EquijoinOperation
pair_judgments = []
parsed_pair_answers = []
base_recorded_parse = recorded_parse
base_compare = EquijoinOperation.compare_pair

def audited_parse(self: Any, response: Any, *args: Any, **kwargs: Any) -> Any:
    parsed = base_recorded_parse(self, response, *args, **kwargs)
    if len(parsed) == 1 and type(parsed[0].get('is_match')) is bool:
        parsed_pair_answers.append(parsed[0]['is_match'])
    return parsed

def audited_compare(self: Any, prompt: str, model: str, left: dict, right: dict,
                    *args: Any, **kwargs: Any) -> tuple[bool, float]:
    before = len(parsed_pair_answers)
    first_call = example_call is None
    match, cost = base_compare(self, prompt, model, left, right, *args, **kwargs)
    if first_call and example_call is not None:
        example_call['pair_keys'] = [[left['id'], left['reviewId']], [right['id'], right['reviewId']]]
    if len(parsed_pair_answers) != before + 1 or parsed_pair_answers[-1] != match:
        raise ValueError('A pair did not receive a valid Boolean answer; stop rather than score it as false.')
    pair_judgments.append({'left': [left['id'], left['reviewId']],
                           'right': [right['id'], right['reviewId']], 'is_match': match})
    return match, cost

recorded_parse = audited_parse
EquijoinOperation.compare_pair = audited_compare
"""


def run_query(frame: Any, name: str, deadline_seconds: int = 300, *,
              run_dir: Path, model_ref: str, enabled: bool,
              worker_code: str = WORKER_CODE) -> dict:
    """Run a real DocETL query with an overall worker time limit."""
    if not enabled:
        raise RuntimeError('Enable model calls in the provider-choice cell before running a query.')
    folder = run_dir / f'{name}-{uuid.uuid4().hex[:8]}'
    folder.mkdir()
    config_path, worker_path, result_path = (folder / filename for filename in
                                             ('query.yaml', 'worker.py', 'result.json'))
    config_path.write_text(frame.to_yaml())
    worker_path.write_text(worker_code)
    print(f'Running {name}. Detailed output: {folder / "execution.log"}')
    started = time.perf_counter()
    run_logged([sys.executable, str(worker_path), str(config_path), str(result_path)],
               folder / 'execution.log', timeout=deadline_seconds, cwd=folder)
    record = json.loads(result_path.read_text())
    record.update(model=model_ref, cache='bypassed', wall_seconds=time.perf_counter() - started)
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Returned {len(record['rows'])} rows. Record saved in {folder.name}/result.json")
    return record


def run_pair_query(frame: Any, *, run_dir: Path, model_ref: str, enabled: bool) -> dict:
    """Record every pair judgment without changing global worker state."""
    marker = 'frame = docetl.Frame.from_yaml(sys.argv[1])'
    if WORKER_CODE.count(marker) != 1 or WORKER_CODE.count("'rows': rows,") != 1:
        raise ValueError('Query worker changed; check the pair recorder before running.')
    worker = WORKER_CODE.replace(marker, PAIR_AUDIT_CODE + '\n' + marker)
    worker = worker.replace("'rows': rows,", "'rows': rows, 'pair_judgments': pair_judgments,")
    return run_query(frame, 'method-a-pairs', deadline_seconds=1200, run_dir=run_dir,
                     model_ref=model_ref, enabled=enabled, worker_code=worker)



def write_inputs(name: str, rows: list[dict], run_dir: Path) -> Path:
    """Write only validated model inputs for this run."""
    index_inputs(rows)
    path = run_dir / f'{name}.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return path


def make_json_adapter(original: Callable, base_model: type, stats: dict,
                      limit: int = 32) -> Callable:
    """Adapt DeepSeek schema requests to JSON mode and validate every response."""
    import inspect
    import threading
    from pydantic import ValidationError
    lock = threading.Lock()

    def adapted(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get('model', args[0] if args else '')
        schema = kwargs.get('response_format')
        if not (str(model).startswith('deepseek/') and inspect.isclass(schema)
                and issubclass(schema, base_model)):
            return original(*args, **kwargs)
        options = dict(kwargs)
        options['messages'] = [dict(message) for message in options['messages']] + [{
            'role': 'user', 'content': 'Return only a JSON object conforming to this JSON Schema. '
            'Include every required field and use the permitted values. Schema: '
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)}]
        options['response_format'] = {'type': 'json_object'}
        options.setdefault('num_retries', 0)
        options.setdefault('timeout', 90)
        options.setdefault('max_tokens', 8192)
        options.setdefault('extra_body', {'thinking': {'type': 'disabled'}})
        for attempt in range(2):
            with lock:
                if stats['requests'] >= limit:
                    raise RuntimeError('MOAR rewrite-request limit reached; inspect the saved log.')
                stats['requests'] += 1
            response = original(*args, **options)
            try:
                schema.model_validate_json(response.choices[0].message.content, strict=True)
            except ValidationError as error:
                errors = [{'field': list(e['loc']), 'type': e['type']} for e in error.errors()]
                with lock:
                    stats['validation_failures'].append({'schema': schema.__name__,
                                                        'attempt': attempt + 1, 'errors': errors})
                if attempt == 1:
                    raise
                options['messages'] += [
                    {'role': 'assistant', 'content': response.choices[0].message.content},
                    {'role': 'user', 'content': 'The JSON failed validation: ' + json.dumps(errors)
                     + '. Return one complete JSON object matching the schema above.'}]
                continue
            with lock:
                stats['validated'] += 1
            return response
        raise AssertionError('Unreachable validation state.')

    return adapted


@contextmanager
def moar_response_format(model_ref: str) -> Iterator[dict]:
    """Scope the provider adapter to this search; never probe or switch providers."""
    stats = {'requests': 0, 'validated': 0, 'validation_failures': []}
    if not model_ref.startswith('deepseek/'):
        yield stats
        return
    import litellm
    from pydantic import BaseModel
    original = litellm.completion
    adapter = make_json_adapter(original, BaseModel, stats)
    litellm.completion = adapter
    for name, module in tuple(sys.modules.items()):
        if name.startswith('docetl.') and getattr(module, 'completion', None) is original:
            module.completion = adapter
    try:
        yield stats
    finally:
        if litellm.completion is adapter:
            litellm.completion = original
        # Include aliases imported while the search was running.
        for name, module in tuple(sys.modules.items()):
            if name.startswith('docetl.') and getattr(module, 'completion', None) is adapter:
                module.completion = original


def snapshot_search(result: object, original_config: dict, *, score_fn: Callable,
                    references: dict, versions: dict, model_ref: str, rewrite_model: str,
                    manifest: dict, evaluation_failures: list) -> dict:
    """Save returned evaluated plans, not invented candidates or inferred search steps."""
    import math
    import yaml
    plans = []
    for number, point in enumerate(result.all_plans):
        config = yaml.safe_load(Path(point.yaml_path).read_text())
        output_path = Path(config['pipeline']['output']['path'])
        rows = json.loads(output_path.read_text())
        try:
            score = score_fn(rows, references)
            error = None
        except ValueError as exc:
            score, error = None, str(exc)
        plans.append({'id': f'plan-{number}', 'config': config, 'outputs': rows,
                      'score': score, 'score_error': error,
                      'reported_score': point.accuracy if math.isfinite(point.accuracy) else None,
                      'reported_cost_usd': point.cost, 'on_frontier': point.on_frontier})
    record = {'format': MOAR_RECORD_FORMAT, 'docetl_version': versions['docetl'],
              'versions': versions, 'model': model_ref, 'rewrite_model': rewrite_model,
              'optimization_sha256': manifest['sha256']['optimization.json'],
              'reference_sha256': manifest['sha256']['optimization_labels.json'],
              'initial_config': original_config, 'plans': plans,
              'evaluation_failures': list(evaluation_failures),
              'reported_search_cost_usd': result.total_search_cost,
              'search_seconds': result.duration_seconds, 'iterations': result.iterations,
              'pricing_verified': False,
              'plan_list_scope': 'MOARResult.all_plans; not a complete list of failed attempts.'}
    check_record_secrets(record)
    return record


def validate_search_record(record: dict, expected: dict, data_hash: str, label_hash: str, *, score_fn: Callable) -> dict:
    """Verify provenance and recompute scores; never execute stored configurations."""
    import math
    if record.get('format') != MOAR_RECORD_FORMAT or record.get('docetl_version') != '0.3.0':
        raise ValueError('Expected a DocETL 0.3.0 course run record.')
    if record.get('optimization_sha256') != data_hash or record.get('reference_sha256') != label_hash:
        raise ValueError('Saved run uses different optimization data or reference labels.')
    if not isinstance(record.get('initial_config'), dict) or not isinstance(record.get('plans'), list):
        raise ValueError('Saved record is missing configurations or candidates.')
    seen = set()
    for plan in record['plans']:
        if not isinstance(plan.get('id'), str) or plan['id'] in seen or not isinstance(plan.get('config'), dict):
            raise ValueError('Invalid or duplicate candidate identifier/configuration.')
        seen.add(plan['id'])
        try:
            score = score_fn(plan['outputs'], expected)
        except ValueError:
            if plan.get('score') is not None or not plan.get('score_error'):
                raise ValueError('Invalid candidate was saved without an error marker.')
            continue
        if plan.get('score') != score or plan.get('score_error'):
            raise ValueError('Saved score does not match the actual saved output.')
        reported = plan.get('reported_score')
        if (not isinstance(reported, (int, float)) or isinstance(reported, bool)
                or not math.isfinite(reported) or not math.isclose(reported, score['macro_f1'], abs_tol=1e-9)):
            raise ValueError('MOAR score and the course scoring function disagree.')
    check_record_secrets(record)
    return record


def prediction_changes(inputs: list[dict], before: list[dict], after: list[dict],
                       expected: dict) -> list[dict]:
    """Compare labels by ID and display original text, never rewritten candidate text."""
    before_index = {review_key(row): row for row in project_candidate_labels(inputs, before)}
    after_index = {review_key(row): row for row in project_candidate_labels(inputs, after)}
    return [{'reviewId': key[1], 'Movie ID': key[0], 'Original review': row['reviewText'],
             'Before': row['sentiment'], 'After': after_index[key]['sentiment'],
             'Reference': expected[key]}
            for key, row in before_index.items()
            if row['sentiment'] != after_index[key]['sentiment']]


def find_baseline(record: dict) -> dict:
    """Find a saved evaluation of the original operations, not an assumed plan number."""
    if not isinstance(record.get('plans'), list) or not isinstance(record.get('initial_config'), dict):
        raise ValueError('A complete search record is required to identify its baseline.')
    matches = [plan for plan in record['plans']
               if plan['config'].get('operations') == record['initial_config'].get('operations')]
    if len(matches) != 1:
        raise ValueError('Cannot identify one baseline from the saved operation definitions.')
    return matches[0]


def load_prompt_case(comparison_path: Path, output_path: Path, baseline_path: Path,
                     inputs: list[dict], expected: dict, manifest: dict, *, score_fn: Callable) -> dict:
    """Verify the separately executed reference against its raw output and baseline."""
    comparison = json.loads(Path(comparison_path).read_text())
    output_bytes = Path(output_path).read_bytes()
    if hashlib.sha256(output_bytes).hexdigest() != comparison.get('query_result_sha256'):
        raise ValueError('Prompt-case result does not match its recorded hash.')
    record = validate_search_record(json.loads(Path(baseline_path).read_text()), expected,
        manifest['sha256']['optimization.json'], manifest['sha256']['optimization_labels.json'],
        score_fn=score_fn)
    baseline = find_baseline(record)
    after = json.loads(output_bytes)['rows']
    before = baseline['outputs']
    if (score_fn(before, expected) != comparison.get('baseline_score')
            or score_fn(after, expected) != comparison.get('candidate_score')):
        raise ValueError('Prompt-case scores disagree with the saved predictions.')
    prompts = [op.get('prompt') for op in baseline['config']['operations']]
    if comparison.get('before_prompt') not in prompts or not isinstance(comparison.get('after_prompt'), str):
        raise ValueError('Prompt-case wording does not identify the baseline prompt.')
    return {'before_prompt': comparison['before_prompt'], 'after_prompt': comparison['after_prompt'],
            'before_rows': before, 'after_rows': after,
            'changes': prediction_changes(inputs, before, after, expected)}

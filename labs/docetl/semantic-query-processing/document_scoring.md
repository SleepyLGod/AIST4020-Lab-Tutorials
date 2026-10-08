# Checking the sector query

The optimization example asks which documents are sustainability reports and
which sector each retained company belongs to. Python scores the returned rows;
no LLM judge or judge calibration is used.

## Reference decisions

| Document | Expected decision |
| --- | --- |
| R01 Carrier | Manufacturing |
| R02 Boeing | Manufacturing |
| R05 XPO | Transportation and logistics |
| R06 Tallink | Transportation and logistics |
| R08 Lenovo | Excluded |
| R10 Sustainserv | Excluded |

These decisions use the report-selection prompt and the two supplied sector
definitions. They were checked against all six excerpts, including an independent
AI review before consulting the existing reference. They are teaching references,
not independently human-annotated benchmark gold.

[golden_answers.json](document_data/golden_answers.json) supplies the existing
filter IDs, sector pairs, source quotes and input hashes. Its older proposed
map/reduce judging protocol is not used by this experiment. The file is kept
unchanged to preserve the earlier reference work.

Tallink's report qualifies even though its excerpt establishes shore-power
capability, not actual use or who financed the installations. Lenovo's financial
announcement and Sustainserv's job advertisement are excluded by the query.

## One score per document

Each of the six documents earns one point when its final decision is correct.
Accuracy is the number of correct decisions divided by six. A retained document
must have its expected sector; an excluded document must be absent.

Missing a report, retaining an excluded document, or assigning the wrong valid
sector loses that document's point. For example, missing only R06 gives 5/6.
A successfully executed empty result gives 2/6, because only the two exclusions
are correct. Execution failure is not an empty result and receives no score.

Duplicate or unknown IDs, missing fields, and invalid sector names are invalid
outputs. They are not repaired, deduplicated or assigned a guessed score.
Other fields may remain in the output but do not affect this metric.

Counts are computed from validated rows after scoring. They are not the score:
swapping two companies between sectors could leave both counts at two while
making two document decisions wrong. Each retained document represents one
company in this teaching subset.

## Optimization and records

The baseline and every candidate use all six original document records and the
same Python scorer. Only source text and IDs are written to the optimizer's input
file. Sector definitions appear in the query; reference labels stay in the scorer
and are not passed to the model or rewrite agent.

MOAR receives accuracy feedback. Generated plans still need inspection for whether
they answer the same question; matching six visible answers cannot prove semantic
equivalence or generalization. Existing path, provider and model checks are kept,
but they are not an OS sandbox. Use an isolated Colab runtime.

Only a strictly higher valid accuracy replaces the original. Ties keep the
original. Invalid trials remain visible with no score. A run with no valid trial
is reported as failed, not as successful optimization.

Records use `docetl-sector-search-v2` and keep configurations, outputs, scores,
selection and failure status. Old extraction scores cannot be loaded as sector
query results. Automatic search and the optional separate Prompt Rewriting run
have different labels. The selected rows are counted without another model call.

Per-query measurements and total MOAR search cost are reported separately.
Unknown prices remain unknown, including zero SDK estimates that have not been
verified. Do not add overlapping costs. No judge cost is incurred by this scorer.

## Sources

- [DocETL 0.3.0 MOAR interface](https://github.com/ucbepic/docetl/blob/0.3.0/docetl/moar/optimizer.py)
- [MOAR evaluation handling](https://github.com/ucbepic/docetl/blob/0.3.0/docetl/moar/MOARSearch.py)

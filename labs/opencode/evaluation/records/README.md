# Historical OpenCode deliveries

These are excerpts from real course pilots using OpenCode `1.18.34`, not
simulated outputs and not results from the student's current runtime.

| Record | What it contains | Model |
| --- | --- | --- |
| `incomplete.json` | Program repaired, report left as its original title | Ollama `qwen3:8b` |
| `complete.json` | A separate delivery of the original repair task with a completed report | DeepSeek `deepseek-flash` |
| `remediated.json` | A new session completing the incomplete record's report, with other files unchanged | DeepSeek `deepseek-flash` |

The complete delivery is used to check whether the acceptance checks reject
good work. It is not a second attempt at the incomplete run. The remediation
restored the incomplete run's after-files; it did not resume its session.
These records do not measure a change in model success rate.

## What was retained

- Original task and tool permissions.
- Exact before/after project file text and SHA-256 hashes.
- Ordered assistant text and tool requests/results, including errors.
- Recorded model usage and task time.

Conversation excerpts omit reasoning, account metadata, and other unused
fields. Machine-specific project paths become `<project>`; run-specific temporary
paths become `<runtime-path>`. Project file contents are not rewritten.
The `provenance` object records hashes of the original archive and the source
JSON entries used to produce each excerpt. The private archives are not needed
to run this lab.

## How to read the numbers

Tokens are summed from unique assistant-message usage records. Cached input
is included and shown separately. Response counts do not count hidden HTTP
retries. The OpenCode `cost` field is not a billing receipt; the actual charge
is unknown. Task time excludes runtime allocation and installation.

The report-section check only finds missing or empty sections. Read the report
against the recorded test outputs and file differences to assess its claims.

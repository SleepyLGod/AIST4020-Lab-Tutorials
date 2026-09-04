# Optional: Run Colab From Your Local Terminal

This page is optional. The recommended student path is still to open the lab notebook in the browser and run it cell by cell.

Use Colab CLI when you want to control a Colab runtime from your local terminal. This is useful for TAs, instructors, and advanced students who want to:

* start a Colab CPU/GPU runtime without opening the notebook first;
* run a local Python script or notebook on that runtime;
* export a replayable execution log;
* stop the runtime when the run is finished.

Official references:

* [Google Developers Blog: Introducing the Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/)
* [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
* [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

## When to use this path

Use the browser notebook first if you are new to Colab or OpenClaw. It is easier to read the explanations, enter an API key safely, and inspect each output.

Use Colab CLI when you are comfortable with terminal workflows and want a remote Colab runtime behind local commands.

In this course, the two paths have different jobs:

* browser Colab is the normal student path: run cells one by one, read the explanations, use Colab Secrets, and choose the runtime manually;
* Colab CLI is an advanced TA/instructor path: rehearse the notebook, export execution logs, and control a remote runtime from the terminal.

Colab CLI currently supports Linux and macOS. The official README says Windows is not supported at this time. Windows users should prefer browser Colab, or use a Linux environment such as WSL only if they are already comfortable with it.

## Install locally

If you already use `uv`, install Colab CLI as a user-level command-line tool:

```bash
uv tool install google-colab-cli
colab version
```

This keeps the CLI separate from the course notebook environment.

If you do not use `uv`, the official README also supports `pip`:

```bash
pip install google-colab-cli
colab version
```

## First run

Create a named Colab runtime:

```bash
colab new -s openclaw-demo --gpu T4
```

Check the runtime:

```bash
colab status -s openclaw-demo
```

Run a local notebook on the Colab runtime:

```bash
colab exec -s openclaw-demo \
  -f labs/overall-harness/openclaw_overall_harness.ipynb \
  --timeout 1800
```

The default `colab exec` timeout is 30 seconds. That is too short for this OpenClaw notebook, especially when the Ollama path downloads or runs a local model.

Export a log:

```bash
colab log -s openclaw-demo -o openclaw-demo-log.ipynb
```

Stop the runtime:

```bash
colab stop -s openclaw-demo
```

Use `colab exec` for this course notebook. `colab run` is a different workflow: it starts a fresh Colab VM, runs a local Python script, and then releases the VM.

## Authentication

The first command may ask you to authenticate with Google. Follow the CLI prompt.

The current CLI default authentication mode is `oauth2`.

If the default authentication path does not work on your machine, try the OAuth flow:

```bash
colab --auth oauth2 new -s openclaw-demo --gpu T4
```

Colab CLI stores local session metadata and tokens under your user config directory, such as `~/.config/colab-cli/`. Do not use this workflow on a shared machine unless you understand where credentials are stored and how to remove them.

The CLI is still evolving. If an option behaves differently from this page, check your installed version with `colab --help` and `colab <command> --help`.

## API keys and secrets

Do not put API keys in this file, in shell history, or in committed notebooks.

For normal students, browser Colab is easier because the notebook can use Colab Secrets or hidden manual input.

For TA automation, prefer a separate private script or local environment variable flow. Keep that script out of the public teaching repo unless it contains no secrets.

## GPU availability

Colab CLI supports requesting `T4`, `L4`, `G4`, `H100`, and `A100` GPUs. This course uses `--gpu T4` in the example because T4 is a common and usually more practical classroom choice.

Requesting a GPU does not guarantee that the exact GPU will be available. Colab resources depend on account type, quota, subscription tier, current availability, and runtime limits.

If `--gpu T4` fails or is slow:

* retry later;
* run the light parts on CPU;
* use a hosted API provider instead of a local model;
* use browser Colab and choose a runtime manually;
* use local Ollama, vLLM, or SGLang if your machine has enough resources.

## Region note

This CLI is an access path to Colab runtimes. If your Google account can use Colab normally in your region, the CLI is expected to use the same Colab backend.

There is no separate course guarantee that every region, account, or GPU type will work. Test with your own Google account before using this path for a live demo.

## Recommended course policy

For this course, treat Colab CLI as an optional advanced path:

* students: use browser Colab unless you already like terminal workflows;
* TAs: use Colab CLI to rehearse demos, collect logs, or validate that a notebook still runs;
* instructors: use it for live demos only after testing with the same account and runtime type.

# Docker Backup for Overall Harness

This backup path is for students who cannot use Colab but do not want to install OpenClaw directly on their host environment.

The primary local backup is **Docker + API provider**. Docker can isolate the OpenClaw CLI and workspace, while the model is served by DeepSeek or another API provider.

## What Docker can and cannot standardize

Docker is useful for keeping the OpenClaw runtime environment similar across macOS, Windows, and Linux.

Docker does not make GPU-backed local model serving identical across operating systems.

* API providers are the most portable path.
* Host Ollama is usable as an optional path.
* Ollama with GPU inside Docker is hardware- and OS-dependent.
* macOS, Windows, and Linux do not expose GPUs to Docker in the same way.

## Recommended backup path: Docker + DeepSeek API

Use this when Colab is unavailable.

```bash
docker run --rm -it \
  -e DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  -v "$PWD/../workspace:/workspace:ro" \
  ubuntu:24.04 bash
```

Inside the container, install OpenClaw using the same installer flow from the notebook:

```bash
apt-get update
apt-get install -y curl ca-certificates git
curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard
export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:$PATH"
openclaw --version
```

Then copy the lab workspace into OpenClaw's workspace and run the same DeepSeek lane commands from the notebook.

## Optional: Docker + host Ollama

If Ollama is running on the host, a container usually cannot reach it through `127.0.0.1`. Use the host gateway address.

On Docker Desktop for macOS or Windows, this is usually:

```text
host.docker.internal:11434
```

On Linux, Docker Compose can map the host gateway explicitly. This varies by installation, so treat it as optional.

For teaching, prefer the Colab notebook first. Use Docker only as a backup lane.

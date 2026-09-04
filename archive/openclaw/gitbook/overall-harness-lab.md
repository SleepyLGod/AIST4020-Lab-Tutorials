---
layout:
  width: wide
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: true
  tags:
    visible: true
  actions:
    visible: true
---

# Overall Harness Lab

This lab turns the overview into observable runs.

Read [Overall Harness](README.md) first if these terms are new: raw LLM call, harnessed agent turn, model route, context surface, workspace surface, policy surface, and session surface.

The notebook compares three paths: a raw LLM call without workspace context, a raw LLM call with explicit `notes.txt` context, and a harnessed OpenClaw agent turn. The goal is to observe which part of the system constructs input and which part mediates workspace access.

## Observation Path

| Step | Command path | What is passed to the model | Target observation | Harness surface |
| --- | --- | --- | --- | --- |
| A. Raw infer, no workspace context | `openclaw infer model run` | Only the question about `notes.txt`; the file content is not included | The model route works, but the model should not know `HARNESS_CONTEXT_VISIBLE` unless it infers or invents the marker | Explicit input / negative control |
| B. Raw infer, explicit `notes.txt` context | `openclaw infer model run` | The question plus the full text of `notes.txt` pasted into the prompt | The raw model path can answer the marker because the content is now in the prompt | Explicit context construction / positive control |
| C. Harnessed agent turn | `openclaw agent --local` | The user request only; the notebook does not paste `notes.txt` into the prompt | The runtime-mediated agent path can obtain or reference workspace content, with exact wording varying by model and OpenClaw version | Workspace surface / action boundary |
| Policy observation | `openclaw agent --local` output | Runtime-selected tool surface | Output may include `[agents/tool-policy] tool policy removed...` | Policy surface / action boundary |
| Session observation | fresh `--session-key` | A selected transcript/history boundary | A fresh key helps isolate the turn from stale history | Session surface |

In this lab, `notes.txt` is a controlled workspace fixture with one marker:

```text
Workspace marker: HARNESS_CONTEXT_VISIBLE
```

This comparison is not a test of whether the model can magically inspect files. If you paste `notes.txt` into a raw prompt, the model can answer from it. If you do not pass the file content, the raw LLM call should not know it. In the harnessed turn, the runtime can mediate workspace access, assemble context, expose tools, apply policy, and persist session state.

## Open The Notebook

Open the runnable notebook:

[Open in Colab](https://colab.research.google.com/github/SleepyLGod/openclaw-teaching/blob/main/labs/overall-harness/openclaw_overall_harness.ipynb)

You can run either lane depending on your resources. DeepSeek requires your own API key and may cost money. Ollama depends on local or Colab compute for the chosen model. The notebook gives the operational details near each lane.

**Tip:** You can replace the model route. DeepSeek is only the default hosted API example. If you have another configured API provider, you can use that instead. The Ollama lane is also not tied to `llama3.2:3b`; choose a model your runtime can handle. A local or self-hosted backend such as vLLM or SGLang can also be used if it is exposed through an OpenAI-compatible or otherwise configured OpenClaw route.

A few runtime tips: if DeepSeek asks for a key, skip that lane unless you have one. If Ollama cannot connect, restart the Ollama setup cell. If you use a Colab T4, a quantized 7B or 8B Ollama model is a reasonable next experiment after `llama3.2:3b`; a 14B model may fit only with tighter memory and slower full-agent behavior. If Ollama passes the smoke probe but stalls or gives off-task output in the full agent turn, treat that as an observation about local-model capacity under a full agent workload.

## Observation -> Harness Surface

Use this table after running or reading the notebook output.

| If you observe... | First place to look |
| --- | --- |
| API auth or model listing fails | Model route |
| Raw infer A gives the marker even without file content | Prompt leakage / hallucination / stale context |
| Raw infer B cannot answer after file content is pasted into the prompt | Prompt construction / model instruction following |
| Full agent turn ignores `AGENTS.md` or `SOUL.md` | Context surface / workspace setup |
| Agent cannot use or see a tool | Tool surface / policy surface |
| Tool-policy diagnostic removes tools | Policy surface |
| Old behavior appears after you changed files | Session surface / stale transcript |
| Ollama smoke output is close but not exact | Model route works; instruction following is weak |
| Local model is slow, misses the marker, or produces tool-like output in the harnessed turn | Model route / context budget / runtime capacity |
| You cannot tell what happened | Observability surface |

This is the habit behind Harness Engineering: when behavior changes, identify which surface to inspect or improve. Do not jump straight to “the model is bad” or “try again.”

Mini worksheet: choose one changed or surprising behavior from your run. Write down which harness surface you would inspect first, the smallest change you would try, and what output would show whether the change helped.

## Colab Runtime Safety

Each user gets their own Colab runtime when opening the notebook link. Your API key, `/content`, Ollama process, and `~/.openclaw` config are not shared through GitBook or GitHub.

If you use the same Google account in multiple browsers, Colab may reconnect to your own active runtime. Before sharing screenshots or committing notebook changes, clear outputs and avoid printing secrets.

For a clean environment, use one of these Colab actions:

```text
Runtime -> Disconnect and delete runtime
Runtime -> Factory reset runtime
```

## Discussion Questions

1. Which command was only a model-route smoke probe?
2. In raw infer A, what exact input did the model receive?
3. In raw infer B, why can the same raw model path answer the marker?
4. In the harnessed agent turn, what did the notebook not paste manually?
5. What did the tool-policy line show about the action boundary?
6. Why can a model smoke probe pass while `agent --local` is slow or unstable?
7. Which observation maps to context, workspace, policy, session, or model route?
8. Pick one changed or surprising output. Which harness surface would you inspect or improve first, and what smallest change would you make?
9. Which surfaces should be saved for deeper Guardrails, Eval, Multi-Agent, or Skills chapters?

## References

* [OpenAI Harness Engineering](https://openai.com/index/harness-engineering/)
* [Martin Fowler: Harness engineering for coding agent users](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
* [OpenClaw Agent Runtimes](https://docs.openclaw.ai/concepts/agent-runtimes)
* [OpenClaw Agent Harness Plugins](https://docs.openclaw.ai/plugins/sdk-agent-harness)
* [OpenClaw Agent Loop](https://docs.openclaw.ai/concepts/agent-loop)
* [OpenClaw Context](https://docs.openclaw.ai/context/)
* [OpenClaw Agent CLI](https://docs.openclaw.ai/cli/agent)
* [OpenClaw Infer CLI](https://docs.openclaw.ai/cli/infer)
* [OpenClaw Models CLI](https://docs.openclaw.ai/cli/models)
* [OpenClaw Provider Plugins](https://docs.openclaw.ai/plugins/sdk-provider-plugins)
* [OpenClaw DeepSeek Provider](https://docs.openclaw.ai/providers/deepseek)
* [OpenClaw Ollama Provider](https://docs.openclaw.ai/providers/ollama)
* [OpenClaw Docker](https://docs.openclaw.ai/install/docker)
* [Ollama Library](https://ollama.com/library)
* [Ollama qwen2.5-coder](https://ollama.com/library/qwen2.5-coder)
* [Ollama Llama 3.1](https://ollama.com/library/llama3.1)
* [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/serving/openai_compatible_server.html)
* [SGLang OpenAI-Compatible API](https://sgl-project-sglang-93.mintlify.app/backend/openai-compatible-api)
* [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

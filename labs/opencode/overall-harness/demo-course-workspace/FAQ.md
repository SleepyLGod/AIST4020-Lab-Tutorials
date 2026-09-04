# FAQ

## Do I have to use DeepSeek?

No. DeepSeek is one optional API route. You may use another configured OpenClaw provider if you have access.

## Do I have to use Ollama?

No. Ollama is one optional local-model route. You may use another local or self-hosted backend if OpenClaw can route to it.

## Can I use vLLM or SGLang?

Yes, if you expose the backend through an OpenAI-compatible endpoint or another OpenClaw-supported route. The notebook uses DeepSeek and Ollama because they are easy to explain in class.

## Why does the local model sometimes behave worse in the agent task?

The full agent task includes more context, tool descriptions, session state, and workspace instructions than a direct LLM call. A small local model may handle a short prompt but struggle with a full agent workload.

## Is this lab teaching RAG and memory from scratch?

No. Students have already learned those concepts. This lab shows where workspace grounding, memory notes, and tool use appear in a real OpenClaw run.

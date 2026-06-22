# P0-005 Thought-Channel Parser

## Scope

Prevent Gemma reasoning markers and hidden thought text from reaching OpenAI or
Anthropic clients, and prevent assistant history forwarded through the gateway
from carrying hidden thought blocks.

## Design

- Replace regex-only stripping with a small state-machine sanitizer.
- Support canonical Gemma channel markers:
  `<|channel>thought`, `<channel|>`, and `<|channel>final`.
- Preserve existing `<think>...</think>` and `<thought>...</thought>` behavior.
- Expose a non-stream helper for full payloads and a streaming sanitizer for
  chunk-boundary-safe OpenAI/Anthropic output.
- Apply sanitization to:
  - incoming assistant history in OpenAI and Anthropic request conversion,
  - non-stream OpenAI responses,
  - non-stream Anthropic response conversion,
  - OpenAI streaming chunks,
  - Anthropic streaming event conversion.
- Strip OpenAI `reasoning_content` and `logprobs` fields from public responses
  and forwarded assistant history until there is an explicit sanitized token
  metadata contract; these fields can otherwise carry hidden reasoning tokens.

## Acceptance Checks

- Canonical thought-channel examples return only final visible text.
- Plain responses remain unchanged.
- Streaming markers split across chunks do not leak tags or hidden text.
- Malformed channel output with a later final marker keeps the final answer.
- Gateway OpenAI and Anthropic surfaces expose only sanitized text.

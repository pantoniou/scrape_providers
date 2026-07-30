# OpenRouter vs OpenAI Responses API

Last verified: 2026-07-30

OpenRouter exposes a Responses-compatible endpoint:

```text
OpenAI:     POST https://api.openai.com/v1/responses
OpenRouter: POST https://openrouter.ai/api/v1/responses
```

The common request and response envelope is similar enough for basic text,
streaming, reasoning, function calls, and some MCP use. It is **not** safe to
assume that every OpenAI request field, output item, built-in tool, or lifecycle
operation works on OpenRouter. In particular, OpenAI's native shell protocol is
not part of OpenRouter's documented Responses compatibility surface.

This repository therefore records the OpenRouter Responses endpoint with these
tools:

- hosted: `web_search`
- caller-executed: `function_calling`, `mcp`

It deliberately does not record `local_shell`.

The catalog also records the behavioral distinction explicitly on each
Responses endpoint:

```yaml
# OpenAI
capabilities:
  shell_tool_supported: true
  server_side_conversation_state_supported: true
  previous_response_id_supported: true
  response_lifecycle_supported: true

# OpenRouter
capabilities:
  shell_tool_supported: false
  server_side_conversation_state_supported: false
  previous_response_id_supported: false
  response_lifecycle_supported: false
```

## Compatibility at a glance

| Area | OpenAI | OpenRouter | Porting consequence |
|---|---|---|---|
| Endpoint | `/v1/responses` | `/api/v1/responses` | Change the base URL. |
| Model IDs | OpenAI IDs such as `gpt-5.6` | Routed IDs such as `openai/gpt-5.6`, plus models from other vendors | Do not pass bare OpenAI IDs blindly. |
| API maturity | Native API | Compatibility API | Treat OpenRouter behavior as its own contract. |
| Conversation state | Can store responses and continue with `previous_response_id` | Documented as stateless; the client must replay the complete history | Do not rely on a response ID as the conversation store. |
| Create response | Supported | Supported | The common text-generation path is portable. |
| Retrieve/delete/cancel response and input-item APIs | Native response lifecycle APIs exist | The compatibility documentation centers on `POST /responses` | Feature-detect; do not assume the rest of `/v1/responses/*`. |
| Function tools | Responses `type: "function"` shape | Supported in the Responses shape | Most portable tool mechanism. |
| Native shell | `local_shell` and newer `shell` item families exist | Not documented as a supported native Responses tool | Translate shell into a function tool, or reject it before sending. |
| Other OpenAI built-ins | Includes OpenAI-hosted tools such as file search, code interpreter, image generation, and others | Availability differs; OpenRouter documents its own server tools, notably `openrouter:*` tools | Tool names and execution ownership are not interchangeable. |
| Web search | OpenAI `web_search` tool | OpenRouter recommends `openrouter:web_search`; its older web plugin is deprecated | Use a provider-specific tool definition. |
| MCP | Supported | Supported | Test the chosen routed model/provider; support can still vary. |
| Reasoning | OpenAI model-dependent reasoning configuration and items | Translates reasoning across routed models; supported effort values and returned items can vary | Preserve opaque reasoning items and test per model. |
| Routing | One provider | OpenRouter may route or fail over among upstream providers | Identical requests may encounter different upstream capabilities. |
| Errors | OpenAI error and response status model | Reduced OpenAI-style codes plus OpenRouter's top-level `error_type`; an error after generation starts may arrive in a `200` body or SSE event | Inspect HTTP status, response status/event, `error`, and `error_type`. |
| Extensions | OpenAI request fields | Adds `provider`, `models`, `plugins`, `route`, `cache_control`, `debug`, `trace`, `session_id`, and OpenRouter metadata headers | Keep extensions out of provider-neutral request types. |

## The shell incompatibility

“Shell tool call” can refer to three different wire protocols. They should not
be conflated.

### 1. OpenAI legacy local-shell items

An OpenAI client may advertise:

```json
{
  "type": "local_shell"
}
```

The model returns a `local_shell_call`, and the client sends a
`local_shell_call_output`. The output is a JSON-encoded string. Existing clients,
including coding-agent clients, may still implement this item family.

### 2. OpenAI shell items

The current OpenAI API also defines a `shell` tool:

```json
{
  "type": "shell",
  "environment": {
    "type": "local"
  }
}
```

The model returns a `shell_call` resembling:

```json
{
  "type": "shell_call",
  "call_id": "call_123",
  "action": {
    "commands": ["rg -n \"TODO\" src"],
    "timeout_ms": 10000,
    "max_output_length": 20000
  },
  "environment": {
    "type": "local"
  }
}
```

The caller executes the commands and returns structured output:

```json
{
  "type": "shell_call_output",
  "call_id": "call_123",
  "output": [
    {
      "stdout": "src/main.py:12:# TODO\n",
      "stderr": "",
      "outcome": {
        "type": "exit",
        "exit_code": 0
      }
    }
  ],
  "max_output_length": 20000
}
```

The important details are the item `type`, `call_id`, structured `action`,
structured output chunks, and exit/timeout outcome. This is not ordinary
function calling.

### 3. A portable shell-shaped function

OpenRouter documents generic function calling but not the two native OpenAI
shell item families above. A portable integration should expose the local
executor as a normal function on OpenRouter:

```json
{
  "type": "function",
  "name": "shell",
  "description": "Run a command in the caller's workspace.",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string"
      },
      "timeout_ms": {
        "type": "integer",
        "minimum": 1
      }
    },
    "required": ["command"],
    "additionalProperties": false
  }
}
```

OpenRouter will return a normal `function_call`:

```json
{
  "type": "function_call",
  "call_id": "call_123",
  "name": "shell",
  "arguments": "{\"command\":\"rg -n \\\"TODO\\\" src\",\"timeout_ms\":10000}"
}
```

Return the result as a normal `function_call_output`:

```json
{
  "type": "function_call_output",
  "call_id": "call_123",
  "output": "{\"stdout\":\"src/main.py:12:# TODO\\n\",\"stderr\":\"\",\"exit_code\":0}"
}
```

This fallback preserves the semantics of a caller-executed command, but it does
not make OpenRouter understand `shell_call` or `local_shell_call`. The adapter
must translate in both directions, and the tool result must use the `call_id`
from the function call.

## Function-call shape: Responses vs Chat Completions

OpenRouter supports both endpoints, but their tool schemas are different.
Responses function tools are flat:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get the weather.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"}
    },
    "required": ["city"]
  }
}
```

Chat Completions nests the definition under `function`:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the weather.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string"}
      },
      "required": ["city"]
    }
  }
}
```

Likewise, Responses uses input/output items such as `function_call` and
`function_call_output`; Chat Completions uses assistant `tool_calls` and
`role: "tool"` messages. Do not send the Chat Completions shape merely because
the same OpenRouter host accepts it at another endpoint.

## State and continuation

OpenAI can store a response and use its ID as server-side conversation state.
OpenRouter describes its Responses API as stateless: each request is independent
and the full conversation history must be included on subsequent calls.

For a portable agent loop:

1. Set `store: false`.
2. Keep the user input and every returned output item locally.
3. Append tool outputs using the matching `call_id`.
4. Replay that complete item history on the next request.
5. Preserve reasoning items verbatim when the API returns them; do not parse,
   synthesize, or discard opaque reasoning data needed for continuation.

Although OpenRouter's create schema exposes `previous_response_id`, its
statelessness warning means clients should not depend on OpenAI-style stored
response chaining without a targeted integration test.

## Tools and execution ownership

A provider-neutral tool registry should separate:

- **caller-executed tools**, where the API emits a request and the application
  performs the action;
- **provider-executed tools**, where OpenAI or OpenRouter performs the action;
- **MCP tools**, whose execution and approval flow may involve another server.

The same-looking capability does not imply the same tool type. For example,
OpenAI's `web_search` and OpenRouter's `openrouter:web_search` are
provider-executed tools with provider-specific definitions. OpenRouter also has
its own `openrouter:*` server-tool namespace. These must not be passed to OpenAI
unchanged.

Model routing adds another capability boundary. A field accepted by OpenRouter's
front door may be unsupported, transformed, or ignored by the selected upstream
model/provider. Select models using their advertised supported parameters, pin
provider routing when behavior must be stable, and run an actual tool round trip
as a readiness check.

## Errors and streaming

OpenRouter maps its internal failures into a smaller set of Responses-style
codes:

- `invalid_prompt`
- `rate_limit_exceeded`
- `image_content_policy_violation`
- `server_error`

It adds a top-level `error_type` to retain the more precise OpenRouter category.
For example, several authentication, provider availability, overload, timeout,
and server failures collapse to `server_error`.

An OpenRouter request can also start successfully and then fail during
generation. In that case the HTTP status may already be `200`, with the failure
represented in the response body or a terminal SSE event. A robust client must
not equate HTTP 200 or receipt of the first stream event with successful
completion.

## Recommended adapter boundary

Keep a provider-neutral internal representation, then lower it into the selected
wire protocol:

```text
agent tool request
        |
        +-- OpenAI Responses --> native shell / function / OpenAI built-in
        |
        +-- OpenRouter Responses
              +-- shell semantics --> ordinary function named "shell"
              +-- generic tool    --> Responses function
              +-- hosted search   --> OpenRouter server tool
              +-- MCP             --> OpenRouter Responses MCP
```

At minimum, the OpenRouter adapter should:

1. Rewrite the URL and model ID.
2. Force stateless history replay.
3. Reject or translate `local_shell` and `shell`.
4. Reject unsupported OpenAI built-ins rather than silently dropping them.
5. Translate provider-specific web-search tools.
6. Preserve `call_id` across every tool round trip.
7. Parse OpenRouter's `error_type` and terminal stream errors.
8. Keep OpenRouter routing, plugin, caching, tracing, and metadata fields in an
   OpenRouter-only extension object.
9. Capability-test each model/provider combination used in production.

## Sources

- [OpenRouter Responses API overview](https://openrouter.ai/docs/api_reference/responses/overview)
- [OpenRouter create-response reference](https://openrouter.ai/docs/api/api-reference/responses/create-responses)
- [OpenRouter Responses reasoning](https://openrouter.ai/docs/api_reference/responses/reasoning)
- [OpenRouter Responses web search](https://openrouter.ai/docs/api_reference/responses/web-search)
- [OpenRouter Responses errors](https://openrouter.ai/docs/api_reference/responses/error-handling)
- [OpenRouter server tools](https://openrouter.ai/docs/guides/features/server-tools/overview)
- [OpenAI create-response reference](https://developers.openai.com/api/reference/resources/responses/methods/create)

OpenRouter's Responses API and its server-tool catalog evolve independently of
OpenAI's API. Revalidate this document before treating a newly added tool as
wire-compatible.

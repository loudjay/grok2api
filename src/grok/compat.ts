function makeId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function safeJsonParse(raw: unknown): Record<string, unknown> {
  if (typeof raw !== "string") return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function readChoice(chat: Record<string, unknown>): Record<string, unknown> {
  const choices = Array.isArray(chat.choices) ? chat.choices : [];
  return (choices[0] as Record<string, unknown> | undefined) ?? {};
}

function readMessage(chat: Record<string, unknown>): Record<string, unknown> {
  const choice = readChoice(chat);
  return ((choice.message as Record<string, unknown> | undefined) ?? {});
}

function readToolCalls(chat: Record<string, unknown>): Array<Record<string, unknown>> {
  const message = readMessage(chat);
  return Array.isArray(message.tool_calls) ? (message.tool_calls as Array<Record<string, unknown>>) : [];
}

function responseUsage(chat: Record<string, unknown>): Record<string, unknown> {
  const usage = (chat.usage as Record<string, unknown> | undefined) ?? {};
  const completionDetails = (usage.completion_tokens_details as Record<string, unknown> | undefined) ?? {};
  return {
    input_tokens: Number(usage.prompt_tokens ?? 0),
    output_tokens: Number(usage.completion_tokens ?? 0),
    total_tokens: Number(usage.total_tokens ?? 0),
    output_tokens_details: {
      reasoning_tokens: Number(completionDetails.reasoning_tokens ?? 0),
    },
  };
}

export function buildResponsesJsonFromChat(
  model: string,
  chat: Record<string, unknown>,
): Record<string, unknown> {
  const message = readMessage(chat);
  const toolCalls = readToolCalls(chat);
  const thinking = String(message.reasoning_content ?? "").trim();
  const content = String(message.content ?? "");
  const output: Array<Record<string, unknown>> = [];

  if (thinking) {
    output.push({
      id: makeId("rs"),
      type: "reasoning",
      summary: [{ type: "summary_text", text: thinking }],
      status: "completed",
    });
  }

  if (toolCalls.length) {
    output.push(
      ...toolCalls.map((call) => ({
        id: makeId("fc"),
        type: "function_call",
        call_id: String(call.id ?? makeId("call")),
        name: String(((call.function as Record<string, unknown> | undefined) ?? {}).name ?? ""),
        arguments: String(((call.function as Record<string, unknown> | undefined) ?? {}).arguments ?? "{}"),
        status: "completed",
      })),
    );
  } else {
    output.push({
      id: makeId("msg"),
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: content, annotations: [] }],
      status: "completed",
    });
  }

  return {
    id: makeId("resp"),
    object: "response",
    created_at: Number(chat.created ?? Math.floor(Date.now() / 1000)),
    status: "completed",
    model,
    output,
    usage: responseUsage(chat),
  };
}

export function buildAnthropicJsonFromChat(
  model: string,
  chat: Record<string, unknown>,
): Record<string, unknown> {
  const message = readMessage(chat);
  const toolCalls = readToolCalls(chat);
  const content = toolCalls.length
    ? toolCalls.map((call) => ({
        type: "tool_use",
        id: String(call.id ?? makeId("call")),
        name: String(((call.function as Record<string, unknown> | undefined) ?? {}).name ?? ""),
        input: safeJsonParse(((call.function as Record<string, unknown> | undefined) ?? {}).arguments),
      }))
    : [{ type: "text", text: String(message.content ?? "") }];
  const usage = responseUsage(chat);

  return {
    id: makeId("msg"),
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: toolCalls.length ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: Number(usage.input_tokens ?? 0),
      output_tokens: Number(usage.output_tokens ?? 0),
    },
  };
}

function sse(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function parseChunk(line: string): Record<string, unknown> | null {
  if (!line.startsWith("data:")) return null;
  const payload = line.slice(5).trim();
  if (!payload || payload === "[DONE]") return payload === "[DONE]" ? { done: true } : null;
  try {
    return JSON.parse(payload) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function createResponsesStreamFromChatStream(
  source: ReadableStream<Uint8Array>,
  model: string,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const responseId = makeId("resp");
      const reasoningId = makeId("rs");
      const messageId = makeId("msg");
      const toolItems: Array<Record<string, unknown>> = [];
      const seenCalls = new Set<string>();
      let outputText = "";
      let thinkingText = "";
      let buffer = "";

      const emitCompleted = () => {
        const completed = toolItems.length
          ? {
              id: responseId,
              object: "response",
              created_at: Math.floor(Date.now() / 1000),
              status: "completed",
              model,
              output: thinkingText
                ? [
                    {
                      id: reasoningId,
                      type: "reasoning",
                      summary: [{ type: "summary_text", text: thinkingText }],
                      status: "completed",
                    },
                    ...toolItems,
                  ]
                : toolItems,
              usage: null,
            }
          : {
              id: responseId,
              object: "response",
              created_at: Math.floor(Date.now() / 1000),
              status: "completed",
              model,
              output: [
                ...(thinkingText
                  ? [
                      {
                        id: reasoningId,
                        type: "reasoning",
                        summary: [{ type: "summary_text", text: thinkingText }],
                        status: "completed",
                      },
                    ]
                  : []),
                {
                  id: messageId,
                  type: "message",
                  role: "assistant",
                  content: [{ type: "output_text", text: outputText, annotations: [] }],
                  status: "completed",
                },
              ],
              usage: null,
            };
        controller.enqueue(encoder.encode(sse("response.completed", { type: "response.completed", response: completed })));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      };

      controller.enqueue(
        encoder.encode(
          sse("response.created", {
            type: "response.created",
            response: {
              id: responseId,
              object: "response",
              created_at: Math.floor(Date.now() / 1000),
              status: "in_progress",
              model,
              output: [],
            },
          }),
        ),
      );

      const reader = source.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });
          let idx = buffer.indexOf("\n");
          while (idx >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            idx = buffer.indexOf("\n");
            if (!line) continue;
            const payload = parseChunk(line);
            if (!payload) continue;
            if (payload.done) {
              emitCompleted();
              controller.close();
              return;
            }
            const choice = readChoice(payload);
            const delta = ((choice.delta as Record<string, unknown> | undefined) ?? {});
            const reasoning = String(delta.reasoning_content ?? "");
            const content = String(delta.content ?? "");
            const deltaToolCalls = Array.isArray(delta.tool_calls)
              ? (delta.tool_calls as Array<Record<string, unknown>>)
              : [];

            if (reasoning) {
              thinkingText += reasoning;
              controller.enqueue(
                encoder.encode(
                  sse("response.reasoning_summary_text.delta", {
                    type: "response.reasoning_summary_text.delta",
                    item_id: reasoningId,
                    output_index: 0,
                    summary_index: 0,
                    delta: reasoning,
                  }),
                ),
              );
            }
            if (content) {
              outputText += content;
              controller.enqueue(
                encoder.encode(
                  sse("response.output_text.delta", {
                    type: "response.output_text.delta",
                    item_id: messageId,
                    output_index: thinkingText ? 1 : 0,
                    content_index: 0,
                    delta: content,
                  }),
                ),
              );
            }
            for (const call of deltaToolCalls) {
              const callId = String(call.id ?? "");
              if (!callId || seenCalls.has(callId)) continue;
              seenCalls.add(callId);
              const item = {
                id: makeId("fc"),
                type: "function_call",
                call_id: callId,
                name: String(((call.function as Record<string, unknown> | undefined) ?? {}).name ?? ""),
                arguments: String(((call.function as Record<string, unknown> | undefined) ?? {}).arguments ?? "{}"),
                status: "completed",
              };
              toolItems.push(item);
              controller.enqueue(
                encoder.encode(
                  sse("response.output_item.done", {
                    type: "response.output_item.done",
                    output_index: toolItems.length - 1 + (thinkingText ? 1 : 0),
                    item,
                  }),
                ),
              );
            }

            if (choice.finish_reason) {
              emitCompleted();
              controller.close();
              return;
            }
          }
        }
        emitCompleted();
        controller.close();
      } catch (error) {
        controller.error(error);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

export function createAnthropicStreamFromChatStream(
  source: ReadableStream<Uint8Array>,
  model: string,
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const messageId = makeId("msg");
      let buffer = "";
      let textStarted = false;
      let textIndex = 0;
      let outputTokens = 0;

      const startTextBlock = () => {
        if (textStarted) return;
        textStarted = true;
        controller.enqueue(
          encoder.encode(
            sse("content_block_start", {
              type: "content_block_start",
              index: textIndex,
              content_block: { type: "text", text: "" },
            }),
          ),
        );
      };

      controller.enqueue(
        encoder.encode(
          sse("message_start", {
            type: "message_start",
            message: {
              id: messageId,
              type: "message",
              role: "assistant",
              model,
              content: [],
              stop_reason: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          }),
        ),
      );

      const reader = source.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          buffer += decoder.decode(value, { stream: true });
          let idx = buffer.indexOf("\n");
          while (idx >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            idx = buffer.indexOf("\n");
            if (!line) continue;
            const payload = parseChunk(line);
            if (!payload) continue;
            if (payload.done) {
              controller.enqueue(
                encoder.encode(
                  sse("message_delta", {
                    type: "message_delta",
                    delta: { stop_reason: "end_turn", stop_sequence: null },
                    usage: { output_tokens: outputTokens },
                  }),
                ),
              );
              if (textStarted) {
                controller.enqueue(encoder.encode(sse("content_block_stop", { type: "content_block_stop", index: textIndex })));
              }
              controller.enqueue(encoder.encode(sse("message_stop", { type: "message_stop" })));
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              controller.close();
              return;
            }
            const choice = readChoice(payload);
            const delta = ((choice.delta as Record<string, unknown> | undefined) ?? {});
            const content = String(delta.content ?? "");
            const deltaToolCalls = Array.isArray(delta.tool_calls)
              ? (delta.tool_calls as Array<Record<string, unknown>>)
              : [];

            if (content) {
              startTextBlock();
              outputTokens += Math.max(1, Math.ceil(content.length / 4));
              controller.enqueue(
                encoder.encode(
                  sse("content_block_delta", {
                    type: "content_block_delta",
                    index: textIndex,
                    delta: { type: "text_delta", text: content },
                  }),
                ),
              );
            }

            for (const call of deltaToolCalls) {
              if (textStarted) {
                controller.enqueue(encoder.encode(sse("content_block_stop", { type: "content_block_stop", index: textIndex })));
                textStarted = false;
              }
              textIndex += 1;
              const toolName = String(((call.function as Record<string, unknown> | undefined) ?? {}).name ?? "");
              const argumentsText = String(((call.function as Record<string, unknown> | undefined) ?? {}).arguments ?? "{}");
              outputTokens += Math.max(1, Math.ceil(argumentsText.length / 4));
              controller.enqueue(
                encoder.encode(
                  sse("content_block_start", {
                    type: "content_block_start",
                    index: textIndex,
                    content_block: {
                      type: "tool_use",
                      id: String(call.id ?? makeId("call")),
                      name: toolName,
                      input: {},
                    },
                  }),
                ),
              );
              controller.enqueue(
                encoder.encode(
                  sse("content_block_delta", {
                    type: "content_block_delta",
                    index: textIndex,
                    delta: { type: "input_json_delta", partial_json: argumentsText },
                  }),
                ),
              );
              controller.enqueue(encoder.encode(sse("content_block_stop", { type: "content_block_stop", index: textIndex })));
            }

            if (choice.finish_reason) {
              const finishReason = choice.finish_reason === "tool_calls" ? "tool_use" : "end_turn";
              if (textStarted) {
                controller.enqueue(encoder.encode(sse("content_block_stop", { type: "content_block_stop", index: textIndex })));
              }
              controller.enqueue(
                encoder.encode(
                  sse("message_delta", {
                    type: "message_delta",
                    delta: { stop_reason: finishReason, stop_sequence: null },
                    usage: { output_tokens: outputTokens },
                  }),
                ),
              );
              controller.enqueue(encoder.encode(sse("message_stop", { type: "message_stop" })));
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              controller.close();
              return;
            }
          }
        }
        controller.enqueue(
          encoder.encode(
            sse("message_delta", {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: outputTokens },
            }),
          ),
        );
        if (textStarted) {
          controller.enqueue(encoder.encode(sse("content_block_stop", { type: "content_block_stop", index: textIndex })));
        }
        controller.enqueue(encoder.encode(sse("message_stop", { type: "message_stop" })));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      } catch (error) {
        controller.error(error);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
      }
    },
  });
}

function normalizeResponseContent(content: unknown): unknown {
  if (!Array.isArray(content)) return content;
  const normalized: Array<Record<string, unknown>> = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const item = part as Record<string, unknown>;
    const partType = String(item.type ?? "").trim();
    if (partType === "input_text" || partType === "output_text") {
      normalized.push({ type: "text", text: item.text ?? "" });
      continue;
    }
    if (partType === "input_image" || partType === "image") {
      const source = (item.image_url as Record<string, unknown> | undefined) ?? (item.source as Record<string, unknown> | undefined) ?? {};
      const url = String(source.url ?? item.image_url ?? item.source ?? "").trim();
      if (url) normalized.push({ type: "image_url", image_url: { url } });
    }
  }
  return normalized;
}

export function parseResponsesInput(
  inputValue: unknown,
  instructions: unknown,
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  const prompt = String(instructions ?? "").trim();
  if (prompt) messages.push({ role: "system", content: prompt });
  if (typeof inputValue === "string") {
    messages.push({ role: "user", content: inputValue });
    return messages;
  }
  if (!Array.isArray(inputValue)) return messages;
  for (const item of inputValue) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const itemType = String(record.type ?? (record.role ? "message" : "")).trim();
    if (itemType === "function_call") {
      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: String(record.call_id ?? makeId("call")),
            type: "function",
            function: {
              name: String(record.name ?? ""),
              arguments: String(record.arguments ?? "{}"),
            },
          },
        ],
      });
      continue;
    }
    if (itemType === "function_call_output") {
      messages.push({
        role: "tool",
        tool_call_id: String(record.call_id ?? ""),
        content: String(record.output ?? ""),
      });
      continue;
    }
    if (itemType === "message") {
      messages.push({
        role: String(record.role ?? "user"),
        content: normalizeResponseContent(record.content),
      });
    }
  }
  return messages;
}

function normalizeAnthropicContent(role: string, content: unknown): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ role, content }];
  if (!Array.isArray(content)) return [];

  const toolResults = content.filter(
    (item) => item && typeof item === "object" && String((item as Record<string, unknown>).type ?? "") === "tool_result",
  );
  if (toolResults.length) {
    return toolResults.map((item) => {
      const record = item as Record<string, unknown>;
      const inner = Array.isArray(record.content) ? record.content : [];
      const text = inner
        .filter((part) => part && typeof part === "object" && String((part as Record<string, unknown>).type ?? "") === "text")
        .map((part) => String((part as Record<string, unknown>).text ?? ""))
        .join("\n");
      return { role: "tool", tool_call_id: String(record.tool_use_id ?? ""), content: text };
    });
  }

  const textParts: Array<Record<string, unknown>> = [];
  const toolCalls: Array<Record<string, unknown>> = [];
  for (const item of content) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const itemType = String(record.type ?? "").trim();
    if (itemType === "text") {
      textParts.push({ type: "text", text: record.text ?? "" });
      continue;
    }
    if (itemType === "tool_use") {
      toolCalls.push({
        id: String(record.id ?? makeId("call")),
        type: "function",
        function: {
          name: String(record.name ?? ""),
          arguments: JSON.stringify(record.input ?? {}),
        },
      });
      continue;
    }
    if (itemType === "image") {
      const source = (record.source as Record<string, unknown> | undefined) ?? {};
      if (String(source.type ?? "") === "base64") {
        textParts.push({
          type: "image_url",
          image_url: {
            url: `data:${String(source.media_type ?? "image/jpeg")};base64,${String(source.data ?? "")}`,
          },
        });
      }
    }
  }

  if (toolCalls.length) return [{ role: "assistant", content: null, tool_calls: toolCalls }];
  if (textParts.length) return [{ role, content: textParts }];
  return [];
}

function normalizeAnthropicSystem(system: unknown): string {
  if (typeof system === "string") return system.trim();
  if (!Array.isArray(system)) return "";
  return system
    .filter((item) => item && typeof item === "object" && String((item as Record<string, unknown>).type ?? "") === "text")
    .map((item) => String((item as Record<string, unknown>).text ?? ""))
    .join("\n")
    .trim();
}

export function parseAnthropicInput(
  messages: unknown,
  system: unknown,
): Array<Record<string, unknown>> {
  const internal: Array<Record<string, unknown>> = [];
  const systemText = normalizeAnthropicSystem(system);
  if (systemText) internal.push({ role: "system", content: systemText });
  if (!Array.isArray(messages)) return internal;
  for (const item of messages) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const role = String(record.role ?? "user").trim() || "user";
    internal.push(...normalizeAnthropicContent(role, record.content));
  }
  return internal;
}

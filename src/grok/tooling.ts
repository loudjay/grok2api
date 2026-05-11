export interface ParsedToolCall {
  call_id: string;
  name: string;
  arguments: string;
}

const TOOL_SYSTEM_HEADER = `You have access to the following tools.

AVAILABLE TOOLS:
{tool_definitions}

TOOL CALL FORMAT:
- When calling a tool, output ONLY the XML block below.
- <parameters> must be a single-line valid JSON object.
- Put multiple tool calls inside ONE <tool_calls> element.

<tool_calls>
  <tool_call>
    <tool_name>TOOL_NAME</tool_name>
    <parameters>{"key":"value"}</parameters>
  </tool_call>
</tool_calls>

{tool_choice_instruction}`;

const OPEN_TAG_RE = /<tool_calls[\s>]?/i;
const CLOSE_TAG_RE = /<\/tool_calls\s*>/i;
const TOOL_SYNTAX_RE = /<tool_calls|<tool_call|<function_call|<invoke\s|"tool_calls"\s*:|\btool_calls\b/i;
const XML_ROOT_RE = /<tool_calls\s*>([\s\S]*?)<\/tool_calls\s*>/i;
const XML_CALL_RE = /<tool_call\s*>([\s\S]*?)<\/tool_call\s*>/gi;
const XML_NAME_RE = /<tool_name\s*>([\s\S]*?)<\/tool_name\s*>/i;
const XML_PARAMS_RE = /<parameters\s*>([\s\S]*?)<\/parameters\s*>/i;

export function extractToolNames(tools: Array<Record<string, unknown>>): string[] {
  return tools
    .map((tool) => String((tool.function as Record<string, unknown> | undefined)?.name ?? "").trim())
    .filter(Boolean);
}

export function normalizeResponsesTools(tools: Array<Record<string, unknown>> | null | undefined): Array<Record<string, unknown>> {
  return (tools ?? []).map((tool) => {
    if (tool.type === "function" && !tool.function && tool.name) {
      return {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description ?? "",
          parameters: tool.parameters,
        },
      };
    }
    return tool;
  });
}

export function normalizeAnthropicTools(tools: Array<Record<string, unknown>> | null | undefined): Array<Record<string, unknown>> {
  return (tools ?? []).map((tool) => ({
    type: "function",
    function: {
      name: tool.name ?? "",
      description: tool.description ?? "",
      parameters: tool.input_schema,
    },
  }));
}

export function normalizeAnthropicToolChoice(choice: unknown): unknown {
  if (!choice || typeof choice === "string") return choice ?? "auto";
  if (typeof choice !== "object") return "auto";
  const record = choice as Record<string, unknown>;
  const type = String(record.type ?? "auto").trim();
  if (type === "any") return "required";
  if (type === "tool") {
    return { type: "function", function: { name: String(record.name ?? "") } };
  }
  return type || "auto";
}

export function buildToolSystemPrompt(tools: Array<Record<string, unknown>>, toolChoice: unknown): string {
  const definitions = tools
    .map((tool) => {
      const fn = (tool.function as Record<string, unknown> | undefined) ?? {};
      const lines = [`Tool: ${String(fn.name ?? "").trim()}`];
      const description = String(fn.description ?? "").trim();
      if (description) lines.push(`Description: ${description}`);
      if (fn.parameters !== undefined) lines.push(`Parameters: ${JSON.stringify(fn.parameters)}`);
      return lines.join("\n");
    })
    .join("\n\n");
  return TOOL_SYSTEM_HEADER
    .replace("{tool_definitions}", definitions)
    .replace("{tool_choice_instruction}", buildToolChoiceInstruction(toolChoice));
}

export function injectToolPrompt(prompt: string, systemPrompt: string): string {
  return `[system]: ${systemPrompt}\n\n${prompt}`;
}

export function toolCallsToXml(toolCalls: Array<Record<string, unknown>>): string {
  const lines = ["<tool_calls>"];
  for (const call of toolCalls) {
    const fn = (call.function as Record<string, unknown> | undefined) ?? {};
    let args = String(fn.arguments ?? "{}").trim() || "{}";
    try {
      args = JSON.stringify(JSON.parse(args));
    } catch {
      // keep original text
    }
    lines.push("  <tool_call>");
    lines.push(`    <tool_name>${String(fn.name ?? "").trim()}</tool_name>`);
    lines.push(`    <parameters>${args}</parameters>`);
    lines.push("  </tool_call>");
  }
  lines.push("</tool_calls>");
  return lines.join("\n");
}

export function parseToolCalls(text: string, availableTools?: string[]): ParsedToolCall[] {
  if (!text.trim() || !TOOL_SYNTAX_RE.test(text)) return [];
  const parsed = parseXmlToolCalls(text);
  if (!availableTools?.length) return parsed;
  return parsed.filter((call) => availableTools.includes(call.name));
}

export class ToolSieve {
  private readonly toolNames: string[];
  private buffer = "";
  private capturing = false;
  private done = false;

  constructor(toolNames: string[]) {
    this.toolNames = toolNames;
  }

  feed(chunk: string): { safeText: string; calls: ParsedToolCall[] | null } {
    if (this.done || !chunk) {
      return { safeText: this.capturing ? "" : chunk, calls: null };
    }
    if (this.capturing) return this.feedCapturing(chunk);
    return this.feedScanning(chunk);
  }

  flush(): ParsedToolCall[] {
    if (this.done || !this.buffer) return [];
    this.done = true;
    const calls = parseToolCalls(this.buffer, this.toolNames);
    this.buffer = "";
    return calls;
  }

  private feedScanning(chunk: string): { safeText: string; calls: ParsedToolCall[] | null } {
    const combined = this.buffer + chunk;
    this.buffer = "";
    const match = combined.match(OPEN_TAG_RE);
    if (!match || match.index === undefined) {
      const { safeText, leftover } = splitAtBoundary(combined, "<tool_calls");
      this.buffer = leftover;
      return { safeText, calls: null };
    }
    const safeText = combined.slice(0, match.index);
    this.buffer = combined.slice(match.index);
    this.capturing = true;
    const result = this.feedCapturing("");
    return { safeText, calls: result.calls };
  }

  private feedCapturing(chunk: string): { safeText: string; calls: ParsedToolCall[] | null } {
    this.buffer += chunk;
    const match = this.buffer.match(CLOSE_TAG_RE);
    if (!match || match.index === undefined) return { safeText: "", calls: null };
    const xmlBlock = this.buffer.slice(0, match.index + match[0].length);
    this.buffer = "";
    this.capturing = false;
    this.done = true;
    return { safeText: "", calls: parseToolCalls(xmlBlock, this.toolNames) };
  }
}

function buildToolChoiceInstruction(toolChoice: unknown): string {
  if (toolChoice === undefined || toolChoice === null || toolChoice === "auto") {
    return "WHEN TO CALL: Call a tool only when it is clearly needed. Otherwise respond in plain text.";
  }
  if (toolChoice === "none") {
    return "WHEN TO CALL: Do NOT call any tools. Respond in plain text only.";
  }
  if (toolChoice === "required") {
    return "WHEN TO CALL: You MUST output a <tool_calls> XML block.";
  }
  if (typeof toolChoice === "object") {
    const record = toolChoice as Record<string, unknown>;
    if (record.type === "function") {
      const name = String((record.function as Record<string, unknown> | undefined)?.name ?? "").trim();
      if (name) {
        return `WHEN TO CALL: You MUST call the tool named "${name}" and output ONLY a <tool_calls> XML block.`;
      }
    }
  }
  return "WHEN TO CALL: Call a tool only when it is clearly needed. Otherwise respond in plain text.";
}

function parseXmlToolCalls(text: string): ParsedToolCall[] {
  const root = text.match(XML_ROOT_RE);
  if (!root) return [];
  const calls: ParsedToolCall[] = [];
  const matches = (root[1] ?? "").matchAll(XML_CALL_RE);
  for (const match of matches) {
    const inner = match[1] ?? "";
    const name = (inner.match(XML_NAME_RE)?.[1] ?? "").trim();
    const parametersRaw = (inner.match(XML_PARAMS_RE)?.[1] ?? "{}").trim() || "{}";
    if (!name) continue;
    calls.push({
      call_id: `call_${Date.now()}_${crypto.randomUUID().slice(0, 6)}`,
      name,
      arguments: normalizeJson(parametersRaw),
    });
  }
  return calls;
}

function normalizeJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw));
  } catch {
    return raw;
  }
}

function splitAtBoundary(text: string, prefix: string): { safeText: string; leftover: string } {
  const limit = Math.min(prefix.length - 1, text.length);
  for (let size = limit; size > 0; size -= 1) {
    if (text.endsWith(prefix.slice(0, size))) {
      return { safeText: text.slice(0, -size), leftover: text.slice(-size) };
    }
  }
  return { safeText: text, leftover: "" };
}

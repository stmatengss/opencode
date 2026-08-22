import { describe, expect, test } from "bun:test"
import { LLMEvent, type LLMEvent as Event } from "@opencode-ai/llm"
import { ToolCallParser } from "@/session/llm/tool-call-parser"

const tools = [
  {
    name: "search",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        limit: { type: "integer" },
        exact: { type: "boolean" },
        filters: { type: "object" },
        code: { $ref: "#/$defs/code" },
      },
      $defs: { code: { anyOf: [{ type: "string" }, { type: "null" }] } },
    },
  },
  {
    name: "open",
    inputSchema: { type: "object", properties: { id: { type: "integer" } } },
  },
]

describe("tool-call parser", () => {
  test("selects parsers explicitly", () => {
    expect(ToolCallParser.create("dots", tools)).toBeDefined()
    expect(ToolCallParser.create("hermes", tools)).toBeDefined()
    expect(ToolCallParser.create("qwen3_xml", tools)).toBeDefined()
    expect(ToolCallParser.create("unknown", tools)).toBeUndefined()
  })

  test("parses Dots XML, JSON fallback, and schema types", () => {
    const parser = ToolCallParser.create("dots", tools)!
    const parsed = parser.push(
      'before<dots_function_call><invoke name="search">' +
        '<parameter name="query">chairs</parameter><parameter name="limit">3</parameter>' +
        '<parameter name="exact">true</parameter><parameter name="filters">{"color":"red"}</parameter>' +
        '<parameter name="code">001</parameter>' +
        '</invoke><invoke name="open"><parameter name="id">7</parameter></invoke></dots_function_call>' +
        '<dots_function_call>{"name":"search","arguments":{"query":"tables"}}</dots_function_call>',
    )

    expect(parsed.text).toBe("before")
    expect(parsed.calls).toEqual([
      { name: "search", input: { query: "chairs", limit: 3, exact: true, filters: { color: "red" }, code: "001" } },
      { name: "open", input: { id: 7 } },
      { name: "search", input: { query: "tables" } },
    ])
  })

  test("parses Hermes tagged JSON", () => {
    const parser = ToolCallParser.create("hermes", tools)!
    expect(
      parser.push('<tool_call>{"name":"search","arguments":{"query":"chairs","limit":2}}</tool_call>').calls,
    ).toEqual([{ name: "search", input: { query: "chairs", limit: 2 } }])
  })

  test("parses Qwen3 XML", () => {
    const parser = ToolCallParser.create("qwen3_xml", tools)!
    expect(
      parser.push(
        "<tool_call><function=search><parameter=query>chairs</parameter><parameter=limit>2</parameter></function></tool_call>",
      ).calls,
    ).toEqual([{ name: "search", input: { query: "chairs", limit: 2 } }])
  })

  test("buffers markers split across chunks", () => {
    const parser = ToolCallParser.create("dots", tools)!
    expect(parser.push("visible<dots_func")).toEqual({ text: "visible", calls: [] })
    expect(
      parser.push(
        'tion_call><invoke name="search"><parameter name="query">chairs</parameter></invoke></dots_function_call>',
      ),
    ).toEqual({ text: "", calls: [{ name: "search", input: { query: "chairs" } }] })
  })

  test("surfaces malformed and unknown calls as ordinary text", () => {
    const parser = ToolCallParser.create("dots", tools)!
    expect(parser.push("<dots_function_call>garbage</dots_function_call>").text).toBe("garbage")
    expect(
      parser.push(
        '<dots_function_call><invoke name="ghost"><parameter name="query">chairs</parameter></invoke></dots_function_call>',
      ).text,
    ).toContain("ghost")
  })

  test("converts raw text calls after runtime normalization", () => {
    const adapter = ToolCallParser.createAdapter("dots", tools)!
    const output = events(adapter, [
      LLMEvent.stepStart({ index: 0 }),
      LLMEvent.textStart({ id: "text-1" }),
      LLMEvent.textDelta({ id: "text-1", text: "I'll check.<dots_func" }),
      LLMEvent.textDelta({
        id: "text-1",
        text: 'tion_call><invoke name="search"><parameter name="query">weather</parameter></invoke></dots_function_call>',
      }),
      LLMEvent.textEnd({ id: "text-1" }),
      LLMEvent.stepFinish({ index: 0, reason: "stop" }),
      LLMEvent.finish({ reason: "stop" }),
    ])

    expect(
      output
        .filter((item) => item.type === "text-delta")
        .map((item) => item.text)
        .join(""),
    ).toBe("I'll check.")
    expect(output.find((item) => item.type === "tool-call")).toMatchObject({
      type: "tool-call",
      name: "search",
      input: { query: "weather" },
    })
    expect(output.find((item) => item.type === "step-finish")).toMatchObject({ reason: "tool-calls" })
    expect(output.at(-1)).toMatchObject({ type: "finish", reason: "tool-calls" })
  })

  test("does not duplicate provider-native tool calls", () => {
    const adapter = ToolCallParser.createAdapter("dots", tools)!
    const output = events(adapter, [
      LLMEvent.textStart({ id: "text-1" }),
      LLMEvent.textDelta({
        id: "text-1",
        text: '<dots_function_call><invoke name="search"><parameter name="query">weather</parameter></invoke></dots_function_call>',
      }),
      LLMEvent.textEnd({ id: "text-1" }),
      LLMEvent.toolCall({ id: "native-1", name: "search", input: { query: "weather" } }),
      LLMEvent.stepFinish({ index: 0, reason: "tool-calls" }),
      LLMEvent.finish({ reason: "tool-calls" }),
    ])

    expect(output.filter((item) => item.type === "tool-call")).toEqual([
      expect.objectContaining({ id: "native-1", name: "search" }),
    ])
  })

  test("reports the final step reason after multiple steps", () => {
    const adapter = ToolCallParser.createAdapter("dots", tools)!
    const output = events(adapter, [
      LLMEvent.stepStart({ index: 0 }),
      LLMEvent.textStart({ id: "text-1" }),
      LLMEvent.textDelta({
        id: "text-1",
        text: '<dots_function_call><invoke name="search"><parameter name="query">weather</parameter></invoke></dots_function_call>',
      }),
      LLMEvent.textEnd({ id: "text-1" }),
      LLMEvent.stepFinish({ index: 0, reason: "stop" }),
      LLMEvent.stepStart({ index: 1 }),
      LLMEvent.textStart({ id: "text-2" }),
      LLMEvent.textDelta({ id: "text-2", text: "Sunny" }),
      LLMEvent.textEnd({ id: "text-2" }),
      LLMEvent.stepFinish({ index: 1, reason: "stop" }),
      LLMEvent.finish({ reason: "stop" }),
    ])

    expect(output.at(-1)).toMatchObject({ type: "finish", reason: "stop" })
  })
})

function events(adapter: { push(event: Event): ReadonlyArray<Event> }, input: ReadonlyArray<Event>) {
  return input.flatMap((event) => adapter.push(event))
}

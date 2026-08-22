import { LLMEvent, type LLMEvent as Event } from "@opencode-ai/llm"
import { asSchema, type Tool } from "ai"
import * as Stream from "effect/Stream"

export type ToolDefinition = { readonly name: string; readonly inputSchema: unknown }
export type ParsedCall = { readonly name: string; readonly input: Record<string, unknown> }
type ParsedChunk = { readonly text: string; readonly calls: ReadonlyArray<ParsedCall> }
type Decoder = (value: string, tools: ReadonlyArray<ToolDefinition>) => ReadonlyArray<ParsedCall>

const DOTS_CALL = /<invoke\s+name\s*=\s*([^>]+)>(.*?)<\/invoke>/gs
const DOTS_PARAMETER = /<parameter\s+name\s*=\s*([^>]+)>(.*?)<\/parameter>/gs
const QWEN_CALL = /<function\s*=\s*([^>]+)>(.*?)<\/function>/gs
const QWEN_PARAMETER = /<parameter\s*=\s*([^>]+)>(.*?)<\/parameter>/gs

const formats: Record<string, { start: string; end: string; decode: Decoder }> = {
  dots: { start: "<dots_function_call>", end: "</dots_function_call>", decode: decodeDots },
  hermes: { start: "<tool_call>", end: "</tool_call>", decode: decodeJson },
  qwen3_xml: { start: "<tool_call>", end: "</tool_call>", decode: decodeQwen },
}

export function create(id: string, tools: ReadonlyArray<ToolDefinition>) {
  const format = formats[id]
  if (!format) return
  let buffer = ""

  return {
    push(delta: string): ParsedChunk {
      buffer += delta
      const output: { text: string; calls: ParsedCall[] } = { text: "", calls: [] }
      while (buffer) {
        const start = buffer.indexOf(format.start)
        if (start === -1) {
          const overlap = partialOverlap(buffer, format.start)
          output.text += buffer.slice(0, buffer.length - overlap)
          buffer = buffer.slice(buffer.length - overlap)
          break
        }
        output.text += buffer.slice(0, start)
        const end = buffer.indexOf(format.end, start + format.start.length)
        if (end === -1) {
          buffer = buffer.slice(start)
          break
        }
        const value = buffer.slice(start + format.start.length, end).trim()
        buffer = buffer.slice(end + format.end.length)
        const calls = format.decode(value, tools)
        if (calls.length) output.calls.push(...calls)
        else output.text += value
      }
      return output
    },
    flush(): ParsedChunk {
      const output = { text: buffer, calls: [] }
      buffer = ""
      return output
    },
  }
}

export function transform<E>(input: {
  readonly id?: string
  readonly tools: Record<string, Tool>
  readonly stream: Stream.Stream<Event, E>
}) {
  if (!input.id || !formats[input.id] || !Object.keys(input.tools).length) return input.stream
  const tools = Object.entries(input.tools).map(([name, tool]) => ({
    name,
    inputSchema: asSchema(tool.inputSchema).jsonSchema,
  }))
  return Stream.suspend(() => {
    const adapter = createAdapter(input.id!, tools)!
    return input.stream.pipe(Stream.flatMap((event) => Stream.fromIterable(adapter.push(event))))
  })
}

export function createAdapter(id: string, tools: ReadonlyArray<ToolDefinition>) {
  const initial = create(id, tools)
  if (!initial) return
  let parser = initial
  let calls: ReadonlyArray<ParsedCall> = []
  let textStart: Extract<Event, { type: "text-start" }> | undefined
  let textOpen = false
  let native = false
  let converted = false

  return {
    push(event: Event): ReadonlyArray<Event> {
      if (event.type === "text-start") {
        textStart = event
        return []
      }
      if (event.type === "text-delta") return parsed(parser.push(event.text), event)
      if (event.type === "text-end") {
        const events = [...parsed(parser.flush(), event), ...(textOpen ? [event] : [])]
        textStart = undefined
        textOpen = false
        return events
      }
      if (event.type === "tool-call") native = true
      if (event.type === "step-start") converted = false
      if (event.type !== "step-finish" && event.type !== "finish") return [event]

      const events = finishStep()
      const found = events.some((item) => item.type === "tool-call")
      converted = event.type === "step-finish" ? found : found || converted
      return [...events, converted ? { ...event, reason: "tool-calls" as const } : event]
    },
  }

  function parsed(chunk: ParsedChunk, source: Extract<Event, { type: "text-delta" | "text-end" }>) {
    calls = [...calls, ...chunk.calls]
    if (!chunk.text) return []
    const start = textStart ?? LLMEvent.textStart({ id: source.id, providerMetadata: source.providerMetadata })
    const events: Event[] = textOpen ? [] : [start]
    events.push(LLMEvent.textDelta({ id: start.id, text: chunk.text, providerMetadata: source.providerMetadata }))
    textStart = start
    textOpen = true
    return events
  }

  function finishStep() {
    const textID = textStart?.id ?? `text-${crypto.randomUUID()}`
    const events: Event[] = [...parsed(parser.flush(), LLMEvent.textEnd({ id: textID }))]
    if (textOpen && textStart) events.push(LLMEvent.textEnd({ id: textStart.id }))
    if (!native) events.push(...calls.map((call) => LLMEvent.toolCall({ id: `call_${crypto.randomUUID()}`, ...call })))
    parser = create(id, tools)!
    calls = []
    textStart = undefined
    textOpen = false
    native = false
    return events
  }
}

function decodeDots(value: string, tools: ReadonlyArray<ToolDefinition>) {
  return value.startsWith("<invoke") ? decodeXml(value, tools, DOTS_CALL, DOTS_PARAMETER) : decodeJson(value, tools)
}

function decodeQwen(value: string, tools: ReadonlyArray<ToolDefinition>) {
  return decodeXml(value, tools, QWEN_CALL, QWEN_PARAMETER)
}

function decodeXml(
  value: string,
  tools: ReadonlyArray<ToolDefinition>,
  calls: RegExp,
  parameters: RegExp,
): ParsedCall[] {
  return [...value.matchAll(calls)].flatMap((match) => {
    const name = unquote(match[1])
    const tool = tools.find((item) => item.name === name)
    if (!tool) return []
    const schema = object(tool.inputSchema)
    const properties = object(schema.properties)
    const definitions = { ...object(schema.definitions), ...object(schema.$defs) }
    const input = Object.fromEntries(
      [...match[2].matchAll(parameters)].map((parameter) => {
        const key = unquote(parameter[1])
        return [key, valueFor(parameter[2].trim(), object(properties[key]), definitions)]
      }),
    )
    return [{ name, input }]
  })
}

function decodeJson(value: string, tools: ReadonlyArray<ToolDefinition>): ParsedCall[] {
  const call = object(parse(value))
  if (typeof call.name !== "string" || !tools.some((tool) => tool.name === call.name)) return []
  const input =
    typeof call.arguments === "string" ? object(parse(call.arguments)) : object(call.arguments ?? call.parameters)
  return [{ name: call.name, input }]
}

function valueFor(value: string, schema: Record<string, unknown>, definitions: Record<string, unknown>) {
  const type = schemaType(schema, definitions)
  if (type === "string") return value
  return parse(value) ?? value
}

function schemaType(schema: Record<string, unknown>, definitions: Record<string, unknown>, depth = 0): unknown {
  if (depth > 10) return
  if (schema.type !== undefined)
    return Array.isArray(schema.type) ? schema.type.find((item) => item !== "null") : schema.type
  if (typeof schema.$ref === "string" && schema.$ref.startsWith("#/$defs/"))
    return schemaType(object(definitions[schema.$ref.split("/").at(-1) ?? ""]), definitions, depth + 1)
  for (const key of ["anyOf", "oneOf", "allOf"] as const) {
    if (!Array.isArray(schema[key])) continue
    const type = schema[key]
      .map(object)
      .map((item) => schemaType(item, definitions, depth + 1))
      .find(Boolean)
    if (type) return type
  }
}

function partialOverlap(value: string, marker: string) {
  for (let length = Math.min(value.length, marker.length - 1); length > 0; length--) {
    if (marker.startsWith(value.slice(-length))) return length
  }
  return 0
}

function unquote(value: string) {
  const result = value.trim()
  return result.length >= 2 && result[0] === result.at(-1) && (result[0] === '"' || result[0] === "'")
    ? result.slice(1, -1)
    : result
}

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value))
    : {}
}

function parse(value: string): unknown {
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

export * as ToolCallParser from "./tool-call-parser"

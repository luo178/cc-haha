/**
 * OpenCode Adapter - 将 OpenAI-compatible API 适配为 Anthropic SDK 接口
 */

import OpenAI from 'openai'
import type Anthropic from '@anthropic-ai/sdk'
import { getSessionId } from '../../bootstrap/state.js'
import { logForDebugging } from '../../utils/debug.js'
import { randomUUID } from '../../utils/crypto.js'

const OPENCODE_HOST_PATTERN = 'opencode.ai'
const OPENCODE_BASE_URL = 'https://opencode.ai/zen/v1'

/**
 * Normalise a user-supplied base URL so it always ends with exactly one `/v1`.
 *
 * Handles:
 *   https://host          → https://host/v1
 *   https://host/         → https://host/v1
 *   https://host/v1       → https://host/v1
 *   https://host/v1/      → https://host/v1
 *   https://host/zen/v1   → https://host/zen/v1
 *   https://host/zen/v1/  → https://host/zen/v1
 *   https://host/zen      → https://host/zen/v1
 */
function normaliseBaseUrl(url: string): string {
  const stripped = url.replace(/\/+$/, '')
  if (stripped.endsWith('/v1')) return stripped
  return `${stripped}/v1`
}

// 使用用户配置的 base URL（如果配置了 opencode.ai 相关地址），否则使用默认值
function getOpencodeBaseUrl(): string {
  const userBaseUrl = process.env.ANTHROPIC_BASE_URL
  if (userBaseUrl && userBaseUrl.includes(OPENCODE_HOST_PATTERN)) {
    return normaliseBaseUrl(userBaseUrl)
  }
  return OPENCODE_BASE_URL
}

// opencode 免费模型列表
const OPENCODE_FREE_MODELS = [
  'mimo-v2-omni-free',
  'mimo-v2-pro-free',
  'nemotron-3-super-free',
  'mimo-v2-flash-free',
  'qwen3.6-plus-free',
  'trinity-large-preview-free',
  'minimax-m2.5-free',
  'minimax-m2.7-free',
] as const
// 默认使用 qwen3.6-plus-free
const OPENCODE_DEFAULT_MODEL = 'qwen3.6-plus-free'

/** Environment variable names used by the OpenCode provider */
const OPENCODE_ENV = {
  /** API key for OpenCode (falls back to 'public' if not set) */
  API_KEY: 'OPENCODE_API_KEY',
  /** Project identifier (falls back to 'opencode') */
  PROJECT: 'VITE_OPENCODE_PROJECT',
  /** Client identifier (falls back to 'cli') */
  CLIENT: 'OPENCODE_CLIENT',
} as const

/** HTTP header names used for OpenCode requests */
export const OPENCODE_HEADERS = {
  PROJECT: 'x-opencode-project',
  CLIENT: 'x-opencode-client',
  SESSION: 'x-opencode-session',
  REQUEST: 'x-opencode-request',
} as const

/**
 * Returns the default values for OpenCode environment variables.
 * Separated so consumers (client.ts, opencodeAdapter.ts) can read
 * the same defaults without duplication.
 */
function getOpencodeDefaults() {
  return {
    apiKey: process.env[OPENCODE_ENV.API_KEY] || 'public',
    project: process.env[OPENCODE_ENV.PROJECT] ?? 'opencode',
    client: process.env[OPENCODE_ENV.CLIENT] ?? 'cli',
  }
}

/**
 * Builds the OpenCode-specific headers object.
 * @param sessionId - Optional session ID (falls back to getSessionId())
 * @param requestId - Optional request ID (falls back to crypto.randomUUID())
 */
export function getOpencodeHeaders(sessionId?: string, requestId?: string) {
  const defaults = getOpencodeDefaults()
  return {
    [OPENCODE_HEADERS.PROJECT]: defaults.project,
    [OPENCODE_HEADERS.CLIENT]: defaults.client,
    [OPENCODE_HEADERS.SESSION]: sessionId ?? getSessionId(),
    [OPENCODE_HEADERS.REQUEST]: requestId ?? randomUUID(),
  }
}

/**
 * Resolves a requested model name to the actual model to use with OpenCode.
 * If the requested model is in the free list, use it; otherwise fall back
 * to the default model.
 */
function resolveOpencodeModel(requestedModel: string): string {
  return (OPENCODE_FREE_MODELS as readonly string[]).includes(requestedModel)
    ? requestedModel
    : OPENCODE_DEFAULT_MODEL
}

export function createOpencodeClient(): OpenAI {
  const defaults = getOpencodeDefaults()
  const sessionId = getOpencodeHeaders()['x-opencode-session']

  const client = new OpenAI({
    apiKey: defaults.apiKey,
    baseURL: getOpencodeBaseUrl(),
    defaultHeaders: {
      'Authorization': `Bearer ${defaults.apiKey}`,
      ...getOpencodeHeaders(),
    },
  })

  logForDebugging(
    `[API:opencode] Created OpenAI client for opencode - project: ${defaults.project}, session: ${sessionId}`,
  )

  return client
}

// ---------------------------------------------------------------------------
// Anthropic → OpenAI message conversion
// ---------------------------------------------------------------------------

type OpencodeToolBlock = {
  type: 'tool_use'
  id: string
  name: string
  input: Record<string, unknown>
}

type OpencodeThinkingBlock = {
  type: 'thinking'
  thinking: string
  signature: string
}

/**
 * Map an Anthropic role to the closest OpenAI role.
 * Anthropic: 'user' | 'assistant'
 * OpenAI:    'user' | 'assistant' | 'system' | 'developer' | 'tool'
 */
function mapRole(role: string): 'user' | 'assistant' {
  return role === 'assistant' ? 'assistant' : 'user'
}

/**
 * Enforce OpenAI's alternating-role rule: user/assistant/user/...
 * Consecutive messages with the same role are merged into one.
 * IMPORTANT: 'tool' role messages are NEVER merged — they must follow
 * an assistant message with tool_calls per OpenAI's API contract.
 */
function enforceAlternation(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []
  for (const msg of messages) {
    // tool-role messages must always pass through — they have strict ordering requirements
    if (msg.role === 'tool') {
      result.push(msg)
      continue
    }
    if (result.length > 0) {
      const prev = result[result.length - 1]!
      // Skip alternation if prev is tool role (tool→user/assistant is valid)
      if (prev.role === 'tool') {
        result.push(msg)
        continue
      }
      if (prev.role === msg.role) {
        // Merge: append content
        if (typeof prev.content === 'string' && typeof msg.content === 'string') {
          prev.content += `\n${msg.content}`
        } else if (Array.isArray(prev.content) && Array.isArray(msg.content)) {
          prev.content.push(...msg.content)
        } else if (typeof prev.content === 'string' && Array.isArray(msg.content)) {
          const textPart = { type: 'text' as const, text: prev.content }
          prev.content = [textPart, ...msg.content]
        } else if (Array.isArray(prev.content) && typeof msg.content === 'string') {
          prev.content.push({ type: 'text' as const, text: msg.content })
        }
        // Preserve tool_calls if either has them
        if ('tool_calls' in msg && msg.tool_calls) {
          ;(prev as OpenAI.Chat.ChatCompletionAssistantMessageParam).tool_calls = [
            ...((prev as OpenAI.Chat.ChatCompletionAssistantMessageParam).tool_calls ?? []),
            ...(msg.tool_calls as OpenAI.Chat.ChatCompletionMessageToolCall[]),
          ]
        }
        continue
      }
    }
    result.push(msg)
  }
  return result
}

/**
 * Convert a single Anthropic message to one or more OpenAI message params.
 * Handles text, images, tool_use, tool_result, and thinking blocks.
 */
function convertSingleMessage(
  msg: Anthropic.Messages.MessageParam,
  systemBlocks: string[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const partial: OpenAI.Chat.ChatCompletionMessageParam[] = []

  // --- String content shortcut ---
  if (typeof msg.content === 'string') {
    if (msg.content.trim().length > 0) {
      partial.push({
        role: mapRole(msg.role),
        content: msg.content,
      })
    }
    return partial
  }

  // --- Array content ---
  const textParts: string[] = []
  const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = []
  const imageParts: OpenAI.Chat.ChatCompletionContentPartImage[] = []
  const thinkingBlocks: OpencodeThinkingBlock[] = []

  for (const part of msg.content) {
    if (part.type === 'text') {
      // 过滤掉 system-reminder 内容
      if (!part.text.includes('<system-reminder>')) {
        textParts.push(part.text)
      }
    } else if (part.type === 'image') {
      // 处理图片内容，转换为 OpenAI 的 image_url 格式
      const imagePart = part as Anthropic.ImageBlockParam
      const source = imagePart.source
      if (source?.type === 'base64') {
        const mimeType = source.media_type || 'image/png'
        imageParts.push({
          type: 'image_url',
          image_url: {
            url: `data:${mimeType};base64,${source.data}`,
            detail: 'auto' as const,
          },
        })
      } else if (source?.type === 'url') {
        imageParts.push({
          type: 'image_url',
          image_url: {
            url: source.url,
            detail: 'auto' as const,
          },
        })
      }
    } else if (part.type === 'tool_use') {
      // 记录工具调用（assistant 的消息中）
      const toolPart = part as unknown as OpencodeToolBlock
      toolCalls.push({
        id: toolPart.id,
        type: 'function',
        function: {
          name: toolPart.name,
          arguments: JSON.stringify(toolPart.input || {}),
        },
      })
    } else if (part.type === 'tool_result') {
      // 工具结果需要作为单独的 message 添加（tool 角色）
      const toolResultContent =
        typeof part.content === 'string'
          ? part.content
          : JSON.stringify(part.content)
      partial.push({
        role: 'tool',
        tool_call_id: part.tool_use_id,
        content: toolResultContent,
      })
    } else if ((part as any).type === 'thinking') {
      // Collect thinking blocks — they'll be prepended as system hints
      const thinkingPart = part as unknown as OpencodeThinkingBlock
      thinkingBlocks.push(thinkingPart)
    }
  }

  // Merge text content
  const content = textParts.join('\n').trim()

  // If there are tool_calls, create an assistant message with tool_calls
  if (toolCalls.length > 0) {
    partial.push({
      role: 'assistant',
      content: content || null,
      tool_calls: toolCalls,
    })
  } else if (imageParts.length > 0) {
    // Multi-part: images + optional text
    const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [
      ...imageParts,
      ...(content ? [{ type: 'text' as const, text: content }] : []),
    ]
    partial.push({
      role: mapRole(msg.role),
      content: contentParts,
    })
  } else if (content.length > 0) {
    partial.push({
      role: mapRole(msg.role),
      content: content,
    })
  }

  // Thinking blocks are converted to system hints for the next turn
  for (const tb of thinkingBlocks) {
    systemBlocks.push(`[Thinking] ${tb.thinking}`)
  }

  return partial
}

/**
 * Convert Anthropic messages to OpenAI messages.
 * Also extracts any `params.system` content as a leading system message.
 * Enforces OpenAI's alternating-role rule.
 */
function convertAnthropicMessagesToOpenAI(
  messages: Anthropic.Messages.MessageParam[],
  systemParam?: Anthropic.MessageCreateParams['system'],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []
  const thinkingSystemHints: string[] = []

  for (const msg of messages) {
    result.push(...convertSingleMessage(msg, thinkingSystemHints))
  }

  // Build leading system message from params.system + thinking hints
  let systemContent = ''
  if (systemParam) {
    if (typeof systemParam === 'string') {
      systemContent = systemParam
    } else if (Array.isArray(systemParam)) {
      systemContent = systemParam
        .filter((b): b is Anthropic.TextBlockParam => b.type === 'text')
        .map(b => b.text)
        .join('\n')
    }
  }
  if (thinkingSystemHints.length > 0) {
    systemContent = systemContent
      ? `${systemContent}\n\n${thinkingSystemHints.join('\n\n')}`
      : thinkingSystemHints.join('\n\n')
  }
  if (systemContent.trim()) {
    result.unshift({ role: 'system', content: systemContent.trim() })
  }

  // Enforce alternating role rule (skip the system message at index 0)
  const systemMsg = result[0]?.role === 'system' ? result.shift() : null
  const alternated = enforceAlternation(result)
  if (systemMsg) alternated.unshift(systemMsg)

  // If result is empty, throw — sending a fake 'hello' is unsafe
  if (alternated.length === 0) {
    throw new Error(
      'No valid messages after conversion. Ensure at least one user message is provided.',
    )
  }

  return alternated
}

async function* convertOpenAIStreamToAnthropic(
  stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>,
  model: string,
): AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> {
  let messageId = ''
  let inputTokens = 0
  let outputTokens = 0
  let hasStarted = false
  let hasTextBlock = false
  let textBlockIndex = -1
  // Monotonically increasing block index — text and tools share this counter
  // to avoid index collisions (text at index 0, first tool at index 1, etc.)
  let nextBlockIndex = 0
  const toolBlockIndices: number[] = []
  const toolCallBuffers = new Map<number, { id: string; name: string; args: string }>()

  const sendCleanupEvents = function* (stopReason: string | null) {
    if (hasTextBlock && textBlockIndex >= 0) {
      yield {
        type: 'content_block_stop',
        index: textBlockIndex,
      } as Anthropic.Messages.RawMessageStreamEvent
    }
    for (const idx of toolBlockIndices) {
      yield { type: 'content_block_stop', index: idx } as Anthropic.Messages.RawMessageStreamEvent
    }
    yield {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    } as Anthropic.Messages.RawMessageStreamEvent
    yield { type: 'message_stop' } as Anthropic.Messages.RawMessageStreamEvent
  }

  try {
    for await (const chunk of stream) {
      if (!chunk || !chunk.choices || chunk.choices.length === 0) {
        continue
      }

      const choice = chunk.choices[0]
      const delta = choice?.delta

      if (!hasStarted) {
        hasStarted = true
        messageId = chunk.id || randomUUID()
        inputTokens = chunk.usage?.prompt_tokens || 0
        outputTokens = chunk.usage?.completion_tokens || 0
        yield {
          type: 'message_start',
          message: {
            id: messageId,
            type: 'message',
            role: 'assistant',
            content: [],
            model,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: inputTokens, output_tokens: outputTokens },
          },
        } as Anthropic.Messages.RawMessageStreamEvent
      }

      const textContent = (delta as OpenAI.Chat.ChatCompletionChunk.Choice.Delta & { content?: string })?.content
      if (textContent) {
        if (!hasTextBlock) {
          hasTextBlock = true
          textBlockIndex = nextBlockIndex
          yield {
            type: 'content_block_start',
            index: textBlockIndex,
            content_block: { type: 'text', text: '' },
          } as Anthropic.Messages.RawMessageStreamEvent
          nextBlockIndex++
        }
        yield {
          type: 'content_block_delta',
          index: textBlockIndex,
          delta: { type: 'text_delta', text: textContent },
        } as Anthropic.Messages.RawMessageStreamEvent
      }

      // Tool calls
      const toolCalls = (delta as OpenAI.Chat.ChatCompletionChunk.Choice.Delta & { tool_calls?: Array<OpenAI.Chat.ChatCompletionChunk.Choice.Delta.ToolCall & { index?: number }> })?.tool_calls
      if (toolCalls && Array.isArray(toolCalls)) {
        logForDebugging(`[API:opencode:stream] Received tool_calls: ${JSON.stringify(toolCalls)}, delta keys: ${Object.keys(delta || {})}`)
        for (const toolCall of toolCalls) {
          const openaiIndex = toolCall.index ?? 0
          const toolId = toolCall.id || `tool_${openaiIndex}`

          if (!toolCallBuffers.has(openaiIndex)) {
            const initialName = toolCall.function?.name || ''
            const initialArgs = toolCall.function?.arguments || ''
            const blockIndex = nextBlockIndex++
            toolBlockIndices.push(blockIndex)
            toolCallBuffers.set(openaiIndex, {
              id: toolId,
              name: initialName,
              args: initialArgs,
              blockIndex,
            } as any)

            const toolName = initialName || 'unknown_tool'
            yield {
              type: 'content_block_start',
              index: blockIndex,
              content_block: { type: 'tool_use', id: toolId, name: toolName, input: '' },
            } as Anthropic.Messages.RawMessageStreamEvent

            if (initialArgs) {
              yield {
                type: 'content_block_delta',
                index: blockIndex,
                delta: { type: 'input_json_delta', partial_json: initialArgs },
              } as Anthropic.Messages.RawMessageStreamEvent
            }
          } else {
            const buffer = toolCallBuffers.get(openaiIndex)!
            if (toolCall.function?.name) buffer.name = toolCall.function.name
            if (toolCall.function?.arguments) {
              buffer.args += toolCall.function.arguments
              yield {
                type: 'content_block_delta',
                index: (buffer as any).blockIndex,
                delta: { type: 'input_json_delta', partial_json: toolCall.function.arguments },
              } as Anthropic.Messages.RawMessageStreamEvent
            }
          }
        }
      }

      // Track tokens
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens || inputTokens
        outputTokens = chunk.usage.completion_tokens || outputTokens
      }

      // Finish
      if (choice?.finish_reason) {
        const stopReason = mapFinishReason(choice.finish_reason)
        yield* sendCleanupEvents(stopReason)
        return
      }
    }

    // Stream ended without finish_reason — send cleanup anyway
    if (hasStarted) {
      yield* sendCleanupEvents(null)
    }
  } catch (error) {
    // HTTP errors (502, etc.) or network failures
    const errorMsg = error instanceof Error ? error.message : String(error)
    logForDebugging(`[API:opencode:stream] Stream error: ${errorMsg}`)
    if (hasStarted) {
      yield* sendCleanupEvents(null)
    }
    throw error
  }
}

/**
 * Map OpenAI finish_reason to Anthropic stop_reason.
 *
 * OpenAI:   'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call'
 * Anthropic: 'end_turn' | 'max_tokens' | 'tool_use' | 'stop_sequence'
 */
function mapFinishReason(
  reason: NonNullable<OpenAI.Chat.Choice['finish_reason']>,
): Anthropic.Messages.RawMessageStreamEvent extends { delta?: { stop_reason?: infer T } }
  ? T
  : string {
  switch (reason) {
    case 'stop':
      return 'end_turn' as any
    case 'length':
      return 'max_tokens' as any
    case 'tool_calls':
    case 'function_call':
      return 'tool_use' as any
    case 'content_filter':
      return 'max_tokens' as any
    default:
      return 'end_turn' as any
  }
}

function getResponseDebugMeta(response?: Response) {
  if (!response) {
    return { status: 'unknown', xRequestId: 'n/a', requestId: 'n/a', cfRay: 'n/a' }
  }
  const headers = response.headers
  return {
    status: String(response.status),
    xRequestId: headers.get('x-request-id') ?? 'n/a',
    requestId: headers.get('request-id') ?? 'n/a',
    cfRay: headers.get('cf-ray') ?? 'n/a',
  }
}

function createChainablePromise<T>(
  promiseFn: () => Promise<{ data: T; response?: Response; request_id?: string }>,
): Promise<T> & {
  withResponse: () => Promise<{ data: T; response: Response; request_id: string }>
  asResponse: () => Promise<Response>
} {
  const basePromise = promiseFn()
  const promise = basePromise.then(result => result.data) as Promise<T> & {
    withResponse: () => Promise<{ data: T; response: Response; request_id: string }>
    asResponse: () => Promise<Response>
  }
  promise.withResponse = async () => {
    const result = await basePromise
    return {
      data: result.data,
      response: result.response ?? new Response(null, { status: 200 }),
      request_id: result.request_id ?? randomUUID(),
    }
  }
  promise.asResponse = async () => {
    const result = await basePromise
    return result.response ?? new Response(null, { status: 200 })
  }
  return promise
}

type CountTokensParams = {
  model: string
  messages: Anthropic.Messages.MessageParam[]
  tools?: Anthropic.Tool[]
  betas?: string[]
  thinking?: { type: string; budget_tokens?: number }
}

type CountTokensResult = { input_tokens: number }

type ModelEntry = { id: string; created: number; object: string; owned_by: string }

export class OpencodeAdapter {
  private client: OpenAI
  public messages: ReturnType<typeof this.createMessagesProxy>
  public beta: { messages: ReturnType<typeof this.createMessagesProxy> & { countTokens: (params: CountTokensParams) => Promise<CountTokensResult> } }
  public models: { list: (params?: { betas?: string[] }) => AsyncIterable<ModelEntry> }

  constructor() {
    this.client = createOpencodeClient()
    this.messages = this.createMessagesProxy()
    this.beta = {
      messages: Object.assign(this.createMessagesProxy(), {
        countTokens: async (params: { model: string; messages: Anthropic.Messages.MessageParam[]; tools?: Anthropic.Tool[]; betas?: string[]; thinking?: { type: string; budget_tokens?: number } }) => {
          // OpenCode doesn't expose a count-tokens endpoint, so convert messages
          // to OpenAI format and make a minimal request to estimate tokens.
          // Fall back to a rough character-based estimation if that fails.
          try {
            const model = resolveOpencodeModel(params.model)
            const openaiMessages = convertAnthropicMessagesToOpenAI(params.messages)
            const completion = await this.client.chat.completions.create({
              model,
              messages: openaiMessages,
              max_tokens: 1,
              temperature: 0,
            })
            const promptTokens = completion.usage?.prompt_tokens ?? 0
            return { input_tokens: promptTokens }
          } catch {
            // Rough estimation: ~4 chars per token for English text
            let totalChars = 0
            for (const msg of params.messages) {
              if (typeof msg.content === 'string') {
                totalChars += msg.content.length
              } else if (Array.isArray(msg.content)) {
                for (const part of msg.content) {
                  if (part.type === 'text') totalChars += part.text.length
                }
              }
            }
            return { input_tokens: Math.ceil(totalChars / 4) }
          }
        },
      }),
    }
    this.models = {
      list: async function* (_params?: { betas?: string[] }) {
        // OpenCode doesn't expose a model listing endpoint.
        // Return the known free models so that refreshModelCapabilities()
        // has something to cache.
        for (const modelId of OPENCODE_FREE_MODELS) {
          yield {
            id: modelId,
            created: 0,
            object: 'model',
            owned_by: 'opencode',
          }
        }
      },
    }
  }

  private createMessagesProxy() {
    const self = this
    return {
      create: (params: Anthropic.MessageCreateParams, options?: { signal?: AbortSignal; headers?: Record<string, string> }) => {
        return self.createMessages(params, options)
      },
    }
  }

  private createMessages(
    params: Anthropic.MessageCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): Promise<
    AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> & {
      usage: { input_tokens: number; output_tokens: number; cache_read_input_tokens: number | null; cache_creation_input_tokens: number | null }
    }
  > & {
    withResponse: () => Promise<{ data: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> & { usage: { input_tokens: number; output_tokens: number } }; response: Response; request_id: string }>
    asResponse: () => Promise<Response>
  } {
    const originalModel = params.model
    // 使用免费模型
    const model = resolveOpencodeModel(originalModel)
    const isStream = params.stream === true

    // 转换消息格式（含 system 参数处理）
    const messages = convertAnthropicMessagesToOpenAI(params.messages, params.system)

    // Convert tools from Anthropic format to OpenAI format
    const tools = params.tools
      ?.filter((tool): tool is Anthropic.Tool => 'name' in tool && 'input_schema' in tool)
      .map((tool) => ({
        type: 'function' as const,
        function: {
          name: tool.name,
          description: (tool as any).description,
          parameters: tool.input_schema ?? {},
        },
      }))

    // Map tool_choice from Anthropic to OpenAI format
    const toolChoice = mapToolChoice(params.tool_choice)

    // Map stop_sequences from Anthropic to OpenAI
    const stop = params.stop_sequences ?? undefined

    logForDebugging(`[API:opencode] Original model: ${originalModel}, Using free model: ${model}, stream: ${isStream}, messages: ${messages.length}, tools: ${tools?.length || 0}`)
    logForDebugging(`[API:opencode] Converted messages: ${JSON.stringify(messages.map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content.substring(0, 50) : '...', hasToolCalls: 'tool_calls' in m, toolCallId: 'tool_call_id' in m ? (m as any).tool_call_id : undefined })))}`)

    const promiseFn = async (): Promise<{ data: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>; response?: Response; request_id?: string }> => {
      const requestParams: OpenAI.Chat.ChatCompletionCreateParams = {
        model,
        messages,
        max_tokens: params.max_tokens || 4096,
        temperature: params.temperature ?? 0.7,
        stream: isStream,
      }

      // Add tools and tool_choice if provided
      if (tools && tools.length > 0) {
        requestParams.tools = tools
        if (toolChoice) {
          requestParams.tool_choice = toolChoice
        }
      }

      // Add stop sequences if provided
      if (stop && stop.length > 0) {
        requestParams.stop = stop
      }

      // Build RequestOptions for signal/headers (second argument to create)
      const hasSignal = !!options?.signal
      const hasHeaders = !!options?.headers
      const requestOptions = (hasSignal || hasHeaders)
        ? {
            signal: options?.signal,
            headers: options?.headers,
          }
        : undefined

      const openaiPromise = this.client.chat.completions.create(
        requestParams,
        requestOptions,
      )
      let openaiResponse: OpenAI.Chat.ChatCompletion | AsyncIterable<OpenAI.Chat.ChatCompletionChunk>
      let rawResponse: Response | undefined
      let requestId: string | undefined

      if ('withResponse' in (openaiPromise as any) && typeof (openaiPromise as any).withResponse === 'function') {
        const wrapped = await (openaiPromise as any).withResponse()
        openaiResponse = wrapped.data
        rawResponse = wrapped.response
        requestId = wrapped.request_id
        const debugMeta = getResponseDebugMeta(rawResponse)
        logForDebugging(
          `[API:opencode] Upstream response (withResponse) status=${debugMeta.status}, request_id=${requestId ?? 'n/a'}, x-request-id=${debugMeta.xRequestId}, request-id=${debugMeta.requestId}, cf-ray=${debugMeta.cfRay}`,
        )
      } else {
        openaiResponse = await openaiPromise
        logForDebugging(
          '[API:opencode] Upstream response metadata unavailable (OpenAI SDK promise has no withResponse)',
        )
      }

      if (isStream) {
        return {
          data: convertOpenAIStreamToAnthropic(openaiResponse as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>, model),
          response: rawResponse,
          request_id: requestId,
        }
      }

      // 非流式响应处理 — 返回一个既是 Message 又是 AsyncIterable 的 hybrid 对象
      // 下游代码可能同步访问 response.usage，也可能迭代事件流
      const completion = openaiResponse as OpenAI.Chat.ChatCompletion
      const choice = completion.choices[0]
      const content = choice?.message?.content || ''
      const toolCalls = choice?.message?.tool_calls
      const stopReason = choice?.finish_reason
        ? mapFinishReason(choice.finish_reason)
        : null

      // Build content blocks matching Anthropic's Message.content shape
      const contentBlocks: Anthropic.Messages.ContentBlock[] = []
      if (toolCalls && toolCalls.length > 0) {
        for (const tc of toolCalls) {
          let parsedInput: Record<string, unknown> = {}
          try {
            parsedInput = JSON.parse((tc as any).function?.arguments || '{}')
          } catch {
            parsedInput = {}
          }
          contentBlocks.push({
            type: 'tool_use',
            id: tc.id,
            name: (tc as any).function?.name ?? 'unknown',
            input: parsedInput,
          } as unknown as Anthropic.Messages.ContentBlock)
        }
      }
      if (content) {
        contentBlocks.push({
          type: 'text',
          text: content,
        })
      }

      const inputTokens = completion.usage?.prompt_tokens || 0
      const outputTokens = completion.usage?.completion_tokens || 0

      // Build the Message object — this is what downstream code accesses synchronously
      const messageObj: Anthropic.Messages.Message = {
        id: completion.id,
        type: 'message',
        role: 'assistant',
        content: contentBlocks,
        model: completion.model,
        stop_reason: stopReason as any,
        stop_sequence: null,
        usage: {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          cache_creation: null,
          inference_geo: null,
        },
        container: null,
      } as unknown as Anthropic.Messages.Message

      // Make it also iterable (for stream-compatible consumers)
      // IMPORTANT: must follow Anthropic's stream protocol where content_block_start
      // has EMPTY text/input, and content_block_delta delivers the actual content.
      // The consumer (claude.ts) resets text to '' on content_block_start (line ~2027)
      // and expects text_delta to fill it.
      const iterable: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> = {
        [Symbol.asyncIterator]: async function* () {
          // message_start — content is EMPTY (matching Anthropic's real stream)
          yield {
            type: 'message_start',
            message: {
              ...messageObj,
              content: [],
            },
          } as Anthropic.Messages.RawMessageStreamEvent

          // Deliver each content block via start/delta/stop
          for (let i = 0; i < contentBlocks.length; i++) {
            const block = contentBlocks[i]!
            if (block.type === 'tool_use') {
              const toolInput = (block as any).input || {}
              // content_block_start — input MUST be empty string
              yield {
                type: 'content_block_start',
                index: i,
                content_block: {
                  type: 'tool_use',
                  id: block.id,
                  name: block.name,
                  input: '',
                },
              } as Anthropic.Messages.RawMessageStreamEvent
              // input_json_delta — full JSON at once (non-streaming emulation)
              const inputStr = JSON.stringify(toolInput)
              if (inputStr) {
                yield {
                  type: 'content_block_delta',
                  index: i,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: inputStr,
                  },
                } as Anthropic.Messages.RawMessageStreamEvent
              }
              yield {
                type: 'content_block_stop',
                index: i,
              } as Anthropic.Messages.RawMessageStreamEvent
            } else if (block.type === 'text') {
              // content_block_start — text MUST be empty string
              yield {
                type: 'content_block_start',
                index: i,
                content_block: {
                  type: 'text',
                  text: '',
                },
              } as Anthropic.Messages.RawMessageStreamEvent
              // text_delta — full text at once (non-streaming emulation)
              yield {
                type: 'content_block_delta',
                index: i,
                delta: {
                  type: 'text_delta',
                  text: block.text,
                },
              } as Anthropic.Messages.RawMessageStreamEvent
              yield {
                type: 'content_block_stop',
                index: i,
              } as Anthropic.Messages.RawMessageStreamEvent
            }
          }

          // message_delta
          yield {
            type: 'message_delta',
            delta: {
              stop_reason: stopReason,
              stop_sequence: null,
            },
            usage: { input_tokens: inputTokens, output_tokens: outputTokens },
          } as Anthropic.Messages.RawMessageStreamEvent

          // message_stop
          yield { type: 'message_stop' } as Anthropic.Messages.RawMessageStreamEvent
        },
      }

      // Hybrid object: acts as both Message (for sync access) and AsyncIterable (for streaming)
      Object.assign(messageObj, {
        [Symbol.asyncIterator]: iterable[Symbol.asyncIterator],
      })

      return {
        data: messageObj as unknown as AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>,
        response: rawResponse,
        request_id: requestId,
      }
    }

    return createChainablePromise(promiseFn)
  }
}

/**
 * Map Anthropic tool_choice to OpenAI tool_choice format.
 *
 * Anthropic: 'auto' | 'any' | 'tool' | { type: 'tool', name: string }
 * OpenAI:    'auto' | 'none' | 'required' | { type: 'function', function: { name: string } }
 */
function mapToolChoice(
  toolChoice: Anthropic.MessageCreateParams['tool_choice'],
): OpenAI.Chat.ChatCompletionCreateParams['tool_choice'] {
  if (!toolChoice) return undefined

  if (typeof toolChoice === 'string') {
    switch (toolChoice) {
      case 'auto':
        return 'auto'
      case 'any':
        return 'required'
      case 'tool':
        return 'required'
      default:
        return 'auto'
    }
  }

  // { type: 'tool', name: string }
  if (toolChoice.type === 'tool' && toolChoice.name) {
    return {
      type: 'function',
      function: { name: toolChoice.name },
    }
  }

  return undefined
}

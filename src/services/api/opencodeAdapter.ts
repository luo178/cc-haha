/**
 * OpenCode 适配器：将 OpenAI-compatible API 适配为 Anthropic SDK 接口
 */

import OpenAI from 'openai'
import type Anthropic from '@anthropic-ai/sdk'
import { getSessionId } from '../../bootstrap/state.js'
import { logForDebugging } from '../../utils/debug.js'
import { randomUUID } from '../../utils/crypto.js'

const OPENCODE_HOST_PATTERN = 'opencode.ai'
const OPENCODE_BASE_URL = 'https://opencode.ai/zen/v1'

/**
 * 规范化用户提供的 base URL。
 * 如果没有版本段则补 `/v1`；
 * 如果已存在版本段（如 `/v2`、`/api/v1`）则保持不变。
 *
 * 示例：
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
  // 若 URL 已以版本化路径结尾（如 /v1、/v2、/api/v1），直接保留，
  // 避免盲目追加 /v1。
  if (/\/v\d+$/.test(stripped) || stripped.endsWith('/api/v1')) return stripped
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

/** OpenCode 相关环境变量名 */
const OPENCODE_ENV = {
  /** OpenCode API key（未配置时回退到 'public'） */
  API_KEY: 'OPENCODE_API_KEY',
  /** 项目标识（未配置时回退到 'opencode'） */
  PROJECT: 'VITE_OPENCODE_PROJECT',
  /** 客户端标识（未配置时回退到 'cli'） */
  CLIENT: 'OPENCODE_CLIENT',
  /** 客户端真实 IP */
  REAL_IP: 'OPENCODE_REAL_IP',
} as const

/** OpenCode 请求使用的 HTTP Header 名称 */
export const OPENCODE_HEADERS = {
  AUTH: 'Authorization',
  PROJECT: 'x-opencode-project',
  CLIENT: 'x-opencode-client',
  SESSION: 'x-opencode-session',
  REQUEST: 'x-opencode-request',
  REAL_IP: 'x-real-ip'
} as const

/** 获取 OpenCode 环境变量的默认值。 */
function getOpencodeEnv() {
  return {
    apiKey: process.env[OPENCODE_ENV.API_KEY] || 'public',
    project: process.env[OPENCODE_ENV.PROJECT] ?? 'opencode',
    client: process.env[OPENCODE_ENV.CLIENT] ?? 'cli',
    realIp: process.env[OPENCODE_ENV.REAL_IP] ?? '192.168.10.2',
  }
}

/**
 * 生成 OpenCode 请求头。
 * @param sessionId 可选的会话 ID
 * @param requestId 可选的请求 ID
 */
export function getOpencodeHeaders(sessionId?: string, requestId?: string) {
  const defaultEnv = getOpencodeEnv()
  return {
    [OPENCODE_HEADERS.PROJECT]: defaultEnv.project,
    [OPENCODE_HEADERS.CLIENT]: defaultEnv.client,
    [OPENCODE_HEADERS.REAL_IP]: defaultEnv.realIp,
    [OPENCODE_HEADERS.SESSION]: sessionId ?? getSessionId(),
    [OPENCODE_HEADERS.REQUEST]: requestId ?? randomUUID(),
  }
}

/**
 * 将请求模型解析为 OpenCode 实际可用模型。
 * 若在免费模型列表中则原样使用，否则回退到默认模型。
 */
function resolveOpencodeModel(requestedModel: string): string {
  const resolved = (OPENCODE_FREE_MODELS as readonly string[]).includes(requestedModel)
    ? requestedModel
    : OPENCODE_DEFAULT_MODEL
  if (resolved !== requestedModel) {
    logForDebugging(
      `[API:opencode] Model "${requestedModel}" not in free list, falling back to "${resolved}"`,
    )
  }
  return resolved
}

export function createOpencodeClient(): OpenAI {
  const defaultEnv = getOpencodeEnv()
  const opencodeHeaders = getOpencodeHeaders()
  // x-opencode-request 应为“请求级”而非“客户端级”。
  const { [OPENCODE_HEADERS.REQUEST]: _requestHeader, ...staticHeaders } = opencodeHeaders
  const sessionId = opencodeHeaders[OPENCODE_HEADERS.SESSION]

  const client = new OpenAI({
    apiKey: defaultEnv.apiKey,
    baseURL: getOpencodeBaseUrl(),
    defaultHeaders: {
      ...staticHeaders,
    },
  })

  logForDebugging(
    `[API:opencode] Created OpenAI client for opencode - project: ${defaultEnv.project}, session: ${sessionId}`,
  )

  return client
}

// ---------------------------------------------------------------------------
// Anthropic → OpenAI 消息转换
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

type ToolCallBufferEntry = {
  id: string
  name: string
  args: string
  blockIndex: number
  started: boolean
}

/**
 * 将 Anthropic role 映射到最接近的 OpenAI role。
 * Anthropic: 'user' | 'assistant'
 * OpenAI:    'user' | 'assistant' | 'system' | 'developer' | 'tool'
 */
function mapRole(role: string): 'user' | 'assistant' {
  return role === 'assistant' ? 'assistant' : 'user'
}

/**
 * 执行 OpenAI 的角色交替规则：user/assistant/user/...
 * 连续同角色消息会合并为一条。
 * 重要：'tool' 消息绝不合并，因为每条工具结果都必须保留独立
 * tool_call_id 才能正确配对。
 */
function enforceAlternation(
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []
  for (const msg of messages) {
    if (result.length > 0) {
      const prev = result[result.length - 1]!

      // 连续 tool 消息保持分离，确保每个 tool_call_id 不丢失。
      // 多工具轮次里常见 tool→tool。
      if (prev.role === 'tool' && msg.role === 'tool') {
        result.push(msg)
        continue
      }

      // tool→user/assistant 是合法序列，直接透传
      if (prev.role === 'tool') {
        result.push(msg)
        continue
      }

      // 同角色（user→user 或 assistant→assistant）：合并
      if (prev.role === msg.role) {
        // 合并内容：追加到上一条
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
        // 若任一消息带 tool_calls，合并后保留
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
 * 将单条 Anthropic 消息转换为一条或多条 OpenAI message 参数。
 * 支持处理 text、image、tool_use、tool_result、thinking。
 *
 * 重要：tool_result 会作为独立消息返回，以便放在 assistant
 * tool_calls 之后（而不是并入 user 消息）。
 */
function convertSingleMessage(
  msg: Anthropic.Messages.MessageParam,
  systemBlocks: string[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  // --- 字符串内容快捷路径 ---
  if (typeof msg.content === 'string') {
    if (msg.content.trim().length > 0) {
      return [{
        role: mapRole(msg.role),
        content: msg.content,
      }]
    }
    return []
  }

  // --- 数组内容路径 ---
  const textParts: string[] = []
  const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = []
  const imageParts: OpenAI.Chat.ChatCompletionContentPartImage[] = []
  const thinkingBlocks: OpencodeThinkingBlock[] = []
  const toolResults: OpenAI.Chat.ChatCompletionMessageParam[] = []

  for (const part of msg.content) {
    if (part.type === 'text') {
      const trimmed = part.text.trim()
      if (!(/^<system-reminder>/.test(trimmed) && /<\/system-reminder>$/.test(trimmed))) {
        textParts.push(part.text)
      }
    } else if (part.type === 'image') {
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
      const toolResultContent =
        typeof part.content === 'string'
          ? part.content
          : JSON.stringify(part.content)
      toolResults.push({
        role: 'tool',
        tool_call_id: part.tool_use_id,
        content: toolResultContent,
      })
    } else if ((part as any).type === 'thinking') {
      const thinkingPart = part as unknown as OpencodeThinkingBlock
      thinkingBlocks.push(thinkingPart)
    }
  }

  const content = textParts.join('\n').trim()

  // 按正确顺序构造消息：
  // 若同一条 Anthropic 消息同时包含 tool_calls 与 toolResults，
  // 输出 assistant(tool_calls) → tool(role) 序列。
  // 这覆盖 Anthropic 中“单条 user 消息里交错 tool_result + text”的模式。
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []

  if (toolCalls.length > 0) {
    // 带 tool_calls 的 assistant 消息
    result.push({
      role: 'assistant',
      content: content || null,
      tool_calls: toolCalls,
    })
    // tool 结果紧随其后；逐条发出，保持与 tool_call 的对应关系
    result.push(...toolResults)
  } else if (toolResults.length > 0) {
    // 独立 tool 结果（该消息内没有 tool_calls）
    result.push(...toolResults)
    // tool 结果后的剩余文本
    if (content.length > 0) {
      result.push({ role: mapRole(msg.role), content })
    }
  } else if (imageParts.length > 0) {
    const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [
      ...imageParts,
      ...(content ? [{ type: 'text' as const, text: content }] : []),
    ]
    result.push({
      role: mapRole(msg.role),
      content: contentParts,
    })
  } else if (content.length > 0) {
    result.push({
      role: mapRole(msg.role),
      content,
    })
  }

  for (const tb of thinkingBlocks) {
    systemBlocks.push(`[Thinking] ${tb.thinking}`)
  }

  return result
}

/**
 * 将 Anthropic 消息数组转换为 OpenAI 消息数组。
 * 同时抽取 `params.system` 作为首条 system 消息，
 * 并执行 OpenAI 的角色交替规则。
 */
function convertAnthropicMessagesToOpenAI(
  messages: Anthropic.Messages.MessageParam[],
  systemParam?: Anthropic.MessageCreateParams['system'],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []
  const thinkingSystemHints: string[] = []

  for (const msg of messages) {
    const converted = convertSingleMessage(msg, thinkingSystemHints)
    result.push(...converted)
  }

  // 基于 params.system + thinking 提示构建前置 system 消息
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

  // 执行角色交替规则（跳过索引 0 的 system 消息）
  const systemMsg = result[0]?.role === 'system' ? result.shift() : null
  const alternated = enforceAlternation(result)
  if (systemMsg) alternated.unshift(systemMsg)

  // 若转换后为空则抛错；伪造 'hello' 存在风险
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
  // 统一递增的内容块索引：文本与工具共享同一计数器，
  // 避免索引冲突（如文本 0、首个工具 1）。
  let nextBlockIndex = 0
  const toolCallBuffers = new Map<number, ToolCallBufferEntry>()

  const sendCleanupEvents = function* (stopReason: string | null) {
    if (hasTextBlock && textBlockIndex >= 0) {
      yield {
        type: 'content_block_stop',
        index: textBlockIndex,
      } as Anthropic.Messages.RawMessageStreamEvent
    }
    for (const entry of toolCallBuffers.values()) {
      // 仅关闭已 start 的块；否则在流提前结束时会出现
      // 非法的 stop-before-start 序列。
      if (entry.started) {
        yield {
          type: 'content_block_stop',
          index: entry.blockIndex,
        } as Anthropic.Messages.RawMessageStreamEvent
      }
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

      // 工具调用
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
            toolCallBuffers.set(openaiIndex, {
              id: toolId,
              name: initialName,
              args: initialArgs,
              blockIndex,
              started: false, // 标记 content_block_start 是否已发出
            })

            // 若已有工具名，立即发 content_block_start。
            // 否则延迟到后续 chunk 拿到 name 再发（见下面 else 分支）。
            if (initialName) {
              toolCallBuffers.get(openaiIndex)!.started = true
              yield {
                type: 'content_block_start',
                index: blockIndex,
                content_block: { type: 'tool_use', id: toolId, name: initialName, input: '' },
              } as Anthropic.Messages.RawMessageStreamEvent
            }

            if (initialArgs) {
              // 确保首个 delta 前一定先发 content_block_start
              const buf = toolCallBuffers.get(openaiIndex)!
              if (!buf.started) {
                buf.started = true
                const nameToUse = buf.name || 'unknown_tool'
                yield {
                  type: 'content_block_start',
                  index: blockIndex,
                  content_block: { type: 'tool_use', id: toolId, name: nameToUse, input: '' },
                } as Anthropic.Messages.RawMessageStreamEvent
              }
              yield {
                type: 'content_block_delta',
                index: blockIndex,
                delta: { type: 'input_json_delta', partial_json: initialArgs },
              } as Anthropic.Messages.RawMessageStreamEvent
            }
          } else {
            const buffer = toolCallBuffers.get(openaiIndex)!
            if (toolCall.function?.name) {
              buffer.name = toolCall.function.name
            }
            // name 晚到且尚未 start 时，此处补发 content_block_start
            if (!buffer.started && buffer.name) {
              buffer.started = true
              yield {
                type: 'content_block_start',
                index: buffer.blockIndex,
                content_block: {
                  type: 'tool_use',
                  id: buffer.id,
                  name: buffer.name,
                  input: '',
                },
              } as Anthropic.Messages.RawMessageStreamEvent
            }
            if (toolCall.function?.arguments) {
              buffer.args += toolCall.function.arguments
              yield {
                type: 'content_block_delta',
                index: buffer.blockIndex,
                delta: { type: 'input_json_delta', partial_json: toolCall.function.arguments },
              } as Anthropic.Messages.RawMessageStreamEvent
            }
          }
        }
      }

      // 跟踪 token 用量
      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens || inputTokens
        outputTokens = chunk.usage.completion_tokens || outputTokens
      }

      // 完成态
      if (choice?.finish_reason) {
        const stopReason = mapFinishReason(choice.finish_reason)
        yield* sendCleanupEvents(stopReason)
        return
      }
    }

    // 流结束但未给 finish_reason，也要补发收尾事件
    if (hasStarted) {
      yield* sendCleanupEvents(null)
    }
  } catch (error) {
    // HTTP 错误（如 502）或网络失败
    const errorMsg = error instanceof Error ? error.message : String(error)
    logForDebugging(`[API:opencode:stream] Stream error: ${errorMsg}`)
    if (hasStarted) {
      yield* sendCleanupEvents(null)
    }
    throw error
  }
}

/**
 * 将 OpenAI finish_reason 映射为 Anthropic stop_reason。
 *
 * OpenAI:    'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call'
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
      return 'end_turn' as any
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
          // OpenCode 无 count-tokens 接口：先转为 OpenAI 消息，
          // 用最小请求估算；失败时回退到基于字符的粗略估算。
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
            // 启发式 token 估算（按语言加权）：
            // - ASCII 文本：约 4.5 字符 / token（英文占多）
            // - CJK 文本：约 1.5 字符 / token（中日韩）
            // 混合内容按加权平均估算。
            let asciiChars = 0
            let cjkChars = 0
            const cjkPattern = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/

            for (const msg of params.messages) {
              let text = ''
              if (typeof msg.content === 'string') {
                text = msg.content
              } else if (Array.isArray(msg.content)) {
                for (const part of msg.content) {
                  if (part.type === 'text') text += part.text
                  else if (part.type === 'tool_use' && 'input' in part) text += JSON.stringify(part.input)
                  else if (part.type === 'tool_result' && 'content' in part) {
                    if (typeof part.content === 'string') text += part.content
                    else if (Array.isArray(part.content)) {
                      for (const sub of part.content) {
                        if (sub.type === 'text') text += sub.text
                      }
                    }
                  }
                }
              }
              for (const ch of text) {
                if (cjkPattern.test(ch)) cjkChars++
                else asciiChars++
              }
            }
            return { input_tokens: Math.ceil(asciiChars / 4.5 + cjkChars / 1.5) }
          }
        },
      }),
    }
    this.models = {
      list: async function* (_params?: { betas?: string[] }) {
        // OpenCode 无模型列表接口；
        // 返回已知免费模型，供 refreshModelCapabilities() 缓存。
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
      usage?: { input_tokens: number; output_tokens: number; cache_read_input_tokens: number | null; cache_creation_input_tokens: number | null }
    }
  > & {
    withResponse: () => Promise<{ data: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> & { usage?: { input_tokens: number; output_tokens: number } }; response: Response; request_id: string }>
    asResponse: () => Promise<Response>
  } {
    const originalModel = params.model
    // 使用免费模型
    const model = resolveOpencodeModel(originalModel)
    const isStream = params.stream === true

    // 转换消息格式（含 system 参数处理）
    const messages = convertAnthropicMessagesToOpenAI(params.messages, params.system)

    // 将工具定义从 Anthropic 格式转换为 OpenAI 格式
    const tools = params.tools
      ?.filter((tool): tool is Anthropic.Tool => 'name' in tool && 'input_schema' in tool)
      .map((tool) => ({
        type: 'function' as const,
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.input_schema ?? {},
        },
      }))

    // 将 tool_choice 从 Anthropic 映射到 OpenAI 格式
    const toolChoice = mapToolChoice(params.tool_choice)

    // 将 stop_sequences 从 Anthropic 映射到 OpenAI
    const stop = params.stop_sequences ?? undefined

    logForDebugging(`[API:opencode] Original model: ${originalModel}, Using free model: ${model}, stream: ${isStream}, messages: ${messages.length}, tools: ${tools?.length || 0}`)
    logForDebugging(`[API:opencode] Converted messages: ${JSON.stringify(messages.map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content.substring(0, 50) : '...', hasToolCalls: 'tool_calls' in m, toolCallId: 'tool_call_id' in m ? (m as any).tool_call_id : undefined })))}`)

    const promiseFn = async (): Promise<{ data: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>; response?: Response; request_id?: string }> => {
      const requestParams: OpenAI.Chat.ChatCompletionCreateParams = {
        model,
        messages,
        max_tokens: params.max_tokens ?? 4096,
        temperature: params.temperature ?? 0.7,
        stream: isStream,
      }

      // 若提供 tools/tool_choice，则写入请求参数
      if (tools && tools.length > 0) {
        requestParams.tools = tools
        if (toolChoice) {
          requestParams.tool_choice = toolChoice
        }
      }

      // 若提供 stop_sequences，则写入请求参数
      if (stop && stop.length > 0) {
        requestParams.stop = stop
      }

      // 组装“请求级”OpenCode headers；request id 每次请求必须唯一。
      const requestHeaders = {
        ...getOpencodeHeaders(undefined, randomUUID()),
        ...(options?.headers ?? {}),
      }

      // 构造 create 的第二参数：RequestOptions（signal/headers）
      const hasSignal = !!options?.signal
      const hasHeaders = Object.keys(requestHeaders).length > 0
      const requestOptions = (hasSignal || hasHeaders)
        ? {
            signal: options?.signal,
            headers: requestHeaders,
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

      // 非流式响应处理：返回既可当 Message 用，也可迭代事件流的 hybrid 对象
      // 下游既可能同步读取 response.usage，也可能按流式协议消费事件
      const completion = openaiResponse as OpenAI.Chat.ChatCompletion
      const choice = completion.choices[0]
      const content = choice?.message?.content || ''
      const toolCalls = choice?.message?.tool_calls
      const stopReason = choice?.finish_reason
        ? mapFinishReason(choice.finish_reason)
        : null

      // 构造与 Anthropic Message.content 形状一致的 content blocks
      const contentBlocks: Anthropic.Messages.ContentBlock[] = []
      if (toolCalls && toolCalls.length > 0) {
        for (const tc of toolCalls) {
          let parsedInput: Record<string, unknown> = {}
          try {
            parsedInput = JSON.parse(tc.function?.arguments || '{}')
          } catch {
            parsedInput = {}
          }
          contentBlocks.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.function?.name ?? 'unknown',
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

      // 构造 Message 对象（供下游同步访问）
      const messageObj: Anthropic.Messages.Message & { _request_id?: string } = {
        id: completion.id,
        type: 'message',
        role: 'assistant',
        content: contentBlocks,
        model: completion.model,
        stop_reason: stopReason as any,
        stop_sequence: null,
        _request_id: requestId,
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

      // 同时让其可迭代（兼容按流式协议消费的调用方）
      // 重要：必须遵循 Anthropic 流协议：content_block_start 的 text/input 为空，
      // 实际内容通过 content_block_delta 下发。
      // 调用方（claude.ts）会在 content_block_start 时将 text 重置为 ''，
      // 并依赖后续 text_delta 填充内容。
      const iterable: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> = {
        [Symbol.asyncIterator]: async function* () {
          // message_start：content 为空（与 Anthropic 真实流一致）
          yield {
            type: 'message_start',
            message: {
              ...messageObj,
              content: [],
            },
          } as Anthropic.Messages.RawMessageStreamEvent

          // 逐个内容块按 start/delta/stop 输出
          for (let i = 0; i < contentBlocks.length; i++) {
            const block = contentBlocks[i]!
            if (block.type === 'tool_use') {
              const toolInput = (block as any).input || {}
              // content_block_start：input 必须为空字符串
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
              // input_json_delta：一次性下发完整 JSON（非流式模拟）
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
              // content_block_start：text 必须为空字符串
              yield {
                type: 'content_block_start',
                index: i,
                content_block: {
                  type: 'text',
                  text: '',
                },
              } as Anthropic.Messages.RawMessageStreamEvent
              // text_delta：一次性下发完整文本（非流式模拟）
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

      // Hybrid 对象：既可同步当 Message 用，也可按流式迭代
      Object.assign(messageObj, {
        [Symbol.asyncIterator]: iterable[Symbol.asyncIterator],
        withResponse: async () => ({
          data: messageObj,
          response: rawResponse ?? new Response(null, { status: 200 }),
          request_id: requestId ?? randomUUID(),
        }),
        asResponse: async () =>
          rawResponse ?? new Response(null, { status: 200 }),
      })

      return {
        data: messageObj as unknown as AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> & { withResponse: () => Promise<{ data: Anthropic.Messages.Message; response: Response; request_id: string }>; asResponse: () => Promise<Response> },
        response: rawResponse,
        request_id: requestId,
      }
    }

    return createChainablePromise(promiseFn)
  }
}

/**
 * 将 Anthropic tool_choice 映射为 OpenAI tool_choice。
 *
 * Anthropic：
 *   - 'auto' | 'any' | 'tool'
 *   - { type: 'auto' | 'any', disable_parallel_tool_use?: boolean }
 *   - { type: 'tool', name: string, disable_parallel_tool_use?: boolean }
 *
 * 说明：disable_parallel_tool_use 在 OpenAI 中无直接等价字段。
 * OpenAI: 'auto' | 'none' | 'required' | { type: 'function', function: { name: string } }
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

  // 对象形态：{ type: 'auto' | 'any' | 'tool', ... }
  if (typeof toolChoice === 'object' && 'type' in toolChoice) {
    if ('disable_parallel_tool_use' in toolChoice && toolChoice.disable_parallel_tool_use) {
      logForDebugging(
        '[API:opencode] tool_choice.disable_parallel_tool_use is not supported by OpenAI tool_choice and will be ignored',
      )
    }

    if (toolChoice.type === 'auto') return 'auto'
    if (toolChoice.type === 'any') return 'required'
    if (toolChoice.type === 'tool' && 'name' in toolChoice && toolChoice.name) {
      return {
        type: 'function',
        function: { name: toolChoice.name },
      }
    }
  }

  return undefined
}

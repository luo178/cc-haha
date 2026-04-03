/**
 * OpenCode Adapter - 将 OpenAI-compatible API 适配为 Anthropic SDK 接口
 */

import OpenAI from 'openai'
import type Anthropic from '@anthropic-ai/sdk'
import { getSessionId } from '../../bootstrap/state.js'
import { logForDebugging } from '../../utils/debug.js'

const OPENCODE_BASE_URL = 'https://opencode.ai/zen/v1'

// 使用用户配置的 base URL（如果配置了 opencode.ai 相关地址），否则使用默认值
function getOpencodeBaseUrl(): string {
  const userBaseUrl = process.env.ANTHROPIC_BASE_URL
  if (userBaseUrl && userBaseUrl.includes('opencode.ai')) {
    // 确保以 /v1 结尾（OpenAI 兼容接口）
    return userBaseUrl.endsWith('/v1') ? userBaseUrl : `${userBaseUrl.replace(/\/+$/, '')}/v1`
  }
  return OPENCODE_BASE_URL
}

// opencode 免费模型列表
const FREE_MODELS = [
  'mimo-v2-omni-free',
  'mimo-v2-pro-free',
  'nemotron-3-super-free',
  'mimo-v2-flash-free',
  'qwen3.6-plus-free',
  'trinity-large-preview-free',
  'minimax-m2.5-free',
  'minimax-m2.7-free',
]

// 默认使用 qwen3.6-plus-free
const DEFAULT_MODEL = 'qwen3.6-plus-free'

export function createOpencodeClient(): OpenAI {
  const apiKey = process.env.OPENCODE_API_KEY || 'public'
  const opencodeProject = process.env.VITE_OPENCODE_PROJECT ?? 'opencode'
  const opencodeClient = process.env.OPENCODE_CLIENT ?? 'cli'
  const sessionId = getSessionId()

  const client = new OpenAI({
    apiKey,
    baseURL: getOpencodeBaseUrl(),
    defaultHeaders: {
      'Authorization': `Bearer ${apiKey}`,
      'x-opencode-project': opencodeProject,
      'x-opencode-client': opencodeClient,
      'x-opencode-session': sessionId,
      'x-opencode-request': crypto.randomUUID(),
    },
  })

  logForDebugging(
    `[API:opencode] Created OpenAI client for opencode - project: ${opencodeProject}, session: ${sessionId}`,
  )

  return client
}

function convertAnthropicMessagesToOpenAI(
  messages: Anthropic.Messages.MessageParam[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.ChatCompletionMessageParam[] = []

  for (const msg of messages) {
    // 处理字符串内容
    if (typeof msg.content === 'string') {
      if (msg.content.trim().length > 0) {
        result.push({
          role: msg.role === 'assistant' ? 'assistant' : 'user',
          content: msg.content,
        })
      }
      continue
    }

    // 处理数组内容
    const textParts: string[] = []
    const toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = []
    const imageParts: OpenAI.Chat.ChatCompletionContentPartImage[] = []

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
        toolCalls.push({
          id: part.id,
          type: 'function',
          function: {
            name: part.name,
            arguments: JSON.stringify(part.input || {}),
          },
        })
      } else if (part.type === 'tool_result') {
        // 工具结果需要作为单独的 message 添加（user 角色）
        const toolResultContent = typeof part.content === 'string'
          ? part.content
          : JSON.stringify(part.content)
        result.push({
          role: 'tool',
          tool_call_id: part.tool_use_id,
          content: toolResultContent,
        })
      }
    }

    // 合并文本内容
    const content = textParts.join('\n').trim()

    // 如果有工具调用，创建 assistant message 带 tool_calls
    if (toolCalls.length > 0) {
      result.push({
        role: 'assistant',
        content: content || null,
        tool_calls: toolCalls,
      })
    } else if (imageParts.length > 0) {
      // 有图片内容，使用多部分内容格式
      const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [
        ...imageParts,
        ...(content ? [{ type: 'text' as const, text: content }] : []),
      ]
      result.push({
        role: msg.role === 'assistant' ? 'assistant' : 'user',
        content: contentParts,
      })
    } else if (content.length > 0) {
      // 普通文本消息
      result.push({
        role: msg.role === 'assistant' ? 'assistant' : 'user',
        content: content,
      })
    }
  }

  // 确保至少有一条消息
  if (result.length === 0) {
    result.push({ role: 'user', content: 'hello' })
  }

  return result
}

async function* convertOpenAIStreamToAnthropic(
  stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>,
  model: string,
): AsyncIterable<Anthropic.Messages.RawMessageStreamEvent> {
  let messageId = ''
  let outputTokens = 0
  let hasStarted = false
  let hasTextBlock = false
  let textBlockIndex = 0
  let toolBlockIndex = 1
  const toolCallBuffers = new Map<string, { id: string; name: string; args: string }>()

  for await (const chunk of stream) {
    // 安全检查 chunk 结构
    if (!chunk || !chunk.choices || chunk.choices.length === 0) {
      continue
    }

    const choice = chunk.choices[0]
    const delta = choice?.delta

    // 第一个 chunk，发送 message_start
    if (!hasStarted) {
      hasStarted = true
      messageId = chunk.id || crypto.randomUUID()
      const msgStart = {
        type: 'message_start',
        message: {
          id: messageId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      } as Anthropic.Messages.RawMessageStreamEvent
      logForDebugging(`[API:opencode:stream] Yielding message_start, type: ${msgStart.type}`)
      yield msgStart
    }

    // 处理文本内容
    const textContent = (delta as any)?.content
    if (textContent) {
      // 如果没有文本块，先发送 content_block_start
      if (!hasTextBlock) {
        hasTextBlock = true
        const blockStart = {
          type: 'content_block_start',
          index: textBlockIndex,
          content_block: { type: 'text', text: '' },
        } as Anthropic.Messages.RawMessageStreamEvent
        logForDebugging(`[API:opencode:stream] Yielding content_block_start (text), type: ${blockStart.type}`)
        yield blockStart
      }
      const blockDelta = {
        type: 'content_block_delta',
        index: textBlockIndex,
        delta: { type: 'text_delta', text: textContent },
      } as Anthropic.Messages.RawMessageStreamEvent
      logForDebugging(`[API:opencode:stream] Yielding content_block_delta (text), type: ${blockDelta.type}`)
      yield blockDelta
    }

    // 处理工具调用
    const toolCalls = (delta as any)?.tool_calls
    if (toolCalls && Array.isArray(toolCalls)) {
      logForDebugging(`[API:opencode:stream] Received tool_calls: ${JSON.stringify(toolCalls)}, delta keys: ${Object.keys(delta || {})}`)
      for (const toolCall of toolCalls) {
        const toolIndex = toolCall.index ?? 0
        const toolId = toolCall.id || `tool_${toolBlockIndex + toolIndex}`

        // 使用 index 作为 key 来跟踪工具调用（因为 id 可能延迟出现）
        const bufferKey = `${toolBlockIndex + toolIndex}`

        // 初始化或更新工具调用缓冲区
        if (!toolCallBuffers.has(bufferKey)) {
          const initialName = toolCall.function?.name || ''
          const initialArgs = toolCall.function?.arguments || ''
          toolCallBuffers.set(bufferKey, {
            id: toolId,
            name: initialName,
            args: initialArgs,
          })

          // 发送 tool_use 的 content_block_start（即使没有 name，也先发送）
          const toolName = initialName || 'unknown_tool'
          const toolBlockStart = {
            type: 'content_block_start',
            index: toolBlockIndex + toolIndex,
            content_block: {
              type: 'tool_use',
              id: toolId,
              name: toolName,
              input: {},
            },
          } as Anthropic.Messages.RawMessageStreamEvent
          logForDebugging(`[API:opencode:stream] Yielding content_block_start (tool_use), type: ${toolBlockStart.type}, tool: ${toolName}, index: ${toolBlockIndex + toolIndex}`)
          yield toolBlockStart

          // 如果初始就有参数，发送参数增量
          if (initialArgs) {
            const toolDelta = {
              type: 'content_block_delta',
              index: toolBlockIndex + toolIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: initialArgs,
              },
            } as Anthropic.Messages.RawMessageStreamEvent
            logForDebugging(`[API:opencode:stream] Yielding content_block_delta (tool initial), type: ${toolDelta.type}, args: ${initialArgs}`)
            yield toolDelta
          }
        } else {
          // 更新缓冲区
          const buffer = toolCallBuffers.get(bufferKey)!
          if (toolCall.function?.name) {
            buffer.name = toolCall.function.name
          }
          if (toolCall.function?.arguments) {
            buffer.args += toolCall.function.arguments

            // 发送参数增量
            const toolDelta = {
              type: 'content_block_delta',
              index: toolBlockIndex + toolIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: toolCall.function.arguments,
              },
            } as Anthropic.Messages.RawMessageStreamEvent
            logForDebugging(`[API:opencode:stream] Yielding content_block_delta (tool), type: ${toolDelta.type}, args: ${toolCall.function.arguments}`)
            yield toolDelta
          }
        }
      }
    }

    // 记录 token 使用情况
    if (chunk.usage) {
      outputTokens = chunk.usage.completion_tokens || 0
    }

    // 检查是否结束
    if (choice?.finish_reason) {
      // 关闭文本块
      if (hasTextBlock) {
        const blockStop = { type: 'content_block_stop', index: textBlockIndex } as Anthropic.Messages.RawMessageStreamEvent
        logForDebugging(`[API:opencode:stream] Yielding content_block_stop (text), type: ${blockStop.type}`)
        yield blockStop
      }

      // 关闭所有工具块
      for (let i = 0; i < toolCallBuffers.size; i++) {
        const toolBlockStop = { type: 'content_block_stop', index: toolBlockIndex + i } as Anthropic.Messages.RawMessageStreamEvent
        logForDebugging(`[API:opencode:stream] Yielding content_block_stop (tool), type: ${toolBlockStop.type}`)
        yield toolBlockStop
      }

      const msgDelta = {
        type: 'message_delta',
        delta: {
          stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : 'max_tokens',
          stop_sequence: null,
        },
        usage: { output_tokens: outputTokens },
      } as Anthropic.Messages.RawMessageStreamEvent
      logForDebugging(`[API:opencode:stream] Yielding message_delta, type: ${msgDelta.type}`)
      yield msgDelta

      const msgStop = { type: 'message_stop' } as Anthropic.Messages.RawMessageStreamEvent
      logForDebugging(`[API:opencode:stream] Yielding message_stop, type: ${msgStop.type}`)
      yield msgStop
      return  // 确保流结束
    }
  }
}

function createChainablePromise<T>(
  promiseFn: () => Promise<T>,
): Promise<T> & { withResponse: () => Promise<{ data: T; response: Response; request_id: string }> } {
  const promise = promiseFn() as Promise<T> & { withResponse: () => Promise<{ data: T; response: Response; request_id: string }> }
  promise.withResponse = async () => {
    const data = await promise
    const mockResponse = new Response(null, { status: 200 })
    return { data, response: mockResponse, request_id: crypto.randomUUID() }
  }
  return promise
}

export class OpencodeAdapter {
  private client: OpenAI
  public messages: ReturnType<typeof this.createMessagesProxy>
  public beta: { messages: ReturnType<typeof this.createMessagesProxy> }

  constructor() {
    this.client = createOpencodeClient()
    this.messages = this.createMessagesProxy()
    this.beta = { messages: this.createMessagesProxy() }
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
    _options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): Promise<AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>> & { withResponse: () => Promise<{ data: AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>; response: Response; request_id: string }> } {
    const originalModel = params.model
    // 使用免费模型
    const model = FREE_MODELS.includes(originalModel) ? originalModel : DEFAULT_MODEL
    const isStream = params.stream === true
    
    // 转换消息格式
    const messages = convertAnthropicMessagesToOpenAI(params.messages)

    // Convert tools from Anthropic format to OpenAI format if provided
    const tools = (params as any).tools?.map((tool: any) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema || tool.inputJSONSchema || {},
      },
    }))

    logForDebugging(`[API:opencode] Original model: ${originalModel}, Using free model: ${model}, stream: ${isStream}, messages: ${messages.length}, tools: ${tools?.length || 0}`)
    logForDebugging(`[API:opencode] Converted messages: ${JSON.stringify(messages.map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content.substring(0, 50) : '...', hasToolCalls: 'tool_calls' in m, toolCallId: 'tool_call_id' in m ? (m as any).tool_call_id : undefined })))}`)

    const promiseFn = async (): Promise<AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>> => {
      const requestParams: OpenAI.Chat.ChatCompletionCreateParams = {
        model,
        messages,
        max_tokens: params.max_tokens || 4096,
        temperature: params.temperature ?? 0.7,
        stream: isStream,
      }
      
      // Add tools if provided
      if (tools && tools.length > 0) {
        requestParams.tools = tools
      }
      
      const response = await this.client.chat.completions.create(requestParams)
      
      if (isStream) {
        return convertOpenAIStreamToAnthropic(response as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>, model)
      }
      
      // 非流式响应处理
      const completion = response as OpenAI.Chat.ChatCompletion
      const content = completion.choices[0]?.message?.content || ''
      return {
        [Symbol.asyncIterator]: async function* () {
          yield { 
            type: 'message_start', 
            message: { 
              id: completion.id, 
              type: 'message', 
              role: 'assistant', 
              content: [{ type: 'text', text: content }], 
              model: completion.model, 
              stop_reason: completion.choices[0]?.finish_reason === 'stop' ? 'end_turn' : null, 
              stop_sequence: null, 
              usage: { 
                input_tokens: completion.usage?.prompt_tokens || 0, 
                output_tokens: completion.usage?.completion_tokens || 0 
              } 
            } 
          } as Anthropic.Messages.RawMessageStreamEvent
          yield { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } as Anthropic.Messages.RawMessageStreamEvent
          yield { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: content } } as Anthropic.Messages.RawMessageStreamEvent
          yield { type: 'content_block_stop', index: 0 } as Anthropic.Messages.RawMessageStreamEvent
          yield { 
            type: 'message_delta', 
            delta: { 
              stop_reason: completion.choices[0]?.finish_reason === 'stop' ? 'end_turn' : null, 
              stop_sequence: null 
            }, 
            usage: { output_tokens: completion.usage?.completion_tokens || 0 } 
          } as Anthropic.Messages.RawMessageStreamEvent
          yield { type: 'message_stop' } as Anthropic.Messages.RawMessageStreamEvent
        },
      } as AsyncIterable<Anthropic.Messages.RawMessageStreamEvent>
    }

    return createChainablePromise(promiseFn)
  }
}

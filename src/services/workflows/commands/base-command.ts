import type { WorkflowContext, StepCallbacks } from '../../../stores/workflow-store'
import { useLLMStore } from '../../../stores/llm-store'
import { globalEventBus, EventPayloadMap } from '../../../shared/event-bus'
import type { BasePromptBuilder } from '../../prompts/prompt-builder'

export interface CommandExecuteParams {
  step: unknown
  context: WorkflowContext
  callbacks: StepCallbacks
}

/**
 * 工作流执行环节的抽象基类 (Command Pattern)
 * 将原本混乱的 workflow 闭包拆分为可独立测试、状态解耦的命令单元。
 */
export abstract class BaseWorkflowCommand<TResult = string> {
  
  /** 抽象执行入口 */
  abstract execute(params: CommandExecuteParams): Promise<TResult>

  /** 获取 LLM 大模型连接代理（支持取消） */
  protected async callLLM(
    prompt: string, 
    systemPrompt: string, 
    callbacks: StepCallbacks,
    options?: { responseFormat?: { type: string }; thinking?: boolean; maxTokens?: number },
    context?: WorkflowContext
  ): Promise<string> {
    const llmStore = useLLMStore.getState()
    if (!llmStore.defaultModelId) throw new Error('未配置默认 AI 模型')

    callbacks.setProgress(10)

    return new Promise((resolve, reject) => {
      let fullContent = ''
      let streamRequestId = ''

      // 取消监听：轮询 context.cancelled，主动中断 LLM 流
      let cancelCheckTimer: ReturnType<typeof setInterval> | null = null
      if (context) {
        cancelCheckTimer = setInterval(() => {
          if (context.cancelled && streamRequestId) {
            clearInterval(cancelCheckTimer!)
            cancelCheckTimer = null
            llmStore.cancelGeneration(streamRequestId).catch(() => {})
            reject(new Error('工作流已取消'))
          }
        }, 200)
      }

      const cleanup = () => {
        if (cancelCheckTimer) {
          clearInterval(cancelCheckTimer)
          cancelCheckTimer = null
        }
      }

      llmStore.generateStream(
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt }
        ],
        {
          onChunk: (chunk) => {
            // 取消后不再追加输出
            if (context?.cancelled) return
            fullContent += chunk
            callbacks.appendText(chunk)
          },
          onDone: (text) => {
            cleanup()
            // 取消后不 resolve，让 reject 生效
            if (context?.cancelled) {
              reject(new Error('工作流已取消'))
              return
            }
            callbacks.setProgress(90)
            const raw = text || fullContent
            const cleaned = this.stripThinkingTags(raw)
            resolve(cleaned)
          },
          onError: (err) => {
            cleanup()
            reject(new Error(err || '流式生成失败'))
          }
        },
        undefined,
        options
      ).then(reqId => {
        streamRequestId = reqId
        // 如果在 generateStream 返回前已经取消
        if (context?.cancelled) {
          llmStore.cancelGeneration(reqId).catch(() => {})
          cleanup()
          reject(new Error('工作流已取消'))
        }
      }).catch(err => {
        cleanup()
        reject(err)
      })
    })
  }

  /**
   * 使用 Builder 的 systemRole + prompt 一键调用 LLM
   * 角色定位由模板自带，command 不再需要硬编码 system message
   */
  protected async callLLMWithBuilder(
    builder: BasePromptBuilder,
    callbacks: StepCallbacks,
    options?: { responseFormat?: { type: string }; thinking?: boolean; maxTokens?: number },
    context?: WorkflowContext
  ): Promise<string> {
    return this.callLLM(builder.build(), builder.getSystemRole(), callbacks, options, context)
  }

  /**
   * 去除 DeepSeek 等模型的 <think> 标签，保证落盘纯净
   */
  protected stripThinkingTags(text: string): string {
    return text.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim()
  }

  /**
   * 全局容错 JSON 解析器
   * 自动剥离 Markdown ```json 代码块并处理尾随逗号等常见大模型幻觉
   */
  protected parseJSON<T>(text: string): T {
    try {
      // 1. 剥离 Markdown 块
      let cleanText = text.replace(/```json?\n?/gi, '').replace(/```\n?/gi, '').trim()
      // 2. 如果存在前序引导语，截取第一把括号到最后一把括号
      const firstBrace = cleanText.indexOf('{')
      const firstBracket = cleanText.indexOf('[')
      const lastBrace = cleanText.lastIndexOf('}')
      const lastBracket = cleanText.lastIndexOf(']')

      if (firstBrace !== -1 && lastBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)) {
        cleanText = cleanText.substring(firstBrace, lastBrace + 1)
      } else if (firstBracket !== -1 && lastBracket !== -1) {
        cleanText = cleanText.substring(firstBracket, lastBracket + 1)
      }

      try {
        return JSON.parse(cleanText) as T
      } catch (parseErr) {
        // 3. 截断恢复：尝试追加缺失的闭合括号
        const repaired = this.repairTruncatedJSON(cleanText)
        if (repaired) {
          return JSON.parse(repaired) as T
        }
        throw parseErr
      }
    } catch {
      // 4. 友好错误提示：给出具体诊断信息
      const trimmed = text.replace(/\s+/g, ' ').trim()
      const tail = trimmed.slice(-120)
      const isTruncated = this.looksTruncatedJSON(tail)
      const hint = isTruncated
        ? '\n→ 响应疑似被 max_tokens 截断，建议：增加模型配置的 max_tokens（至少 16000）或更换更大上下文模型'
        : ''
      throw new Error(
        `AI 返回的数据格式乱码，无法解析为有效层级结构。${hint}\n响应内容末端: ${tail}`
      )
    }
  }

  /**
   * 尝试修复被截断的 JSON
   * 常见模式：模型在输出长 JSON 时被 max_tokens 截断，最后一个对象字段不完整
   */
  private repairTruncatedJSON(text: string): string | null {
    // 策略：向后扫描找到最后的完整体 JSON 值，追加缺失的闭合符号
    // 检查是否在字符串值中间被截断
    const lastQuote = text.lastIndexOf('"')
    const lastColon = text.lastIndexOf(':')

    // 如果末尾在字符串内容中（有未闭合的引号），移除该不完整字段
    if (lastQuote > lastColon && text.lastIndexOf('":', lastQuote) === -1) {
      // 向前找到该字段的 key 开始位置
      const beforeKey = text.lastIndexOf('",', lastQuote)
      if (beforeKey !== -1) {
        // 截断到上一个完整字段末尾，然后闭合剩余结构
        let fixed = text.substring(0, beforeKey + 1)
        // 补全缺失的闭合符号：尝试统计并补充 } ] }
        const openBraces = (fixed.match(/\{/g) || []).length
        const closeBraces = (fixed.match(/\}/g) || []).length
        const openBrackets = (fixed.match(/\[/g) || []).length
        const closeBrackets = (fixed.match(/\]/g) || []).length

        // 先闭合数组再闭合对象（符合 JSON 嵌套层级）
        if (closeBrackets < openBrackets) {
          fixed += ']'.repeat(openBrackets - closeBrackets)
        }
        if (closeBraces < openBraces) {
          fixed += '}'.repeat(openBraces - closeBraces)
        }

        // 验证修复结果
        try {
          JSON.parse(fixed)
          return fixed
        } catch {
          // 修复失败，返回 null
        }
      }
    }
    return null
  }

  /**
   * 判断 JSON 结尾是否看起来像被截断（而非格式错误）
   */
  private looksTruncatedJSON(tail: string): boolean {
    // 如果末尾不在合法的 JSON 终止符上（}, ], "），且不在引号内的值结尾，则可能是截断
    const trimmed = tail.trimEnd()
    if (trimmed.endsWith('}') || trimmed.endsWith(']') || trimmed.endsWith('"')) return false
    // 末尾有冒号或逗号，明显被截断
    if (trimmed.endsWith(':') || trimmed.endsWith(',')) return true
    // 末尾是大段文字（可能在字段值中间），被截断
    if (trimmed.length > 10 && !trimmed.includes('"') && !trimmed.includes('{')) return true
    return false
  }

  /**
   * 解耦的事件驱动：通知 UI 层去更新资产树，而无需去 import Zustand Store
   */
  protected notifyRefresh(resources: EventPayloadMap['REFRESH_RESOURCE']['resources']) {
    globalEventBus.emit('REFRESH_RESOURCE', { resources })
  }
}


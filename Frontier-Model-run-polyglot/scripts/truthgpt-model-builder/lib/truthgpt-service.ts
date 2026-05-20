/**
 * TruthGPT Service - Servicio Mejorado
 * =====================================
 * 
 * Servicio mejorado con retry, cache, streaming y más
 */

export interface TruthGPTConfig {
  apiUrl?: string;
  apiKey?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeout?: number;
  retries?: number;
  enableCache?: boolean;
  enableStreaming?: boolean;
}

export interface TruthGPTMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface TruthGPTResponse {
  content: string;
  model: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  finishReason?: string;
}

export interface TruthGPTStreamChunk {
  content: string;
  done: boolean;
}

export class TruthGPTService {
  private config: Required<TruthGPTConfig>;
  private cache: Map<string, TruthGPTResponse> = new Map();

  constructor(config: TruthGPTConfig = {}) {
    this.config = {
      apiUrl: config.apiUrl || '/api/truthgpt',
      apiKey: config.apiKey || '',
      model: config.model || 'truthgpt',
      temperature: config.temperature ?? 0.7,
      maxTokens: config.maxTokens ?? 2000,
      timeout: config.timeout ?? 30000,
      retries: config.retries ?? 3,
      enableCache: config.enableCache ?? false,
      enableStreaming: config.enableStreaming ?? false,
    };
  }

  /**
   * Enviar mensaje con retry automático
   */
  async sendMessage(
    message: string,
    history: TruthGPTMessage[] = [],
    options?: Partial<TruthGPTConfig>
  ): Promise<string> {
    const finalConfig = { ...this.config, ...options };
    
    // Verificar cache
    if (finalConfig.enableCache) {
      const cacheKey = this.getCacheKey(message, history);
      const cached = this.cache.get(cacheKey);
      if (cached) {
        return cached.content;
      }
    }

    // Si streaming está habilitado, usar streaming
    if (finalConfig.enableStreaming) {
      let fullContent = '';
      for await (const chunk of this.streamMessage(message, history, options)) {
        fullContent += chunk.content;
      }
      return fullContent;
    }

    // Intentar con retry
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= finalConfig.retries; attempt++) {
      try {
        const response = await this.makeRequest(message, history, finalConfig);
        
        // Guardar en cache
        if (finalConfig.enableCache) {
          const cacheKey = this.getCacheKey(message, history);
          this.cache.set(cacheKey, response);
        }
        
        return response.content;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        // Si no es el último intento, esperar antes de reintentar
        if (attempt < finalConfig.retries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Error al enviar mensaje');
  }

  /**
   * Stream de mensajes
   */
  async *streamMessage(
    message: string,
    history: TruthGPTMessage[] = [],
    options?: Partial<TruthGPTConfig>
  ): AsyncGenerator<TruthGPTStreamChunk> {
    const finalConfig = { ...this.config, ...options };
    
    try {
      const response = await fetch(finalConfig.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(finalConfig.apiKey && { Authorization: `Bearer ${finalConfig.apiKey}` }),
        },
        body: JSON.stringify({
          message,
          history,
          model: finalConfig.model,
          temperature: finalConfig.temperature,
          max_tokens: finalConfig.maxTokens,
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      let buffer = '';
      let done = false;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;

        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                yield { content: '', done: true };
                return;
              }

              try {
                const parsed = JSON.parse(data);
                if (parsed.content) {
                  yield { content: parsed.content, done: false };
                }
              } catch {
                // Ignorar líneas inválidas
              }
            }
          }
        }
      }

      yield { content: '', done: true };
    } catch (error) {
      throw error;
    }
  }

  /**
   * Hacer request HTTP
   */
  private async makeRequest(
    message: string,
    history: TruthGPTMessage[],
    config: Required<TruthGPTConfig>
  ): Promise<TruthGPTResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), config.timeout);

    try {
      const response = await fetch(config.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.apiKey && { Authorization: `Bearer ${config.apiKey}` }),
        },
        body: JSON.stringify({
          message,
          history,
          model: config.model,
          temperature: config.temperature,
          max_tokens: config.maxTokens,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error?.message || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();
      return {
        content: data.content || data.response || data.message || '',
        model: data.model || config.model,
        usage: data.usage,
        finishReason: data.finish_reason,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }

  /**
   * Generar clave de cache
   */
  private getCacheKey(message: string, history: TruthGPTMessage[]): string {
    const historyStr = history.map((m) => `${m.role}:${m.content}`).join('|');
    return `${message}|${historyStr}`;
  }

  /**
   * Limpiar cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Actualizar configuración
   */
  updateConfig(newConfig: Partial<TruthGPTConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Obtener configuración actual
   */
  getConfig(): Required<TruthGPTConfig> {
    return { ...this.config };
  }
}

// Instancia singleton
let serviceInstance: TruthGPTService | null = null;

export const getTruthGPTService = (config?: TruthGPTConfig): TruthGPTService => {
  if (!serviceInstance) {
    serviceInstance = new TruthGPTService(config);
  }
  return serviceInstance;
};

export default TruthGPTService;

/**
 * Create a TruthGPT model - Main entry point for model creation
 * This function orchestrates the entire model creation workflow
 */
export async function createTruthGPTModel(
  modelId: string,
  modelName: string,
  description: string
): Promise<void> {
  const {
    analyzeAndPrepareSpec,
    prepareModelDirectory,
    generateModelFiles,
    generateSupportingFiles,
    performTruthGPTIntegration,
  } = await import('./services/model-creation-service')
  const {
    setModelStatus,
    updateModelProgress,
    markModelCompleted,
    markModelFailed,
  } = await import('./services/model-status-service')

  try {
    // Set initial status
    setModelStatus(modelId, {
      status: 'creating',
      progress: 0,
      currentStep: 'Iniciando creación del modelo',
    })

    // Step 1: Analyze and prepare specification
    updateModelProgress(modelId, 10, 'Analizando descripción del modelo')
    const spec = await analyzeAndPrepareSpec(description)
    setModelStatus(modelId, { spec })

    // Step 2: Prepare model directory
    updateModelProgress(modelId, 20, 'Preparando directorio del modelo')
    const modelDir = await prepareModelDirectory(modelName)

    // Step 3: Generate model files
    updateModelProgress(modelId, 40, 'Generando archivos del modelo')
    await generateModelFiles(modelDir, modelName, description, spec)

    // Step 4: Generate supporting files
    updateModelProgress(modelId, 60, 'Generando archivos de soporte')
    await generateSupportingFiles(modelDir, modelName, description, spec)

    // Step 5: Integrate with TruthGPT
    updateModelProgress(modelId, 80, 'Integrando con TruthGPT')
    await performTruthGPTIntegration(modelDir, modelName)

    // Step 6: Mark as completed
    updateModelProgress(modelId, 100, 'Modelo completado')
    markModelCompleted(modelId)
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : 'Error desconocido'
    markModelFailed(modelId, errorMessage)
    throw error
  }
}

/**
 * Get model status
 */
export async function getModelStatus(modelId: string) {
  const { getModelStatus: getStatus } = await import(
    './services/model-status-service'
  )
  return getStatus(modelId)
}

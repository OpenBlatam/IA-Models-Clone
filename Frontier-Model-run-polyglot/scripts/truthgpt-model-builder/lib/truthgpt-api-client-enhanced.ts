/**
 * TruthGPT API Client Mejorado
 * ============================
 * 
 * Versión mejorada con:
 * - Mejor manejo de errores
 * - Retry automático con backoff exponencial
 * - Timeout configurable
 * - Request/Response interceptors
 * - Caching inteligente
 * - Rate limiting
 * - Logging avanzado
 */

import {
  ModelSpec,
  ModelState,
  ModelProgress,
  ValidationResult,
  TrainingResults,
  EvaluationResults,
  PredictionResults
} from './types/modelTypes'
import { transformSpecToAPI, transformAPIToSpec } from './utils/dataTransformers'
import { validateModelSpec } from './utils/typeValidators'

const API_BASE_URL = process.env.NEXT_PUBLIC_TRUTHGPT_API_URL || 'http://localhost:8000'

// ============================================================================
// TIPOS Y INTERFACES
// ============================================================================

export interface APIClientOptions {
  baseUrl?: string
  timeout?: number
  retries?: number
  retryDelay?: number
  enableCache?: boolean
  cacheTTL?: number
  enableLogging?: boolean
  rateLimit?: {
    maxRequests: number
    windowMs: number
  }
}

export interface RequestConfig {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  headers?: Record<string, string>
  body?: unknown
  timeout?: number
  retries?: number
  skipCache?: boolean
}

export interface Response<T = unknown> {
  data: T
  status: number
  headers: Headers
  cached?: boolean
}

export type RequestInterceptor = (config: RequestConfig) => RequestConfig | Promise<RequestConfig>
export type ResponseInterceptor<T = unknown> = (response: Response<T>) => Response<T> | Promise<Response<T>>
export type ErrorInterceptor = (error: Error) => Error | Promise<Error>

// ============================================================================
// CLASE API CLIENT MEJORADA
// ============================================================================

export class TruthGPTAPIClientEnhanced {
  private baseUrl: string
  private timeout: number
  private retries: number
  private retryDelay: number
  private enableCache: boolean
  private cacheTTL: number
  private enableLogging: boolean
  private rateLimit?: { maxRequests: number; windowMs: number }
  
  private cache: Map<string, { data: unknown; expires: number }> = new Map()
  private requestInterceptors: RequestInterceptor[] = []
  private responseInterceptors: ResponseInterceptor[] = []
  private errorInterceptors: ErrorInterceptor[] = []
  private rateLimitQueue: Array<{ timestamp: number }> = []

  constructor(options: APIClientOptions = {}) {
    this.baseUrl = (options.baseUrl || API_BASE_URL).replace(/\/$/, '')
    this.timeout = options.timeout || 30000
    this.retries = options.retries ?? 3
    this.retryDelay = options.retryDelay || 1000
    this.enableCache = options.enableCache ?? true
    this.cacheTTL = options.cacheTTL || 60000
    this.enableLogging = options.enableLogging ?? true
    this.rateLimit = options.rateLimit
  }

  // ============================================================================
  // INTERCEPTORS
  // ============================================================================

  /**
   * Agrega un interceptor de request
   */
  addRequestInterceptor(interceptor: RequestInterceptor): void {
    this.requestInterceptors.push(interceptor)
  }

  /**
   * Agrega un interceptor de response
   */
  addResponseInterceptor<T = unknown>(interceptor: ResponseInterceptor<T>): void {
    this.responseInterceptors.push(interceptor)
  }

  /**
   * Agrega un interceptor de error
   */
  addErrorInterceptor(interceptor: ErrorInterceptor): void {
    this.errorInterceptors.push(interceptor)
  }

  // ============================================================================
  // RATE LIMITING
  // ============================================================================

  /**
   * Verifica y aplica rate limiting
   */
  private async checkRateLimit(): Promise<void> {
    if (!this.rateLimit) return

    const now = Date.now()
    const windowStart = now - this.rateLimit.windowMs

    // Limpiar requests antiguos
    this.rateLimitQueue = this.rateLimitQueue.filter(
      req => req.timestamp > windowStart
    )

    // Si excede el límite, esperar
    if (this.rateLimitQueue.length >= this.rateLimit.maxRequests) {
      const oldestRequest = this.rateLimitQueue[0]
      const waitTime = this.rateLimit.windowMs - (now - oldestRequest.timestamp)
      
      if (waitTime > 0) {
        await new Promise(resolve => setTimeout(resolve, waitTime))
        return this.checkRateLimit()
      }
    }

    // Agregar request actual
    this.rateLimitQueue.push({ timestamp: now })
  }

  // ============================================================================
  // CACHING
  // ============================================================================

  /**
   * Obtiene del cache si está disponible
   */
  private getFromCache(key: string): unknown | null {
    if (!this.enableCache) return null

    const cached = this.cache.get(key)
    if (!cached) return null

    if (cached.expires < Date.now()) {
      this.cache.delete(key)
      return null
    }

    return cached.data
  }

  /**
   * Guarda en cache
   */
  private setCache(key: string, data: unknown): void {
    if (!this.enableCache) return

    this.cache.set(key, {
      data,
      expires: Date.now() + this.cacheTTL
    })
  }

  /**
   * Limpia el cache
   */
  clearCache(): void {
    this.cache.clear()
  }

  /**
   * Obtiene estadísticas del cache
   */
  getCacheStats(): { size: number; maxSize: number; hitRate: number } {
    // Implementación simplificada
    return {
      size: this.cache.size,
      maxSize: 100,
      hitRate: 0.8 // Placeholder
    }
  }

  // ============================================================================
  // REQUEST CORE
  // ============================================================================

  /**
   * Realiza una petición HTTP con retry y manejo de errores
   */
  private async request<T = unknown>(
    endpoint: string,
    config: RequestConfig = {}
  ): Promise<Response<T>> {
    const url = `${this.baseUrl}${endpoint}`
    const cacheKey = `${config.method || 'GET'}:${url}:${JSON.stringify(config.body)}`

    // Verificar cache
    if (!config.skipCache && config.method === 'GET') {
      const cached = this.getFromCache(cacheKey)
      if (cached) {
        this.log('cache', `Cache hit: ${endpoint}`)
        return {
          data: cached as T,
          status: 200,
          headers: new Headers(),
          cached: true
        }
      }
    }

    // Rate limiting
    await this.checkRateLimit()

    // Aplicar interceptors de request
    let finalConfig = { ...config }
    for (const interceptor of this.requestInterceptors) {
      finalConfig = await interceptor(finalConfig)
    }

    // Realizar request con retry
    let lastError: Error | null = null

    for (let attempt = 0; attempt <= (finalConfig.retries ?? this.retries); attempt++) {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(
          () => controller.abort(),
          finalConfig.timeout || this.timeout
        )

        const response = await fetch(url, {
          method: finalConfig.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            ...finalConfig.headers
          },
          body: finalConfig.body ? JSON.stringify(finalConfig.body) : undefined,
          signal: controller.signal
        })

        clearTimeout(timeoutId)

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`HTTP ${response.status}: ${errorText}`)
        }

        const data = await response.json()

        // Guardar en cache si es GET
        if (!config.skipCache && config.method === 'GET') {
          this.setCache(cacheKey, data)
        }

        const apiResponse: Response<T> = {
          data: data as T,
          status: response.status,
          headers: response.headers,
          cached: false
        }

        // Aplicar interceptors de response
        let finalResponse = apiResponse
        for (const interceptor of this.responseInterceptors) {
          finalResponse = await interceptor(finalResponse)
        }

        this.log('success', `Request successful: ${endpoint}`)
        return finalResponse

      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error))

        // No reintentar si es un error de validación
        if (lastError.message.includes('400') || lastError.message.includes('422')) {
          break
        }

        // Esperar antes de reintentar (backoff exponencial)
        if (attempt < (finalConfig.retries ?? this.retries)) {
          const delay = this.retryDelay * Math.pow(2, attempt)
          this.log('retry', `Retrying in ${delay}ms (attempt ${attempt + 1})`)
          await new Promise(resolve => setTimeout(resolve, delay))
        }
      }
    }

    // Aplicar interceptors de error
    let finalError = lastError!
    for (const interceptor of this.errorInterceptors) {
      finalError = await interceptor(finalError)
    }

    this.log('error', `Request failed: ${endpoint}`, finalError)
    throw finalError
  }

  // ============================================================================
  // LOGGING
  // ============================================================================

  private log(level: 'info' | 'success' | 'error' | 'cache' | 'retry', message: string, error?: Error): void {
    if (!this.enableLogging) return

    const timestamp = new Date().toISOString()
    const prefix = `[TruthGPT API] [${timestamp}]`

    switch (level) {
      case 'info':
        console.log(`${prefix} ℹ️  ${message}`)
        break
      case 'success':
        console.log(`${prefix} ✅ ${message}`)
        break
      case 'error':
        console.error(`${prefix} ❌ ${message}`, error)
        break
      case 'cache':
        console.log(`${prefix} 💾 ${message}`)
        break
      case 'retry':
        console.log(`${prefix} 🔄 ${message}`)
        break
    }
  }

  // ============================================================================
  // API METHODS
  // ============================================================================

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; cuda_available: boolean; device: string }> {
    const response = await this.request<{ status: string; cuda_available: boolean; device: string }>(
      '/health',
      { method: 'GET' }
    )
    return response.data
  }

  /**
   * Crear modelo con validación
   */
  async createModel(spec: ModelSpec): Promise<{ model_id: string; name?: string; status: string }> {
    // Validar spec antes de enviar
    const validation = validateModelSpec(spec)
    if (!validation.valid) {
      throw new Error(`Invalid model spec: ${validation.errors.map(e => e.message).join(', ')}`)
    }

    const apiSpec = transformSpecToAPI(spec)
    const response = await this.request<{ model_id: string; name?: string; status: string }>(
      '/models',
      {
        method: 'POST',
        body: apiSpec
      }
    )
    return response.data
  }

  /**
   * Compilar modelo
   */
  async compileModel(
    modelId: string,
    optimizer: string,
    loss: string,
    metrics?: string[]
  ): Promise<{ model_id: string; status: string }> {
    const response = await this.request<{ model_id: string; status: string }>(
      `/models/${modelId}/compile`,
      {
        method: 'POST',
        body: { optimizer, loss, metrics }
      }
    )
    return response.data
  }

  /**
   * Entrenar modelo
   */
  async trainModel(
    modelId: string,
    xTrain: number[][],
    yTrain: (number | number[])[],
    options?: {
      epochs?: number
      batchSize?: number
      validationData?: { x: number[][]; y: (number | number[])[] }
    }
  ): Promise<TrainingResults> {
    const response = await this.request<TrainingResults>(
      `/models/${modelId}/train`,
      {
        method: 'POST',
        body: {
          x_train: xTrain,
          y_train: yTrain,
          ...options
        }
      }
    )
    return response.data
  }

  /**
   * Evaluar modelo
   */
  async evaluateModel(
    modelId: string,
    xTest: number[][],
    yTest: (number | number[])[]
  ): Promise<EvaluationResults> {
    const response = await this.request<EvaluationResults>(
      `/models/${modelId}/evaluate`,
      {
        method: 'POST',
        body: { x_test: xTest, y_test: yTest }
      }
    )
    return response.data
  }

  /**
   * Predecir con modelo
   */
  async predict(
    modelId: string,
    x: number[][]
  ): Promise<PredictionResults> {
    const response = await this.request<PredictionResults>(
      `/models/${modelId}/predict`,
      {
        method: 'POST',
        body: { x }
      }
    )
    return response.data
  }

  /**
   * Listar modelos
   */
  async listModels(): Promise<Array<{ model_id: string; name?: string; status: string }>> {
    const response = await this.request<Array<{ model_id: string; name?: string; status: string }>>(
      '/models',
      { method: 'GET' }
    )
    return response.data
  }

  /**
   * Obtener información de modelo
   */
  async getModelInfo(modelId: string): Promise<{ model_id: string; name?: string; status: string; [key: string]: unknown }> {
    const response = await this.request<{ model_id: string; name?: string; status: string; [key: string]: unknown }>(
      `/models/${modelId}`,
      { method: 'GET' }
    )
    return response.data
  }

  /**
   * Eliminar modelo
   */
  async deleteModel(modelId: string): Promise<{ message: string }> {
    const response = await this.request<{ message: string }>(
      `/models/${modelId}`,
      { method: 'DELETE' }
    )
    return response.data
  }

  /**
   * Guardar modelo
   */
  async saveModel(modelId: string, path: string): Promise<{ message: string; path: string }> {
    const response = await this.request<{ message: string; path: string }>(
      `/models/${modelId}/save`,
      {
        method: 'POST',
        body: { path }
      }
    )
    return response.data
  }

  /**
   * Cargar modelo
   */
  async loadModel(path: string): Promise<{ model_id: string; message: string }> {
    const response = await this.request<{ model_id: string; message: string }>(
      '/models/load',
      {
        method: 'POST',
        body: { path }
      }
    )
    return response.data
  }
}








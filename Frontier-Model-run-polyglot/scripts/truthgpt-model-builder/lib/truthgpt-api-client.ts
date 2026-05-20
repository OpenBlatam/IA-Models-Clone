/**
 * TruthGPT API Client
 * ===================
 * 
 * Cliente para conectar con la API REST de TruthGPT
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_TRUTHGPT_API_URL || 'http://localhost:8000'

export interface LayerConfig {
  type: string
  params: Record<string, any>
}

export interface CreateModelRequest {
  layers: LayerConfig[]
  name?: string
}

export interface CompileModelRequest {
  optimizer: string
  optimizer_params?: Record<string, any>
  loss: string
  loss_params?: Record<string, any>
  metrics?: string[]
}

export interface TrainModelRequest {
  x_train: number[][]
  y_train: (number | number[])[]
  epochs?: number
  batch_size?: number
  validation_data?: {
    x: number[][]
    y: (number | number[])[]
  }
  verbose?: number
}

export interface EvaluateModelRequest {
  x_test: number[][]
  y_test: (number | number[])[]
  verbose?: number
}

export interface PredictRequest {
  x: number[][]
  verbose?: number
}

export interface ModelResponse {
  model_id: string
  name?: string
  status: string
  message?: string
}

export interface TrainResponse extends ModelResponse {
  history: {
    loss?: number[]
    accuracy?: number[]
    val_loss?: number[]
    val_accuracy?: number[]
    [key: string]: any
  }
}

export interface EvaluateResponse extends ModelResponse {
  results: {
    loss: number
    metrics?: number[]
  }
}

export interface PredictResponse extends ModelResponse {
  predictions: number[][]
}

export class TruthGPTAPIClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '') // Remove trailing slash
  }

  /**
   * Check if API server is healthy
   */
  async healthCheck(): Promise<{ status: string; cuda_available: boolean; device: string }> {
    const response = await fetch(`${this.baseUrl}/health`)
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`)
    }
    return response.json()
  }

  /**
   * Create a new model
   */
  async createModel(request: CreateModelRequest): Promise<ModelResponse> {
    const response = await fetch(`${this.baseUrl}/models/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to create model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Compile a model
   */
  async compileModel(modelId: string, request: CompileModelRequest): Promise<ModelResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}/compile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to compile model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Train a model
   */
  async trainModel(modelId: string, request: TrainModelRequest): Promise<TrainResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to train model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Evaluate a model
   */
  async evaluateModel(modelId: string, request: EvaluateModelRequest): Promise<EvaluateResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to evaluate model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Make predictions with a model
   */
  async predict(modelId: string, request: PredictRequest): Promise<PredictResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to predict: ${response.status}`)
    }

    return response.json()
  }

  /**
   * List all models
   */
  async listModels(): Promise<{ models: Array<{ model_id: string; name: string; compiled: boolean }>; count: number }> {
    const response = await fetch(`${this.baseUrl}/models`)
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.status}`)
    }
    return response.json()
  }

  /**
   * Get model information
   */
  async getModelInfo(modelId: string): Promise<{ model_id: string; name: string; compiled: boolean }> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}`)
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.status}`)
    }
    return response.json()
  }

  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<ModelResponse> {
    const response = await fetch(`${this.baseUrl}/models/${modelId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to delete model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Save a model to disk
   */
  async saveModel(modelId: string, filepath?: string): Promise<ModelResponse & { filepath: string }> {
    const url = new URL(`${this.baseUrl}/models/${modelId}/save`)
    if (filepath) {
      url.searchParams.set('filepath', filepath)
    }

    const response = await fetch(url.toString(), {
      method: 'POST',
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to save model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Load a model from disk
   */
  async loadModel(filepath: string, modelId?: string): Promise<ModelResponse & { filepath: string }> {
    const response = await fetch(`${this.baseUrl}/models/load`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filepath, model_id: modelId }),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `Failed to load model: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Get model status (for compatibility with legacy API)
   */
  async getModelStatus(modelId: string): Promise<{
    status: string
    progress?: number
    currentStep?: string
    githubUrl?: string
    spec?: any
    error?: string
  }> {
    try {
      const info = await this.getModelInfo(modelId)
      // Return a compatible status format
      return {
        status: info.compiled ? 'completed' : 'creating',
        progress: info.compiled ? 100 : 50,
        currentStep: info.compiled ? 'Model compiled' : 'Creating model',
      }
    } catch (error) {
      // If model doesn't exist, return creating status
      return {
        status: 'creating',
        progress: 0,
        currentStep: 'Initializing...',
      }
    }
  }

  /**
   * Helper: Convert model description to layers configuration
   * This is a simplified version - you may want to enhance this with AI/NLP
   */
  descriptionToLayers(description: string): LayerConfig[] {
    // Simple heuristic-based conversion
    // In a real implementation, you'd use AI to parse the description
    const lowerDesc = description.toLowerCase()
    
    // Default layers for classification
    const layers: LayerConfig[] = [
      { type: 'dense', params: { units: 128, activation: 'relu' } },
      { type: 'dropout', params: { rate: 0.2 } },
    ]

    // Add more layers based on keywords
    if (lowerDesc.includes('convolutional') || lowerDesc.includes('cnn') || lowerDesc.includes('image')) {
      layers.unshift(
        { type: 'conv2d', params: { filters: 32, kernel_size: 3, activation: 'relu' } },
        { type: 'maxpooling2d', params: { pool_size: 2 } },
        { type: 'conv2d', params: { filters: 64, kernel_size: 3, activation: 'relu' } },
        { type: 'maxpooling2d', params: { pool_size: 2 } },
        { type: 'flatten', params: {} }
      )
    }

    if (lowerDesc.includes('lstm') || lowerDesc.includes('sequence') || lowerDesc.includes('text')) {
      layers.unshift(
        { type: 'lstm', params: { units: 64, return_sequences: false } }
      )
    }

    // Output layer - try to detect number of classes
    let outputUnits = 10 // default
    const classMatch = description.match(/\b(\d+)\s*(?:class|category|categor|output)/i)
    if (classMatch) {
      outputUnits = parseInt(classMatch[1], 10)
    }

    layers.push({ type: 'dense', params: { units: outputUnits, activation: 'softmax' } })

    return layers
  }

  /**
   * Create and compile a model from description
   */
  async createModelFromDescription(
    description: string,
    modelName?: string
  ): Promise<{ modelId: string; name: string }> {
    // Convert description to layers
    const layers = this.descriptionToLayers(description)

    // Create model
    const createResponse = await this.createModel({
      layers,
      name: modelName,
    })

    // Compile with default settings
    await this.compileModel(createResponse.model_id, {
      optimizer: 'adam',
      optimizer_params: { learning_rate: 0.001 },
      loss: 'sparsecategoricalcrossentropy',
      metrics: ['accuracy'],
    })

    return {
      modelId: createResponse.model_id,
      name: createResponse.name || modelName || 'Model',
    }
  }
}

// Singleton instance
let clientInstance: TruthGPTAPIClient | null = null

export const getTruthGPTAPIClient = (baseUrl?: string): TruthGPTAPIClient => {
  if (!clientInstance) {
    clientInstance = new TruthGPTAPIClient(baseUrl)
  }
  return clientInstance
}

export default TruthGPTAPIClient


import axios, { AxiosInstance, AxiosError } from 'axios'
import type {
  PaperResponse,
  PaperLinkRequest,
  TrainingRequest,
  TrainingResponse,
  CodeImproveRequest,
  CodeImproveResponse,
  RepositoryAnalyzeRequest,
  RepositoryAnalyzeResponse,
  ModelStatusResponse,
  HealthCheckResponse,
  PapersListResponse,
  VectorStoreStats,
  CacheStats,
  MetricsStats,
} from './types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 300000, // 5 minutes for long operations
    })

    this.setupInterceptors()
  }

  private setupInterceptors(): void {
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          const message =
            (error.response.data as { detail?: string })?.detail ||
            error.message ||
            'An error occurred'
          throw new Error(message)
        }
        throw error
      }
    )
  }

  // Health Check
  async getHealth(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>(
      '/api/research-paper-code-improver/health'
    )
    return response.data
  }

  // Papers
  async uploadPaper(file: File): Promise<PaperResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await this.client.post<PaperResponse>(
      '/api/research-paper-code-improver/papers/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  }

  async processLink(request: PaperLinkRequest): Promise<PaperResponse> {
    const response = await this.client.post<PaperResponse>(
      '/api/research-paper-code-improver/papers/link',
      request
    )
    return response.data
  }

  async listPapers(limit = 50): Promise<PapersListResponse> {
    const response = await this.client.get<PapersListResponse>(
      '/api/research-paper-code-improver/papers',
      {
        params: { limit },
      }
    )
    return response.data
  }

  async getPaper(paperId: string): Promise<PaperResponse> {
    const response = await this.client.get<PaperResponse>(
      `/api/research-paper-code-improver/papers/${paperId}`
    )
    return response.data
  }

  // Training
  async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
    const response = await this.client.post<TrainingResponse>(
      '/api/research-paper-code-improver/training/train',
      request
    )
    return response.data
  }

  async getModelStatus(modelId: string): Promise<ModelStatusResponse> {
    const response = await this.client.get<ModelStatusResponse>(
      `/api/research-paper-code-improver/models/${modelId}/status`
    )
    return response.data
  }

  // Code Improvement
  async improveCode(request: CodeImproveRequest): Promise<CodeImproveResponse> {
    const response = await this.client.post<CodeImproveResponse>(
      '/api/research-paper-code-improver/code/improve',
      request
    )
    return response.data
  }

  async improveCodeText(
    code: string,
    context?: string,
    modelId?: string
  ): Promise<CodeImproveResponse> {
    const response = await this.client.post<CodeImproveResponse>(
      '/api/research-paper-code-improver/code/improve-text',
      null,
      {
        params: { code, context, model_id: modelId },
      }
    )
    return response.data
  }

  async analyzeRepository(
    request: RepositoryAnalyzeRequest
  ): Promise<RepositoryAnalyzeResponse> {
    const response = await this.client.post<RepositoryAnalyzeResponse>(
      '/api/research-paper-code-improver/repository/analyze',
      request
    )
    return response.data
  }

  // Stats
  async getVectorStoreStats(): Promise<VectorStoreStats> {
    const response = await this.client.get<VectorStoreStats>(
      '/api/research-paper-code-improver/vector-store/stats'
    )
    return response.data
  }

  async getCacheStats(): Promise<CacheStats> {
    const response = await this.client.get<CacheStats>(
      '/api/research-paper-code-improver/cache/stats'
    )
    return response.data
  }

  async getMetricsStats(hours = 24): Promise<MetricsStats> {
    const response = await this.client.get<MetricsStats>(
      '/api/research-paper-code-improver/metrics/stats',
      {
        params: { hours },
      }
    )
    return response.data
  }

  // Cache Management
  async clearCache(olderThanHours?: number): Promise<{ deleted_files: number; message: string }> {
    const response = await this.client.post<{ deleted_files: number; message: string }>(
      '/api/research-paper-code-improver/cache/clear',
      null,
      {
        params: olderThanHours ? { older_than_hours: olderThanHours } : {},
      }
    )
    return response.data
  }
}

export const apiClient = new ApiClient()





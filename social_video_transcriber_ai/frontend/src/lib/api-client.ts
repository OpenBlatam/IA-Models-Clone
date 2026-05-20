import axios, { AxiosInstance, AxiosError } from 'axios';
import {
  TranscriptionRequest,
  TranscriptionResponse,
  VariantRequest,
  VariantResponse,
  QuickVariantRequest,
  AnalysisRequest,
  AnalysisResponse,
  BatchJob,
  VideoInfo,
  JobListItem,
  CacheStats,
  Tier,
  UsageStats,
  HealthResponse,
  FullAnalysisResponse,
  KeywordResponse,
  SummaryResponse,
  SentimentResponse,
  TranscriptionTextResponse,
} from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_URL}/api/v1`,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add API key interceptor if available
    this.client.interceptors.request.use((config) => {
      const apiKey = this.getApiKey();
      if (apiKey) {
        config.headers['X-API-Key'] = apiKey;
      }
      return config;
    });

    // Error handling interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          console.error('Unauthorized: Invalid API key');
        } else if (error.response?.status === 429) {
          // Handle rate limit
          console.error('Rate limit exceeded');
        }
        return Promise.reject(error);
      }
    );
  }

  setApiKey(apiKey: string) {
    if (typeof window !== 'undefined') {
      localStorage.setItem('api_key', apiKey);
    }
  }

  getApiKey(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('api_key');
    }
    return null;
  }

  clearApiKey() {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('api_key');
    }
  }

  // Transcription Endpoints
  async transcribeVideo(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    const response = await this.client.post<TranscriptionResponse>('/transcribe', request);
    return response.data;
  }

  async getTranscriptionStatus(jobId: string): Promise<TranscriptionResponse> {
    const response = await this.client.get<TranscriptionResponse>(`/transcribe/${jobId}`);
    return response.data;
  }

  async getTranscriptionText(
    jobId: string,
    withTimestamps = true,
    format: 'text' | 'srt' | 'vtt' = 'text'
  ): Promise<TranscriptionTextResponse> {
    const response = await this.client.get<TranscriptionTextResponse>(
      `/transcribe/${jobId}/text`,
      {
        params: { with_timestamps: withTimestamps, format },
      }
    );
    return response.data;
  }

  // Analysis Endpoints
  async analyzeText(request: AnalysisRequest): Promise<AnalysisResponse> {
    const response = await this.client.post<AnalysisResponse>('/analyze', request);
    return response.data;
  }

  async fullAnalysis(text: string): Promise<FullAnalysisResponse> {
    const response = await this.client.post<FullAnalysisResponse>('/analyze/full', null, {
      params: { text },
    });
    return response.data;
  }

  async extractKeywords(text: string, maxKeywords = 15): Promise<KeywordResponse> {
    const response = await this.client.post<KeywordResponse>('/analyze/keywords', null, {
      params: { text, max_keywords: maxKeywords },
    });
    return response.data;
  }

  async generateSummary(text: string): Promise<SummaryResponse> {
    const response = await this.client.post<SummaryResponse>('/analyze/summary', null, {
      params: { text },
    });
    return response.data;
  }

  async analyzeSentiment(text: string): Promise<SentimentResponse> {
    const response = await this.client.post<SentimentResponse>('/analyze/sentiment', null, {
      params: { text },
    });
    return response.data;
  }

  async detectSpeakers(jobId: string): Promise<any> {
    const response = await this.client.post(`/analyze/speakers`, null, {
      params: { job_id: jobId },
    });
    return response.data;
  }

  // Variant Endpoints
  async generateVariants(request: VariantRequest): Promise<VariantResponse> {
    const response = await this.client.post<VariantResponse>('/variants', request);
    return response.data;
  }

  async generateQuickVariants(request: QuickVariantRequest): Promise<VariantResponse> {
    const response = await this.client.post<VariantResponse>('/variants/quick', request);
    return response.data;
  }

  async generateVariantsFromText(
    text: string,
    numVariants = 3,
    preserveStructure = true,
    preserveLength = true,
    targetTone?: string
  ): Promise<VariantResponse> {
    const response = await this.client.post<VariantResponse>('/variants/text', null, {
      params: {
        text,
        num_variants: numVariants,
        preserve_structure: preserveStructure,
        preserve_length: preserveLength,
        target_tone: targetTone,
      },
    });
    return response.data;
  }

  // Batch Processing
  async createBatchJob(
    urls: string[],
    includeTimestamps = true,
    includeAnalysis = true,
    language?: string,
    webhookUrl?: string
  ): Promise<BatchJob> {
    const response = await this.client.post<BatchJob>('/batch', null, {
      params: {
        urls,
        include_timestamps: includeTimestamps,
        include_analysis: includeAnalysis,
        language,
        webhook_url: webhookUrl,
      },
    });
    return response.data;
  }

  async getBatchStatus(batchId: string): Promise<BatchJob> {
    const response = await this.client.get<BatchJob>(`/batch/${batchId}`);
    return response.data;
  }

  async getBatchResults(batchId: string): Promise<BatchJob> {
    const response = await this.client.get<BatchJob>(`/batch/${batchId}/results`);
    return response.data;
  }

  // Utility Endpoints
  async getVideoInfo(url: string): Promise<VideoInfo> {
    const response = await this.client.get<VideoInfo>('/video/info', {
      params: { url },
    });
    return response.data;
  }

  async listSupportedPlatforms(): Promise<{ platforms: any[] }> {
    const response = await this.client.get('/platforms');
    return response.data;
  }

  async listJobs(status?: string, limit = 50): Promise<{ jobs: JobListItem[]; total: number }> {
    const response = await this.client.get<{ jobs: JobListItem[]; total: number }>('/jobs', {
      params: { status, limit },
    });
    return response.data;
  }

  async deleteJob(jobId: string): Promise<void> {
    await this.client.delete(`/jobs/${jobId}`);
  }

  // Cache Endpoints
  async getCacheStats(): Promise<CacheStats> {
    const response = await this.client.get<CacheStats>('/cache/stats');
    return response.data;
  }

  async clearExpiredCache(): Promise<{ cleared_entries: number }> {
    const response = await this.client.post<{ cleared_entries: number }>('/cache/clear');
    return response.data;
  }

  // Auth Endpoints
  async listTiers(): Promise<{ tiers: Tier[] }> {
    const response = await this.client.get<{ tiers: Tier[] }>('/auth/tiers');
    return response.data;
  }

  async getUsage(): Promise<UsageStats> {
    const response = await this.client.get<UsageStats>('/auth/usage');
    return response.data;
  }

  // Health Endpoints
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async detailedHealth(): Promise<any> {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }
}

export const apiClient = new ApiClient();





/**
 * PDF Variantes API Client
 * TypeScript client for PDF Variantes API
 * Version: 2.0.0
 */

import type {
  PDFUploadRequest,
  PDFUploadResponse,
  PDFDocument,
  PDFEditRequest,
  PDFEditResponse,
  VariantGenerateRequest,
  VariantGenerateResponse,
  PDFVariant,
  TopicExtractRequest,
  TopicExtractResponse,
  TopicItem,
  BrainstormGenerateRequest,
  BrainstormGenerateResponse,
  BrainstormIdea,
  SearchRequest,
  SearchResponse,
  BatchProcessingRequest,
  BatchProcessingResponse,
  ExportRequest,
  ExportResponse,
  VariantStopRequest,
  VariantStopResponse,
  SystemHealth,
  AnalyticsReport,
  APIResponse,
  PaginatedResponse,
  ListQueryParams,
} from "./types";

export interface APIClientConfig {
  baseURL: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class PDFVariantesAPIClient {
  private config: Required<Omit<APIClientConfig, "headers">> & {
    headers: Record<string, string>;
  };

  constructor(config: APIClientConfig) {
    this.config = {
      baseURL: config.baseURL.replace(/\/$/, ""),
      apiKey: config.apiKey || "",
      timeout: config.timeout || 30000,
      headers: {
        "Content-Type": "application/json",
        ...(config.apiKey && { Authorization: `Bearer ${config.apiKey}` }),
        ...config.headers,
      },
    };
  }

  /**
   * Internal fetch wrapper with error handling
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const url = `${this.config.baseURL}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.config.headers,
          ...options.headers,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        return {
          status: response.status,
          error: {
            message: data.detail || data.message || "An error occurred",
            code: response.status,
            details: data,
          },
        };
      }

      return {
        status: response.status,
        data: data as T,
      };
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === "AbortError") {
        return {
          status: 408,
          error: {
            message: "Request timeout",
            code: 408,
          },
        };
      }

      return {
        status: 0,
        error: {
          message: error instanceof Error ? error.message : "Network error",
          code: "NETWORK_ERROR",
        },
      };
    }
  }

  // ============================================================================
  // PDF Operations
  // ============================================================================

  /**
   * Upload a PDF file
   */
  async uploadPDF(
    file: File,
    options?: Partial<PDFUploadRequest>
  ): Promise<APIResponse<PDFUploadResponse>> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("auto_process", String(options?.auto_process ?? true));
    formData.append("extract_text", String(options?.extract_text ?? true));
    formData.append("detect_language", String(options?.detect_language ?? true));

    if (options?.filename) {
      formData.append("filename", options.filename);
    }

    // Remove Content-Type header for FormData
    const headers = { ...this.config.headers };
    delete headers["Content-Type"];

    const response = await fetch(`${this.config.baseURL}/api/v1/pdf/upload`, {
      method: "POST",
      headers,
      body: formData,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return {
        status: response.status,
        error: {
          message: data.detail || data.message || "Upload failed",
          code: response.status,
          details: data,
        },
      };
    }

    return {
      status: response.status,
      data: data as PDFUploadResponse,
    };
  }

  /**
   * List documents
   */
  async listDocuments(
    params?: ListQueryParams
  ): Promise<APIResponse<PDFDocument[]>> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append("limit", String(params.limit));
    if (params?.offset) queryParams.append("offset", String(params.offset));

    const endpoint = `/api/v1/pdf/documents${
      queryParams.toString() ? `?${queryParams.toString()}` : ""
    }`;

    return this.request<PDFDocument[]>(endpoint);
  }

  /**
   * Get a specific document
   */
  async getDocument(documentId: string): Promise<APIResponse<PDFDocument>> {
    return this.request<PDFDocument>(`/api/v1/pdf/documents/${documentId}`);
  }

  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<APIResponse<{ message: string }>> {
    return this.request(`/api/v1/pdf/documents/${documentId}`, {
      method: "DELETE",
    });
  }

  // ============================================================================
  // Variant Operations
  // ============================================================================

  /**
   * Generate variants
   */
  async generateVariants(
    request: VariantGenerateRequest
  ): Promise<APIResponse<VariantGenerateResponse>> {
    return this.request<VariantGenerateResponse>("/api/v1/variants/generate", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * List variants for a document
   */
  async listVariants(
    documentId: string,
    params?: ListQueryParams
  ): Promise<APIResponse<PDFVariant[]>> {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.append("limit", String(params.limit));
    if (params?.offset) queryParams.append("offset", String(params.offset));

    const endpoint = `/api/v1/variants/documents/${documentId}/variants${
      queryParams.toString() ? `?${queryParams.toString()}` : ""
    }`;

    return this.request<PDFVariant[]>(endpoint);
  }

  /**
   * Get a specific variant
   */
  async getVariant(variantId: string): Promise<APIResponse<PDFVariant>> {
    return this.request<PDFVariant>(`/api/v1/variants/variants/${variantId}`);
  }

  /**
   * Stop variant generation
   */
  async stopGeneration(
    request: VariantStopRequest
  ): Promise<APIResponse<VariantStopResponse>> {
    const formData = new FormData();
    formData.append("document_id", request.document_id);
    formData.append("keep_generated", String(request.keep_generated ?? true));

    const headers = { ...this.config.headers };
    delete headers["Content-Type"];

    const response = await fetch(`${this.config.baseURL}/api/v1/variants/stop`, {
      method: "POST",
      headers,
      body: formData,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      return {
        status: response.status,
        error: {
          message: data.detail || data.message || "Failed to stop generation",
          code: response.status,
          details: data,
        },
      };
    }

    return {
      status: response.status,
      data: data as VariantStopResponse,
    };
  }

  // ============================================================================
  // Topic Operations
  // ============================================================================

  /**
   * Extract topics from a document
   */
  async extractTopics(
    request: TopicExtractRequest
  ): Promise<APIResponse<TopicExtractResponse>> {
    return this.request<TopicExtractResponse>("/api/v1/topics/extract", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * List topics for a document
   */
  async listTopics(
    documentId: string,
    minRelevance?: number
  ): Promise<APIResponse<TopicItem[]>> {
    const queryParams = new URLSearchParams();
    if (minRelevance !== undefined) {
      queryParams.append("min_relevance", String(minRelevance));
    }

    const endpoint = `/api/v1/topics/documents/${documentId}/topics${
      queryParams.toString() ? `?${queryParams.toString()}` : ""
    }`;

    return this.request<TopicItem[]>(endpoint);
  }

  // ============================================================================
  // Brainstorm Operations
  // ============================================================================

  /**
   * Generate brainstorm ideas
   */
  async generateBrainstormIdeas(
    request: BrainstormGenerateRequest
  ): Promise<APIResponse<BrainstormGenerateResponse>> {
    return this.request<BrainstormGenerateResponse>(
      "/api/v1/brainstorm/generate",
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * List brainstorm ideas for a document
   */
  async listBrainstormIdeas(
    documentId: string,
    category?: string
  ): Promise<APIResponse<BrainstormIdea[]>> {
    const queryParams = new URLSearchParams();
    if (category) queryParams.append("category", category);

    const endpoint = `/api/v1/brainstorm/documents/${documentId}/ideas${
      queryParams.toString() ? `?${queryParams.toString()}` : ""
    }`;

    return this.request<BrainstormIdea[]>(endpoint);
  }

  // ============================================================================
  // Search Operations
  // ============================================================================

  /**
   * Search across documents and variants
   */
  async search(request: SearchRequest): Promise<APIResponse<SearchResponse>> {
    return this.request<SearchResponse>("/api/v1/search/search", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // ============================================================================
  // Batch Operations
  // ============================================================================

  /**
   * Process multiple documents in batch
   */
  async batchProcess(
    request: BatchProcessingRequest
  ): Promise<APIResponse<BatchProcessingResponse>> {
    return this.request<BatchProcessingResponse>("/api/v1/batch/process", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // ============================================================================
  // Export Operations
  // ============================================================================

  /**
   * Export content
   */
  async export(request: ExportRequest): Promise<APIResponse<ExportResponse>> {
    return this.request<ExportResponse>("/api/v1/export/export", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * Download exported file
   */
  async downloadFile(fileId: string): Promise<Blob> {
    const response = await fetch(
      `${this.config.baseURL}/api/v1/export/download/${fileId}`,
      {
        headers: {
          ...(this.config.apiKey && {
            Authorization: `Bearer ${this.config.apiKey}`,
          }),
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  // ============================================================================
  // Analytics Operations
  // ============================================================================

  /**
   * Get dashboard analytics data
   */
  async getDashboard(): Promise<APIResponse<Record<string, unknown>>> {
    return this.request<Record<string, unknown>>("/api/v1/analytics/dashboard");
  }

  /**
   * Get analytics report
   */
  async getAnalyticsReport(
    startDate: string,
    endDate: string
  ): Promise<APIResponse<AnalyticsReport>> {
    const queryParams = new URLSearchParams();
    queryParams.append("start_date", startDate);
    queryParams.append("end_date", endDate);

    return this.request<AnalyticsReport>(
      `/api/v1/analytics/reports?${queryParams.toString()}`
    );
  }

  // ============================================================================
  // Health & System Operations
  // ============================================================================

  /**
   * Get system health status
   */
  async getHealth(): Promise<APIResponse<SystemHealth>> {
    return this.request<SystemHealth>("/api/v1/health/status");
  }

  /**
   * Health check (simple endpoint)
   */
  async healthCheck(): Promise<APIResponse<{ status: string; message: string }>> {
    return this.request("/health");
  }
}

// Export singleton instance helper
export function createClient(config: APIClientConfig): PDFVariantesAPIClient {
  return new PDFVariantesAPIClient(config);
}







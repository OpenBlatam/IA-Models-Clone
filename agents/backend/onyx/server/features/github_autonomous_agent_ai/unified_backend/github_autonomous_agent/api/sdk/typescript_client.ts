/**
 * TypeScript SDK para GitHub Autonomous Agent API
 * 
 * Cliente completo con tipos, retry, y manejo de errores.
 */

// ============================================================================
// Types
// ============================================================================

export interface Task {
  id: string;
  repository_owner: string;
  repository_name: string;
  instruction: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  result?: Record<string, any>;
  error?: string;
  metadata?: Record<string, any>;
}

export interface CreateTaskRequest {
  repository_owner: string;
  repository_name: string;
  instruction: string;
  metadata?: Record<string, any>;
}

export interface AgentStatus {
  is_running: boolean;
  current_task_id?: string;
  last_activity?: string;
  metadata: Record<string, any>;
}

export interface RepositoryInfo {
  name: string;
  full_name: string;
  description?: string;
  url: string;
  default_branch: string;
  language?: string;
  stars: number;
  forks: number;
  is_private: boolean;
}

export interface LLMRequest {
  prompt: string;
  system_prompt?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface LLMResponse {
  content: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  finish_reason?: string;
  error?: string;
  latency_ms?: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  services: Record<string, boolean>;
  details: Record<string, any>;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  source?: string;
}

export interface BatchCreateTasksRequest {
  tasks: CreateTaskRequest[];
}

export interface BatchCreateTasksResponse {
  total: number;
  successful: number;
  failed: number;
  tasks: Task[];
  errors: Array<{
    repository: string;
    instruction: string;
    error: string;
  }>;
}

// ============================================================================
// API Client
// ============================================================================

export interface APIClientConfig {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

export class APIClient {
  private baseURL: string;
  private apiKey?: string;
  private timeout: number;
  private retryAttempts: number;
  private retryDelay: number;

  constructor(config: APIClientConfig = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8030';
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.retryAttempts = config.retryAttempts || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    data?: any,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const config: RequestInit = {
      method,
      headers,
      signal: AbortSignal.timeout(this.timeout),
      ...options,
    };

    if (data && method !== 'GET') {
      config.body = JSON.stringify(data);
    }

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
      try {
        const response = await fetch(url, config);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new APIError(
            errorData.detail || errorData.message || response.statusText,
            response.status,
            errorData
          );
        }

        return await response.json();
      } catch (error) {
        lastError = error as Error;

        // No retry para errores 4xx (excepto 429)
        if (error instanceof APIError && error.status < 500 && error.status !== 429) {
          throw error;
        }

        // Esperar antes de reintentar
        if (attempt < this.retryAttempts - 1) {
          await this.sleep(this.retryDelay * Math.pow(2, attempt));
        }
      }
    }

    throw lastError || new Error('Request failed');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ========================================================================
  // Agent Endpoints
  // ========================================================================

  async getAgentStatus(): Promise<AgentStatus> {
    return this.request<AgentStatus>('GET', '/api/v1/agent/status');
  }

  async startAgent(): Promise<{ message: string; status: string }> {
    return this.request('POST', '/api/v1/agent/start');
  }

  async stopAgent(): Promise<{ message: string; status: string }> {
    return this.request('POST', '/api/v1/agent/stop');
  }

  async pauseAgent(): Promise<{ message: string; status: string }> {
    return this.request('POST', '/api/v1/agent/pause');
  }

  async resumeAgent(): Promise<{ message: string; status: string }> {
    return this.request('POST', '/api/v1/agent/resume');
  }

  // ========================================================================
  // Task Endpoints
  // ========================================================================

  async createTask(request: CreateTaskRequest): Promise<Task> {
    return this.request<Task>('POST', '/api/v1/tasks/', request);
  }

  async getTask(taskId: string): Promise<Task> {
    return this.request<Task>('GET', `/api/v1/tasks/${taskId}`);
  }

  async listTasks(params?: {
    status?: string;
    limit?: number;
  }): Promise<Task[]> {
    const query = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<Task[]>(
      'GET',
      `/api/v1/tasks/${query ? `?${query}` : ''}`
    );
  }

  async deleteTask(taskId: string): Promise<{ message: string }> {
    return this.request('DELETE', `/api/v1/tasks/${taskId}`);
  }

  // ========================================================================
  // Batch Endpoints
  // ========================================================================

  async batchCreateTasks(
    request: BatchCreateTasksRequest
  ): Promise<BatchCreateTasksResponse> {
    return this.request<BatchCreateTasksResponse>(
      'POST',
      '/api/v1/batch/tasks',
      request
    );
  }

  async batchDeleteTasks(
    taskIds: string[]
  ): Promise<{
    total: number;
    deleted: number;
    not_found: number;
    errors: Array<{ task_id: string; error: string }>;
  }> {
    return this.request('DELETE', '/api/v1/batch/tasks', { task_ids: taskIds });
  }

  // ========================================================================
  // GitHub Endpoints
  // ========================================================================

  async getRepositoryInfo(
    owner: string,
    repo: string
  ): Promise<RepositoryInfo> {
    return this.request<RepositoryInfo>(
      'GET',
      `/api/v1/github/repositories/${owner}/${repo}`
    );
  }

  // ========================================================================
  // LLM Endpoints
  // ========================================================================

  async generateLLM(request: LLMRequest): Promise<LLMResponse> {
    return this.request<LLMResponse>('POST', '/api/v1/llm/generate', request);
  }

  async analyzeCode(
    code: string,
    language?: string,
    model?: string
  ): Promise<LLMResponse> {
    return this.request<LLMResponse>('POST', '/api/v1/llm/analyze-code', {
      code,
      language,
      model,
    });
  }

  // ========================================================================
  // Health & Stats
  // ========================================================================

  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/health');
  }

  async getStats(): Promise<any> {
    return this.request('GET', '/api/v1/stats/overview');
  }
}

// ============================================================================
// WebSocket Client
// ============================================================================

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private messageQueue: WebSocketMessage[] = [];

  constructor(url: string = 'ws://localhost:8030/ws') {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.flushMessageQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onerror = (event) => {
          // Convert Event to Error for better error handling
          const error = new Error(
            `WebSocket connection error: Failed to connect to ${this.url}. ` +
            `ReadyState: ${this.ws?.readyState ?? 'unknown'}`
          );
          (error as any).event = event;
          (error as any).url = this.url;
          (error as any).readyState = this.ws?.readyState;
          reject(error);
        };

        this.ws.onclose = (event) => {
          // Only attempt reconnect if it wasn't a manual disconnect
          if (event.code !== 1000) {
            this.attemptReconnect();
          }
        };
      } catch (error) {
        const err = error instanceof Error 
          ? error 
          : new Error(`Failed to create WebSocket: ${String(error)}`);
        reject(err);
      }
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    // Notificar listeners por tipo
    const typeListeners = this.listeners.get(message.type);
    if (typeListeners) {
      typeListeners.forEach(listener => listener(message.data));
    }

    // Notificar listeners generales
    const allListeners = this.listeners.get('*');
    if (allListeners) {
      allListeners.forEach(listener => listener(message));
    }
  }

  on(eventType: string, callback: (data: any) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);
  }

  off(eventType: string, callback: (data: any) => void): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.messageQueue.push(message);
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
      }
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect().catch(() => {
          // Reintento fallido, se intentará de nuevo
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
    this.messageQueue = [];
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// ============================================================================
// Error Classes
// ============================================================================

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// ============================================================================
// Default Export
// ============================================================================

export default APIClient;




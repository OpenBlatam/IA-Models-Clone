import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  DocumentRequest,
  DocumentResponse,
  TaskStatus,
  TaskListResponse,
  DocumentsResponse,
  HealthResponse,
  StatsResponse,
  WebSocketMessage,
} from '@/types/api';

interface RetryConfig {
  retries: number;
  retryDelay: number;
  retryCondition?: (error: AxiosError) => boolean;
}

const defaultRetryConfig: RetryConfig = {
  retries: 3,
  retryDelay: 1000,
  retryCondition: (error) => {
    return error.response?.status === 429 || error.response?.status === 503 || !error.response;
  },
};

export class BULApiClient {
  private client: AxiosInstance;
  private wsBaseUrl: string;
  private wsConnections: Map<string, WebSocket> = new Map();

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') {
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor with retry logic
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const config = error.config as any;
        
        if (!config || !config.retry) {
          config.retry = { ...defaultRetryConfig };
        }

        const { retries, retryDelay, retryCondition } = config.retry;

        if (retries > 0 && retryCondition && retryCondition(error)) {
          config.retry.retries -= 1;
          
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
          
          return this.client(config);
        }

        return Promise.reject(error);
      }
    );

    this.wsBaseUrl = baseUrl.replace(/^http/, 'ws');
  }

  // System Endpoints
  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await this.client.get<HealthResponse>('/api/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al verificar el estado del sistema');
    }
  }

  async getStats(): Promise<StatsResponse> {
    try {
      const response = await this.client.get<StatsResponse>('/api/stats');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al obtener estadísticas');
    }
  }

  // Document Endpoints
  async generateDocument(request: DocumentRequest): Promise<DocumentResponse> {
    try {
      const response = await this.client.post<DocumentResponse>('/api/documents/generate', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al generar documento');
    }
  }

  async getTaskDocument(taskId: string) {
    try {
      const response = await this.client.get(`/api/tasks/${taskId}/document`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al obtener documento');
    }
  }

  async listDocuments(limit: number = 50, offset: number = 0): Promise<DocumentsResponse> {
    try {
      const response = await this.client.get<DocumentsResponse>('/api/documents', {
        params: { limit, offset },
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al listar documentos');
    }
  }

  // Task Endpoints
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    try {
      const response = await this.client.get<TaskStatus>(`/api/tasks/${taskId}/status`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al obtener estado de tarea');
    }
  }

  async listTasks(options: {
    status?: string;
    user_id?: string;
    limit?: number;
    offset?: number;
    search?: string;
  } = {}): Promise<TaskListResponse> {
    try {
      const response = await this.client.get<TaskListResponse>('/api/tasks', {
        params: options,
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Error al listar tareas');
    }
  }

  async deleteTask(taskId: string): Promise<void> {
    try {
      await this.client.delete(`/api/tasks/${taskId}`);
    } catch (error) {
      throw this.handleError(error, 'Error al eliminar tarea');
    }
  }

  async cancelTask(taskId: string): Promise<void> {
    try {
      await this.client.post(`/api/tasks/${taskId}/cancel`);
    } catch (error) {
      throw this.handleError(error, 'Error al cancelar tarea');
    }
  }

  // WebSocket with reconnection logic
  connectTaskWebSocket(
    taskId: string,
    onMessage: (message: WebSocketMessage) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
  ): Promise<WebSocket> {
    const wsKey = `task_${taskId}`;

    if (this.wsConnections.has(wsKey)) {
      const existing = this.wsConnections.get(wsKey);
      if (existing && existing.readyState === WebSocket.OPEN) {
        return Promise.resolve(existing);
      }
    }

    const ws = new WebSocket(`${this.wsBaseUrl}/api/ws/${taskId}`);

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      this.wsConnections.delete(wsKey);
      if (onClose) onClose();
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
      }, 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        this.wsConnections.set(wsKey, ws);
        
        // Send ping periodically to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          } else {
            clearInterval(pingInterval);
          }
        }, 30000);

        resolve(ws);
      };

      ws.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
    });
  }

  disconnectTaskWebSocket(taskId: string): void {
    const wsKey = `task_${taskId}`;
    const ws = this.wsConnections.get(wsKey);
    if (ws) {
      ws.close();
      this.wsConnections.delete(wsKey);
    }
  }

  disconnectAllWebSockets(): void {
    this.wsConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
    this.wsConnections.clear();
  }

  private handleError(error: any, defaultMessage: string): Error {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.detail || error.message || defaultMessage;
      return new Error(message);
    }
    return error instanceof Error ? error : new Error(defaultMessage);
  }
}

// Singleton instance
export const apiClient = new BULApiClient();

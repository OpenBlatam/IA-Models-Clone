/**
 * BUL API Client
 * ==============
 * 
 * Cliente TypeScript para consumir la API de BUL desde el frontend
 */

import type {
  DocumentRequest,
  DocumentResponse,
  TaskStatus,
  HealthResponse,
  StatsResponse,
  TaskListResponse,
  DocumentListResponse,
  TaskDocumentResponse,
  ApiConfig,
  ErrorResponse
} from './frontend_types';

// ============================================================================
// API Client Class
// ============================================================================

export interface WebSocketMessage {
  type: 'task_update' | 'initial_state' | 'connected' | 'error' | 'pong';
  task_id?: string;
  data?: any;
  message?: string;
  timestamp: string;
}

export type WebSocketCallback = (message: WebSocketMessage) => void;

export class BULApiClient {
  private baseUrl: string;
  private wsBaseUrl: string;
  private timeout: number;
  private headers: Record<string, string>;
  private wsConnections: Map<string, WebSocket> = new Map();

  constructor(config: ApiConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    // Convert HTTP to WebSocket URL
    this.wsBaseUrl = this.baseUrl.replace(/^http/, 'ws');
    this.timeout = config.timeout || 30000; // Default 30 seconds
    this.headers = {
      'Content-Type': 'application/json',
      ...config.headers
    };
  }

  /**
   * Realiza una petición HTTP con manejo de errores
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.headers,
          ...options.headers
        },
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`
        }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  // ============================================================================
  // System Endpoints
  // ============================================================================

  /**
   * Obtiene información del sistema
   */
  async getRoot(): Promise<{
    message: string;
    version: string;
    status: string;
    timestamp: string;
    docs: string;
    health: string;
  }> {
    return this.request('/');
  }

  /**
   * Health check del sistema
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  /**
   * Obtiene estadísticas del sistema
   */
  async getStats(): Promise<StatsResponse> {
    return this.request<StatsResponse>('/api/stats');
  }

  // ============================================================================
  // Document Endpoints
  // ============================================================================

  /**
   * Genera un nuevo documento
   */
  async generateDocument(request: DocumentRequest): Promise<DocumentResponse> {
    return this.request<DocumentResponse>('/api/documents/generate', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  /**
   * Obtiene el documento generado de una tarea
   */
  async getTaskDocument(taskId: string): Promise<TaskDocumentResponse> {
    return this.request<TaskDocumentResponse>(`/api/tasks/${taskId}/document`);
  }

  /**
   * Lista documentos generados
   */
  async listDocuments(
    limit: number = 50,
    offset: number = 0
  ): Promise<DocumentListResponse> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    });
    return this.request<DocumentListResponse>(`/api/documents?${params}`);
  }

  // ============================================================================
  // Task Endpoints
  // ============================================================================

  /**
   * Obtiene el estado de una tarea
   */
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    return this.request<TaskStatus>(`/api/tasks/${taskId}/status`);
  }

  /**
   * Lista tareas con filtrado y paginación
   */
  async listTasks(options: {
    status?: string;
    user_id?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<TaskListResponse> {
    const params = new URLSearchParams();
    if (options.status) params.append('status', options.status);
    if (options.user_id) params.append('user_id', options.user_id);
    if (options.limit) params.append('limit', options.limit.toString());
    if (options.offset) params.append('offset', options.offset.toString());

    const queryString = params.toString();
    return this.request<TaskListResponse>(
      `/api/tasks${queryString ? `?${queryString}` : ''}`
    );
  }

  /**
   * Elimina una tarea
   */
  async deleteTask(taskId: string): Promise<{ message: string; task_id: string }> {
    return this.request(`/api/tasks/${taskId}`, {
      method: 'DELETE'
    });
  }

  /**
   * Cancela una tarea en ejecución
   */
  async cancelTask(taskId: string): Promise<{ message: string; task_id: string }> {
    return this.request(`/api/tasks/${taskId}/cancel`, {
      method: 'POST'
    });
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Polling para esperar a que una tarea se complete
   */
  async waitForTaskCompletion(
    taskId: string,
    options: {
      interval?: number; // Intervalo de polling en ms
      maxAttempts?: number; // Máximo número de intentos
      onProgress?: (status: TaskStatus) => void; // Callback de progreso
    } = {}
  ): Promise<TaskStatus> {
    const interval = options.interval || 2000; // Default 2 seconds
    const maxAttempts = options.maxAttempts || 150; // Default 5 minutes (150 * 2s)

    let attempts = 0;

    while (attempts < maxAttempts) {
      const status = await this.getTaskStatus(taskId);

      if (options.onProgress) {
        options.onProgress(status);
      }

      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }

      if (status.status === 'cancelled') {
        throw new Error('Task was cancelled');
      }

      await new Promise(resolve => setTimeout(resolve, interval));
      attempts++;
    }

    throw new Error('Task completion timeout');
  }

  /**
   * Genera un documento y espera a que se complete
   */
  async generateDocumentAndWait(
    request: DocumentRequest,
    options: {
      pollingInterval?: number;
      maxAttempts?: number;
      onProgress?: (status: TaskStatus) => void;
      useWebSocket?: boolean;
    } = {}
  ): Promise<TaskDocumentResponse> {
    const response = await this.generateDocument(request);
    
    // Usar WebSocket si está disponible y se solicita
    if (options.useWebSocket !== false) {
      try {
        return await this.waitForTaskCompletionWebSocket(response.task_id, {
          onProgress: options.onProgress,
          timeout: (options.maxAttempts || 150) * (options.pollingInterval || 2000)
        });
      } catch (error) {
        // Fallback a polling si WebSocket falla
        console.warn('WebSocket failed, falling back to polling:', error);
      }
    }
    
    const finalStatus = await this.waitForTaskCompletion(response.task_id, {
      interval: options.pollingInterval,
      maxAttempts: options.maxAttempts,
      onProgress: options.onProgress
    });

    if (finalStatus.status === 'failed') {
      throw new Error(finalStatus.error || 'Document generation failed');
    }

    return this.getTaskDocument(response.task_id);
  }

  // ============================================================================
  // WebSocket Methods
  // ============================================================================

  /**
   * Conecta a WebSocket para recibir actualizaciones de una tarea específica
   */
  async connectTaskWebSocket(
    taskId: string,
    onMessage: WebSocketCallback
  ): Promise<WebSocket> {
    const wsKey = `task_${taskId}`;
    
    // Reutilizar conexión existente si existe
    if (this.wsConnections.has(wsKey)) {
      const existing = this.wsConnections.get(wsKey);
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
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
    };

    ws.onclose = () => {
      this.wsConnections.delete(wsKey);
    };

    // Esperar a que se abra la conexión
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
      setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
    });

    this.wsConnections.set(wsKey, ws);
    return ws;
  }

  /**
   * Conecta a WebSocket para recibir actualizaciones de todas las tareas
   */
  async connectAllTasksWebSocket(
    onMessage: WebSocketCallback
  ): Promise<WebSocket> {
    const wsKey = 'all_tasks';
    
    if (this.wsConnections.has(wsKey)) {
      const existing = this.wsConnections.get(wsKey);
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
    }

    const ws = new WebSocket(`${this.wsBaseUrl}/api/ws`);
    
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
    };

    ws.onclose = () => {
      this.wsConnections.delete(wsKey);
    };

    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
      setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
    });

    this.wsConnections.set(wsKey, ws);
    return ws;
  }

  /**
   * Espera a que una tarea se complete usando WebSocket
   */
  async waitForTaskCompletionWebSocket(
    taskId: string,
    options: {
      onProgress?: (status: TaskStatus) => void;
      timeout?: number;
    } = {}
  ): Promise<TaskDocumentResponse> {
    return new Promise(async (resolve, reject) => {
      const timeout = options.timeout || 300000; // 5 minutos por defecto
      let ws: WebSocket | null = null;
      let timeoutId: NodeJS.Timeout | null = null;

      const cleanup = () => {
        if (timeoutId) clearTimeout(timeoutId);
        if (ws) {
          ws.close();
          this.wsConnections.delete(`task_${taskId}`);
        }
      };

      try {
        ws = await this.connectTaskWebSocket(taskId, (message) => {
          if (message.type === 'task_update' && message.data) {
            const data = message.data;
            
            // Convertir a TaskStatus
            const finalStatus: TaskStatus = {
              task_id: taskId,
              status: data.status,
              progress: data.progress || 0,
              result: data.result,
              error: data.error,
              created_at: message.timestamp,
              updated_at: message.timestamp,
              processing_time: data.processing_time
            };

            if (options.onProgress) {
              options.onProgress(finalStatus);
            }

            if (data.status === 'completed') {
              cleanup();
              this.getTaskDocument(taskId).then(resolve).catch(reject);
            } else if (data.status === 'failed') {
              cleanup();
              reject(new Error(data.error || 'Task failed'));
            }
          } else if (message.type === 'initial_state' && message.data) {
            const data = message.data;
            const finalStatus: TaskStatus = {
              task_id: taskId,
              status: data.status,
              progress: data.progress || 0,
              result: data.result,
              error: data.error,
              created_at: message.timestamp,
              updated_at: message.timestamp
            };

            if (options.onProgress) {
              options.onProgress(finalStatus);
            }

            if (data.status === 'completed') {
              cleanup();
              this.getTaskDocument(taskId).then(resolve).catch(reject);
              return;
            }
          } else if (message.type === 'error') {
            cleanup();
            reject(new Error(message.message || 'WebSocket error'));
          }
        });

        timeoutId = setTimeout(() => {
          cleanup();
          reject(new Error('Task completion timeout'));
        }, timeout);
      } catch (error) {
        cleanup();
        reject(error);
      }
    });
  }

  /**
   * Desconecta todas las conexiones WebSocket
   */
  disconnectAllWebSockets(): void {
    this.wsConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
    this.wsConnections.clear();
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Crea una instancia del cliente API
 */
export function createBULApiClient(config: ApiConfig): BULApiClient {
  return new BULApiClient(config);
}

// ============================================================================
// Default Export
// ============================================================================

export default BULApiClient;


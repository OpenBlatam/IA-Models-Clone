/**
 * BUL API Client - JavaScript/Node.js
 * ====================================
 * 
 * Cliente JavaScript para consumir la API de BUL desde Node.js o navegador
 */

class BULApiClient {
  constructor(config) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.wsBaseUrl = this.baseUrl.replace(/^http/, 'ws');
    this.timeout = config.timeout || 30000;
    this.headers = {
      'Content-Type': 'application/json',
      ...config.headers
    };
    this.wsConnections = new Map();
  }

  /**
   * Realiza una petición HTTP
   */
  async request(endpoint, options = {}) {
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
        const errorData = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`
        }));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }

  // System Endpoints
  async getRoot() {
    return this.request('/');
  }

  async getHealth() {
    return this.request('/api/health');
  }

  async getStats() {
    return this.request('/api/stats');
  }

  // Document Endpoints
  async generateDocument(request) {
    return this.request('/api/documents/generate', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  async getTaskDocument(taskId) {
    return this.request(`/api/tasks/${taskId}/document`);
  }

  async listDocuments(limit = 50, offset = 0) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    });
    return this.request(`/api/documents?${params}`);
  }

  // Task Endpoints
  async getTaskStatus(taskId) {
    return this.request(`/api/tasks/${taskId}/status`);
  }

  async listTasks(options = {}) {
    const params = new URLSearchParams();
    if (options.status) params.append('status', options.status);
    if (options.user_id) params.append('user_id', options.user_id);
    if (options.limit) params.append('limit', options.limit.toString());
    if (options.offset) params.append('offset', options.offset.toString());

    const queryString = params.toString();
    return this.request(`/api/tasks${queryString ? `?${queryString}` : ''}`);
  }

  async deleteTask(taskId) {
    return this.request(`/api/tasks/${taskId}`, {
      method: 'DELETE'
    });
  }

  async cancelTask(taskId) {
    return this.request(`/api/tasks/${taskId}/cancel`, {
      method: 'POST'
    });
  }

  // Polling
  async waitForTaskCompletion(taskId, options = {}) {
    const interval = options.interval || 2000;
    const maxAttempts = options.maxAttempts || 150;

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

  // Generate and wait
  async generateDocumentAndWait(request, options = {}) {
    const response = await this.generateDocument(request);
    
    if (options.useWebSocket !== false) {
      try {
        return await this.waitForTaskCompletionWebSocket(response.task_id, {
          onProgress: options.onProgress,
          timeout: (options.maxAttempts || 150) * (options.pollingInterval || 2000)
        });
      } catch (error) {
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

  // WebSocket
  async connectTaskWebSocket(taskId, onMessage) {
    const wsKey = `task_${taskId}`;
    
    if (this.wsConnections.has(wsKey)) {
      const existing = this.wsConnections.get(wsKey);
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
    }

    const ws = new WebSocket(`${this.wsBaseUrl}/api/ws/${taskId}`);
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
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

    return new Promise((resolve, reject) => {
      ws.onopen = () => {
        this.wsConnections.set(wsKey, ws);
        resolve(ws);
      };
      ws.onerror = reject;
      setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
    });
  }

  async waitForTaskCompletionWebSocket(taskId, options = {}) {
    return new Promise(async (resolve, reject) => {
      const timeout = options.timeout || 300000;
      let ws = null;
      let timeoutId = null;

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
            
            const finalStatus = {
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
            if (data.status === 'completed') {
              cleanup();
              this.getTaskDocument(taskId).then(resolve).catch(reject);
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

  disconnectAllWebSockets() {
    this.wsConnections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
    this.wsConnections.clear();
  }
}

// Factory function
function createBULApiClient(config) {
  return new BULApiClient(config);
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  // Node.js
  module.exports = { BULApiClient, createBULApiClient };
} else if (typeof window !== 'undefined') {
  // Browser
  window.BULApiClient = BULApiClient;
  window.createBULApiClient = createBULApiClient;
}

































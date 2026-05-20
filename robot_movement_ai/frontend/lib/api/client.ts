import axios, { AxiosInstance, AxiosError } from 'axios';
import { toast } from '../utils/toast';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010';

export const apiClient = {
  client: axios.create({
    baseURL: API_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  }),

  // Robot Control
  async moveTo(position: { x: number; y: number; z: number }) {
    const response = await this.client.post('/api/v1/move/to', position);
    return response.data;
  },

  async movePath(path: Array<{ x: number; y: number; z: number }>) {
    const response = await this.client.post('/api/v1/move/path', { path });
    return response.data;
  },

  async stop() {
    const response = await this.client.post('/api/v1/stop');
    return response.data;
  },

  async getStatus() {
    const response = await this.client.get('/api/v1/status');
    return response.data;
  },

  async getStatistics() {
    const response = await this.client.get('/api/v1/statistics');
    return response.data;
  },

  // Chat
  async sendChatMessage(message: string) {
    const response = await this.client.post('/api/v1/chat', { message });
    return response.data;
  },

  // Test connection
  async testConnection() {
    try {
      const response = await this.client.get('/health');
      return { connected: true, data: response.data };
    } catch (error) {
      return { connected: false, error };
    }
  },

  // Metrics
  async getMetrics() {
    const response = await this.client.get('/api/v1/metrics');
    return response.data;
  },

  async getResources() {
    const response = await this.client.get('/api/v1/resources');
    return response.data;
  },

  // System
  async getHealth() {
    const response = await this.client.get('/health');
    return response.data;
  },

  async getSystemVersion() {
    const response = await this.client.get('/api/v1/system/version');
    return response.data;
  },

  async getConfig() {
    const response = await this.client.get('/api/v1/system/config');
    return response.data;
  },

  // Movement History
  async getMovementHistory(limit: number = 100) {
    const response = await this.client.get(`/api/v1/movement/history?limit=${limit}`);
    return response.data;
  },

  // Trajectory
  async optimizeTrajectoryAStar(start: any, goal: any) {
    const response = await this.client.post('/api/v1/trajectory/optimize/astar', { start, goal });
    return response.data;
  },

  async optimizeTrajectoryRRT(start: any, goal: any) {
    const response = await this.client.post('/api/v1/trajectory/optimize/rrt', { start, goal });
    return response.data;
  },

  // Root endpoint
  async getRoot() {
    const response = await this.client.get('/');
    return response.data;
  },
};

// Request interceptor
apiClient.client.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.client.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    // Don't show toast for connection errors on health checks
    const isHealthCheck = error.config?.url?.includes('/health');
    
    if (error.response) {
      const status = error.response.status;
      const message = (error.response.data as any)?.detail || (error.response.data as any)?.message || error.message;

      if (!isHealthCheck) {
        switch (status) {
          case 401:
            toast.error('No autorizado');
            break;
          case 403:
            toast.error('Acceso denegado');
            break;
          case 404:
            toast.error('Recurso no encontrado');
            break;
          case 500:
            toast.error('Error del servidor');
            break;
          default:
            toast.error(`Error: ${message}`);
        }
      }
    } else if (error.request) {
      if (!isHealthCheck) {
        toast.error('No se pudo conectar al servidor. Verifica que el backend esté corriendo.');
      }
    } else {
      if (!isHealthCheck) {
        toast.error(`Error: ${error.message}`);
      }
    }

    return Promise.reject(error);
  }
);

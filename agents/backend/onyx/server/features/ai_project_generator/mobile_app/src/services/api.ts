import axios, { AxiosInstance, AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_CONFIG, API_ENDPOINTS } from '../config/api';
import {
  Project,
  ProjectRequest,
  ProjectResponse,
  GenerationTask,
  ValidationResult,
  QueueStatus,
  Stats,
  ApiError,
} from '../types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create(API_CONFIG);
    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.client.interceptors.request.use(
      async (config) => {
        const token = await AsyncStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          await AsyncStorage.removeItem('auth_token');
        }
        return Promise.reject(error);
      }
    );
  }

  private handleError(error: unknown): ApiError {
    if (axios.isAxiosError(error)) {
      return {
        detail: error.response?.data?.detail || error.message || 'An error occurred',
        status_code: error.response?.status,
      };
    }
    return {
      detail: 'An unexpected error occurred',
    };
  }

  // Generation
  async generateProject(request: ProjectRequest, asyncGeneration = true): Promise<ProjectResponse> {
    try {
      const response = await this.client.post<ProjectResponse>(
        API_ENDPOINTS.GENERATE,
        request,
        { params: { async_generation: asyncGeneration } }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async batchGenerate(projects: ProjectRequest[], parallel = true): Promise<ProjectResponse[]> {
    try {
      const response = await this.client.post<ProjectResponse[]>(
        API_ENDPOINTS.GENERATE_BATCH,
        { projects, parallel }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getGenerationStatus(taskId: string): Promise<GenerationTask> {
    try {
      const response = await this.client.get<GenerationTask>(
        API_ENDPOINTS.GENERATION_TASK(taskId)
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Projects
  async createProject(request: ProjectRequest): Promise<ProjectResponse> {
    try {
      const response = await this.client.post<ProjectResponse>(
        API_ENDPOINTS.PROJECTS,
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getProject(projectId: string): Promise<Project> {
    try {
      const response = await this.client.get<Project>(
        API_ENDPOINTS.PROJECT(projectId)
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async listProjects(params?: {
    status?: string;
    author?: string;
    limit?: number;
    offset?: number;
  }): Promise<Project[]> {
    try {
      const response = await this.client.get<Project[]>(
        API_ENDPOINTS.PROJECTS,
        { params }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async deleteProject(projectId: string): Promise<void> {
    try {
      await this.client.delete(API_ENDPOINTS.PROJECT(projectId));
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getQueueStatus(): Promise<QueueStatus> {
    try {
      const response = await this.client.get<QueueStatus>(
        API_ENDPOINTS.PROJECT_QUEUE_STATUS
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Status & Monitoring
  async getStatus(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.STATUS);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getStats(): Promise<Stats> {
    try {
      const response = await this.client.get<Stats>(API_ENDPOINTS.STATS);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getQueue(): Promise<Project[]> {
    try {
      const response = await this.client.get<Project[]>(API_ENDPOINTS.QUEUE);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Control
  async startGenerator(): Promise<any> {
    try {
      const response = await this.client.post(API_ENDPOINTS.START);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async stopGenerator(): Promise<any> {
    try {
      const response = await this.client.post(API_ENDPOINTS.STOP);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Export
  async exportZip(projectId: string): Promise<Blob> {
    try {
      const response = await this.client.post(
        API_ENDPOINTS.EXPORT_ZIP,
        { project_id: projectId },
        { responseType: 'blob' }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async exportTar(projectId: string): Promise<Blob> {
    try {
      const response = await this.client.post(
        API_ENDPOINTS.EXPORT_TAR,
        { project_id: projectId },
        { responseType: 'blob' }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Validation
  async validateProject(projectId: string): Promise<ValidationResult> {
    try {
      const response = await this.client.post<ValidationResult>(
        API_ENDPOINTS.VALIDATE,
        { project_id: projectId }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Health
  async healthCheck(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.HEALTH);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async detailedHealthCheck(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.HEALTH_DETAILED);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Analytics
  async getAnalyticsTrends(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.ANALYTICS_TRENDS);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getTopAiTypes(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.ANALYTICS_TOP_AI_TYPES);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Performance
  async getPerformanceStats(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.PERFORMANCE_STATS);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getPerformanceOptimize(): Promise<any> {
    try {
      const response = await this.client.get(API_ENDPOINTS.PERFORMANCE_OPTIMIZE);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }
}

export const apiService = new ApiService();


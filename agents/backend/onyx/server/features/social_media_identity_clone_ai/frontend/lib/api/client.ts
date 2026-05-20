import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  ExtractProfileRequest,
  ExtractProfileResponse,
  BuildIdentityRequest,
  BuildIdentityResponse,
  GenerateContentRequest,
  GenerateContentResponse,
  IdentityProfile,
  GeneratedContent,
  TaskResponse,
  Task,
  MetricsResponse,
  DashboardResponse,
  AlertsResponse,
  Template,
  ABTest,
  Recommendation,
  ContentValidation,
  ApiError,
} from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const BASE_URL = `${API_URL}/api/v1`;

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        const message = error.response?.data?.detail || error.message || 'An error occurred';
        return Promise.reject(new Error(message));
      }
    );
  }

  setApiKey(apiKey: string): void {
    this.client.defaults.headers.common['X-API-Key'] = apiKey;
  }

  async extractProfile(request: ExtractProfileRequest): Promise<ExtractProfileResponse> {
    const response = await this.client.post<ExtractProfileResponse>('/extract-profile', request);
    return response.data;
  }

  async buildIdentity(request: BuildIdentityRequest): Promise<BuildIdentityResponse> {
    const response = await this.client.post<BuildIdentityResponse>('/build-identity', request);
    return response.data;
  }

  async generateContent(request: GenerateContentRequest): Promise<GenerateContentResponse> {
    const response = await this.client.post<GenerateContentResponse>('/generate-content', request);
    return response.data;
  }

  async getIdentity(identityId: string): Promise<IdentityProfile> {
    const response = await this.client.get<{ success: boolean; identity: IdentityProfile }>(
      `/identity/${identityId}`
    );
    return response.data.identity;
  }

  async getGeneratedContent(identityId: string, limit = 10): Promise<GeneratedContent[]> {
    const response = await this.client.get<{
      success: boolean;
      count: number;
      content: GeneratedContent[];
    }>(`/identity/${identityId}/generated-content`, {
      params: { limit },
    });
    return response.data.content;
  }

  async createExtractProfileTask(request: ExtractProfileRequest): Promise<TaskResponse> {
    const response = await this.client.post<TaskResponse>('/tasks/extract-profile', request);
    return response.data;
  }

  async createBuildIdentityTask(request: BuildIdentityRequest): Promise<TaskResponse> {
    const response = await this.client.post<TaskResponse>('/tasks/build-identity', request);
    return response.data;
  }

  async getTask(taskId: string): Promise<Task> {
    const response = await this.client.get<{ success: boolean; task: Task }>(`/tasks/${taskId}`);
    return response.data.task;
  }

  async getTasks(): Promise<Task[]> {
    const response = await this.client.get<{ success: boolean; tasks: Task[] }>('/tasks');
    return response.data.tasks;
  }

  async getMetrics(): Promise<MetricsResponse> {
    const response = await this.client.get<MetricsResponse>('/metrics');
    return response.data;
  }

  async getAnalyticsStats(): Promise<unknown> {
    const response = await this.client.get('/analytics/stats');
    return response.data;
  }

  async getAnalyticsForIdentity(identityId: string): Promise<unknown> {
    const response = await this.client.get(`/analytics/identity/${identityId}`);
    return response.data;
  }

  async getAnalyticsTrends(): Promise<unknown> {
    const response = await this.client.get('/analytics/trends');
    return response.data;
  }

  async getDashboard(): Promise<DashboardResponse> {
    const response = await this.client.get<DashboardResponse>('/dashboard');
    return response.data;
  }

  async getAlerts(params?: {
    unacknowledged_only?: boolean;
    severity?: string;
  }): Promise<AlertsResponse> {
    const response = await this.client.get<AlertsResponse>('/alerts', { params });
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<void> {
    await this.client.post(`/alerts/${alertId}/acknowledge`);
  }

  async resolveAlert(alertId: string): Promise<void> {
    await this.client.post(`/alerts/${alertId}/resolve`);
  }

  async getTemplates(): Promise<Template[]> {
    const response = await this.client.get<{ success: boolean; templates: Template[] }>('/templates');
    return response.data.templates;
  }

  async getTemplate(templateId: string): Promise<Template> {
    const response = await this.client.get<{ success: boolean; template: Template }>(
      `/templates/${templateId}`
    );
    return response.data.template;
  }

  async createTemplate(template: Omit<Template, 'template_id' | 'created_at'>): Promise<Template> {
    const response = await this.client.post<{ success: boolean; template: Template }>(
      '/templates',
      template
    );
    return response.data.template;
  }

  async deleteTemplate(templateId: string): Promise<void> {
    await this.client.delete(`/templates/${templateId}`);
  }

  async validateContent(contentId: string): Promise<ContentValidation> {
    const response = await this.client.post<{ success: boolean; validation: ContentValidation }>(
      `/content/${contentId}/validate`
    );
    return response.data.validation;
  }

  async validateContentDirect(content: string): Promise<ContentValidation> {
    const response = await this.client.post<{ success: boolean; validation: ContentValidation }>(
      '/content/validate',
      { content }
    );
    return response.data.validation;
  }

  async predictPerformance(
    content: string,
    platform: string,
    identityId: string
  ): Promise<unknown> {
    const response = await this.client.post('/ml/predict-performance', {
      content,
      platform,
      identity_id: identityId,
    });
    return response.data;
  }

  async analyzeTrends(identityId: string): Promise<unknown> {
    const response = await this.client.get(`/ml/analyze-trends/${identityId}`);
    return response.data;
  }

  async shareIdentity(
    identityId: string,
    sharedWithUserId: string,
    permissionLevel: string,
    sharedByUserId: string
  ): Promise<void> {
    await this.client.post('/collaboration/share', {
      identity_id: identityId,
      shared_with_user_id: sharedWithUserId,
      permission_level: permissionLevel,
      shared_by_user_id: sharedByUserId,
    });
  }

  async getSharedIdentities(): Promise<IdentityProfile[]> {
    const response = await this.client.get<{ success: boolean; identities: IdentityProfile[] }>(
      '/collaboration/shared'
    );
    return response.data.identities;
  }

  async revokeShare(shareId: string): Promise<void> {
    await this.client.delete(`/collaboration/share/${shareId}`);
  }

  async getRecommendationsForIdentity(identityId: string): Promise<Recommendation[]> {
    const response = await this.client.get<{
      success: boolean;
      recommendations: Recommendation[];
    }>(`/recommendations/identity/${identityId}`);
    return response.data.recommendations;
  }

  async getSystemRecommendations(): Promise<Recommendation[]> {
    const response = await this.client.get<{
      success: boolean;
      recommendations: Recommendation[];
    }>('/recommendations/system');
    return response.data.recommendations;
  }

  async createABTest(test: Omit<ABTest, 'test_id'>): Promise<ABTest> {
    const response = await this.client.post<{ success: boolean; test: ABTest }>(
      '/ab-tests/create',
      test
    );
    return response.data.test;
  }

  async startABTest(testId: string): Promise<void> {
    await this.client.post(`/ab-tests/${testId}/start`);
  }

  async stopABTest(testId: string): Promise<void> {
    await this.client.post(`/ab-tests/${testId}/stop`);
  }

  async getABTest(testId: string): Promise<ABTest> {
    const response = await this.client.get<{ success: boolean; test: ABTest }>(
      `/ab-tests/${testId}`
    );
    return response.data.test;
  }

  async getABTestWinner(testId: string): Promise<unknown> {
    const response = await this.client.get(`/ab-tests/${testId}/winner`);
    return response.data;
  }

  async createBackup(): Promise<{ success: boolean; backup_id: string }> {
    const response = await this.client.post<{ success: boolean; backup_id: string }>(
      '/backup/create'
    );
    return response.data;
  }

  async listBackups(): Promise<unknown[]> {
    const response = await this.client.get<{ success: boolean; backups: unknown[] }>(
      '/backup/list'
    );
    return response.data.backups;
  }

  async restoreBackup(backupId: string): Promise<void> {
    await this.client.post('/backup/restore', { backup_id: backupId });
  }

  async cleanupBackups(): Promise<void> {
    await this.client.post('/backup/cleanup');
  }

  async exportIdentityJSON(identityId: string): Promise<unknown> {
    const response = await this.client.get(`/export/identity/${identityId}/json`);
    return response.data;
  }

  async exportIdentityCSV(identityId: string): Promise<string> {
    const response = await this.client.get(`/export/identity/${identityId}/csv`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async getIdentities(): Promise<IdentityProfile[]> {
    const response = await this.client.get<{ success: boolean; identities: IdentityProfile[] }>(
      '/identities'
    );
    return response.data.identities;
  }

  async searchIdentities(filters: unknown): Promise<IdentityProfile[]> {
    const response = await this.client.post<{ success: boolean; identities: IdentityProfile[] }>(
      '/search/identities',
      filters
    );
    return response.data.identities;
  }

  async searchContent(query: string): Promise<GeneratedContent[]> {
    const response = await this.client.get<{ success: boolean; content: GeneratedContent[] }>(
      '/search/content',
      { params: { q: query } }
    );
    return response.data.content;
  }

  async createBatchExtractProfiles(requests: ExtractProfileRequest[]): Promise<TaskResponse> {
    const response = await this.client.post<TaskResponse>('/batch/extract-profiles', {
      profiles: requests,
    });
    return response.data;
  }

  async createBatchGenerateContent(requests: GenerateContentRequest[]): Promise<TaskResponse> {
    const response = await this.client.post<TaskResponse>('/batch/generate-content', {
      content: requests,
    });
    return response.data;
  }

  async createVersion(identityId: string): Promise<{ success: boolean; version_id: string }> {
    const response = await this.client.post<{ success: boolean; version_id: string }>(
      `/identity/${identityId}/version`
    );
    return response.data;
  }

  async getVersions(identityId: string): Promise<unknown[]> {
    const response = await this.client.get<{ success: boolean; versions: unknown[] }>(
      `/identity/${identityId}/versions`
    );
    return response.data.versions;
  }

  async restoreVersion(identityId: string, version: string): Promise<void> {
    await this.client.post(`/identity/${identityId}/restore/${version}`);
  }

  async registerWebhook(url: string, events: string[]): Promise<void> {
    await this.client.post('/webhooks/register', { url, events });
  }

  async getNotifications(): Promise<unknown[]> {
    const response = await this.client.get<{ success: boolean; notifications: unknown[] }>(
      '/notifications'
    );
    return response.data.notifications;
  }

  async markNotificationRead(notificationId: string): Promise<void> {
    await this.client.post(`/notifications/${notificationId}/read`);
  }

  async markAllNotificationsRead(): Promise<void> {
    await this.client.post('/notifications/read-all');
  }

  async createSchedule(
    identityId: string,
    scheduleType: string,
    config: unknown
  ): Promise<unknown> {
    const response = await this.client.post('/scheduler/create', {
      identity_id: identityId,
      schedule_type: scheduleType,
      config,
    });
    return response.data;
  }

  async getSchedules(): Promise<unknown[]> {
    const response = await this.client.get<{ success: boolean; schedules: unknown[] }>(
      '/scheduler'
    );
    return response.data.schedules;
  }

  async getPlugins(): Promise<unknown[]> {
    const response = await this.client.get<{ success: boolean; plugins: unknown[] }>('/plugins');
    return response.data.plugins;
  }

  async getPlugin(pluginId: string): Promise<unknown> {
    const response = await this.client.get<{ success: boolean; plugin: unknown }>(
      `/plugins/${pluginId}`
    );
    return response.data.plugin;
  }

  async healthCheck(): Promise<unknown> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export const apiClient = new ApiClient();


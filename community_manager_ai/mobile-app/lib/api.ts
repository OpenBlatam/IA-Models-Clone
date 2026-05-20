import axios, { AxiosInstance } from 'axios';
import * as SecureStore from 'expo-secure-store';
import Constants from 'expo-constants';
import type {
  Post,
  PostCreate,
  Meme,
  MemeCreate,
  CalendarEvent,
  Platform,
  PlatformConnect,
  Analytics,
  DashboardOverview,
  EngagementSummary,
  Template,
  TemplateCreate,
} from '@/types';

const API_URL = Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for auth token
    this.client.interceptors.request.use(async (config) => {
      const token = await SecureStore.getItemAsync('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized - clear token and redirect to login
          await SecureStore.deleteItemAsync('auth_token');
        }
        return Promise.reject(error);
      }
    );
  }

  // Posts API
  posts = {
    getAll: async (status?: string): Promise<Post[]> => {
      const response = await this.client.get('/posts', { params: { status } });
      return response.data;
    },
    getById: async (postId: string): Promise<Post> => {
      const response = await this.client.get(`/posts/${postId}`);
      return response.data;
    },
    create: async (post: PostCreate): Promise<Post> => {
      const response = await this.client.post('/posts', post);
      return response.data;
    },
    publish: async (postId: string): Promise<{ status: string; results: any }> => {
      const response = await this.client.post(`/posts/${postId}/publish`);
      return response.data;
    },
    cancel: async (postId: string): Promise<{ status: string }> => {
      const response = await this.client.delete(`/posts/${postId}`);
      return response.data;
    },
  };

  // Memes API
  memes = {
    getAll: async (params?: { category?: string; tags?: string; query?: string }): Promise<Meme[]> => {
      const response = await this.client.get('/memes', { params });
      return response.data;
    },
    getById: async (memeId: string): Promise<Meme> => {
      const response = await this.client.get(`/memes/${memeId}`);
      return response.data;
    },
    create: async (formData: FormData): Promise<Meme> => {
      const response = await this.client.post('/memes', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },
    delete: async (memeId: string): Promise<{ status: string }> => {
      const response = await this.client.delete(`/memes/${memeId}`);
      return response.data;
    },
    getRandom: async (category?: string): Promise<Meme> => {
      const response = await this.client.get('/memes/random', { params: { category } });
      return response.data;
    },
    getCategories: async (): Promise<string[]> => {
      const response = await this.client.get('/memes/categories');
      return response.data;
    },
  };

  // Calendar API
  calendar = {
    getEvents: async (startDate?: string, endDate?: string): Promise<CalendarEvent[]> => {
      const response = await this.client.get('/calendar', {
        params: { start_date: startDate, end_date: endDate },
      });
      return response.data;
    },
    getDaily: async (date: string): Promise<CalendarEvent[]> => {
      const response = await this.client.get('/calendar/daily', { params: { date } });
      return response.data;
    },
    getWeekly: async (startDate?: string): Promise<Record<string, CalendarEvent[]>> => {
      const response = await this.client.get('/calendar/weekly', { params: { start_date: startDate } });
      return response.data;
    },
  };

  // Platforms API
  platforms = {
    getAll: async (): Promise<string[]> => {
      const response = await this.client.get('/platforms');
      return response.data;
    },
    connect: async (connection: PlatformConnect): Promise<Platform> => {
      const response = await this.client.post('/platforms/connect', connection);
      return response.data;
    },
    disconnect: async (platform: string): Promise<{ status: string; platform: string }> => {
      const response = await this.client.delete(`/platforms/${platform}`);
      return response.data;
    },
  };

  // Analytics API
  analytics = {
    getPlatformAnalytics: async (platform: string, days: number = 7): Promise<Analytics> => {
      const response = await this.client.get(`/analytics/platform/${platform}`, { params: { days } });
      return response.data;
    },
    getPostAnalytics: async (postId: string, platform?: string): Promise<any> => {
      const response = await this.client.get(`/analytics/post/${postId}`, { params: { platform } });
      return response.data;
    },
    getBestPerforming: async (platform?: string, limit: number = 10): Promise<Post[]> => {
      const response = await this.client.get('/analytics/best-performing', { params: { platform, limit } });
      return response.data;
    },
    getTrends: async (platform: string, days: number = 30): Promise<any> => {
      const response = await this.client.get(`/analytics/trends/${platform}`, { params: { days } });
      return response.data;
    },
  };

  // Dashboard API
  dashboard = {
    getOverview: async (days: number = 7): Promise<DashboardOverview> => {
      const response = await this.client.get('/dashboard/overview', { params: { days } });
      return response.data;
    },
    getEngagement: async (days: number = 7): Promise<EngagementSummary> => {
      const response = await this.client.get('/dashboard/engagement', { params: { days } });
      return response.data;
    },
    getUpcomingPosts: async (limit: number = 10): Promise<Post[]> => {
      const response = await this.client.get('/dashboard/upcoming-posts', { params: { limit } });
      return response.data;
    },
    getRecentActivity: async (limit: number = 20): Promise<any[]> => {
      const response = await this.client.get('/dashboard/recent-activity', { params: { limit } });
      return response.data;
    },
  };

  // Templates API
  templates = {
    getAll: async (params?: { query?: string; platform?: string; category?: string }): Promise<Template[]> => {
      const response = await this.client.get('/templates', { params });
      return response.data;
    },
    getById: async (templateId: string): Promise<Template> => {
      const response = await this.client.get(`/templates/${templateId}`);
      return response.data;
    },
    create: async (template: TemplateCreate): Promise<{ template_id: string; status: string }> => {
      const response = await this.client.post('/templates', template);
      return response.data;
    },
    delete: async (templateId: string): Promise<{ status: string }> => {
      const response = await this.client.delete(`/templates/${templateId}`);
      return response.data;
    },
    render: async (templateId: string, variables: Record<string, string>): Promise<{ rendered_content: string }> => {
      const response = await this.client.post('/templates/render', { template_id: templateId, variables });
      return response.data;
    },
    getByPlatform: async (platform: string): Promise<Template[]> => {
      const response = await this.client.get(`/templates/platform/${platform}`);
      return response.data;
    },
  };
}

export const api = new ApiClient();
export default api;



import { apiClient } from '@/utils/api-client';
import { API_ENDPOINTS } from '@/utils/config';
import type {
  RegisterRequest,
  LoginRequest,
  TokenResponse,
  User,
} from '@/types/api';

export const authService = {
  async register(data: RegisterRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>(
      API_ENDPOINTS.AUTH.REGISTER,
      data
    );
    if (response.access_token) {
      await apiClient.setToken(response.access_token);
    }
    return response;
  },

  async login(data: LoginRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>(
      API_ENDPOINTS.AUTH.LOGIN,
      data
    );
    if (response.access_token) {
      await apiClient.setToken(response.access_token);
    }
    return response;
  },

  async refreshToken(): Promise<{ access_token: string; token_type: string }> {
    const response = await apiClient.post<{
      access_token: string;
      token_type: string;
    }>(API_ENDPOINTS.AUTH.REFRESH);
    if (response.access_token) {
      await apiClient.setToken(response.access_token);
    }
    return response;
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>(API_ENDPOINTS.AUTH.ME);
  },

  async getApiKey(): Promise<{ api_key: string; user_id: string }> {
    const response = await apiClient.get<{ api_key: string; user_id: string }>(
      API_ENDPOINTS.AUTH.API_KEY
    );
    if (response.api_key) {
      await apiClient.setApiKey(response.api_key);
    }
    return response;
  },

  async regenerateApiKey(): Promise<{ api_key: string; message: string }> {
    const response = await apiClient.post<{ api_key: string; message: string }>(
      API_ENDPOINTS.AUTH.REGENERATE_API_KEY
    );
    if (response.api_key) {
      await apiClient.setApiKey(response.api_key);
    }
    return response;
  },

  async logout(): Promise<void> {
    await apiClient.clearCredentials();
  },
};



import { BaseApiClient } from './base-client';
import type {
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  ProfileResponse,
  UpdateProfileRequest,
} from '@/types';

export class AuthApi extends BaseApiClient {
  async register(data: RegisterRequest): Promise<RegisterResponse> {
    const response = await this.client.post<RegisterResponse>(
      this.getUrl('/auth/register'),
      data
    );
    if (response.data.access_token) {
      await this.setToken(response.data.access_token);
    }
    return response.data;
  }

  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>(
      this.getUrl('/auth/login'),
      data
    );
    if (response.data.access_token) {
      await this.setToken(response.data.access_token);
    }
    return response.data;
  }

  async logout(): Promise<void> {
    await this.clearToken();
  }

  async getProfile(userId: string): Promise<ProfileResponse> {
    const response = await this.client.get<ProfileResponse>(
      this.getUrl(`/profile/${userId}`)
    );
    return response.data;
  }

  async updateProfile(data: UpdateProfileRequest): Promise<ProfileResponse> {
    const response = await this.client.post<ProfileResponse>(
      this.getUrl('/update-profile'),
      data
    );
    return response.data;
  }
}

export const authApi = new AuthApi();


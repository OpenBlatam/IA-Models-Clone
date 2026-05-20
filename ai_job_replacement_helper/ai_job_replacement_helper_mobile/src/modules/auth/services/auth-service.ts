import { apiService } from '@/services/api';
import type { LoginCredentials, RegisterData } from '../types';
import type { User, Session } from '@/types';

export class AuthService {
  async login(credentials: LoginCredentials): Promise<Session> {
    const response = await apiService.login(credentials.email, credentials.password);
    if (!response.data) {
      throw new Error(response.error || 'Login failed');
    }
    return response.data;
  }

  async register(data: RegisterData): Promise<User> {
    const response = await apiService.register(data.email, data.username, data.password);
    if (!response.data) {
      throw new Error(response.error || 'Registration failed');
    }
    return response.data;
  }

  async logout(): Promise<void> {
    await apiService.logout();
  }

  async verifySession(): Promise<User | null> {
    const response = await apiService.verifySession();
    return response.data || null;
  }
}

export const authService = new AuthService();



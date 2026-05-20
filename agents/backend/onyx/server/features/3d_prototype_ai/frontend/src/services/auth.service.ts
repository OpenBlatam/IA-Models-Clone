import { apiClient } from '@/lib/api-client'
import type { AuthResponse, User } from '@/types'

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  email: string
  password: string
  username: string
}

export const authService = {
  // Registro
  register: async (data: RegisterRequest): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/api/v1/auth/register', data)
    if (response.access_token) {
      apiClient.setToken(response.access_token)
    }
    return response
  },

  // Login
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/api/v1/auth/login', data)
    if (response.access_token) {
      apiClient.setToken(response.access_token)
    }
    return response
  },

  // Obtener usuario actual
  getCurrentUser: async (): Promise<User> => {
    return apiClient.get<User>('/api/v1/auth/me')
  },

  // Logout
  logout: (): void => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token')
      window.location.href = '/auth/login'
    }
  },
}




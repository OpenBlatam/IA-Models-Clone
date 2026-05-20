import axios, { AxiosInstance, AxiosError } from 'axios'
import type { ApiResponse } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    })

    // Request interceptor para agregar token
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getToken()
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor para manejar errores
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expirado o inválido
          this.clearToken()
          if (typeof window !== 'undefined') {
            window.location.href = '/auth/login'
          }
        }
        return Promise.reject(error)
      }
    )
  }

  private getToken(): string | null {
    if (typeof window === 'undefined') return null
    return localStorage.getItem('auth_token')
  }

  private clearToken(): void {
    if (typeof window === 'undefined') return
    localStorage.removeItem('auth_token')
  }

  setToken(token: string): void {
    if (typeof window === 'undefined') return
    localStorage.setItem('auth_token', token)
  }

  async get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, { params })
    return response.data.data as T
  }

  async post<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data)
    return response.data.data as T
  }

  async put<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data)
    return response.data.data as T
  }

  async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url)
    return response.data.data as T
  }

  async patch<T>(url: string, data?: unknown): Promise<T> {
    const response = await this.client.patch<ApiResponse<T>>(url, data)
    return response.data.data as T
  }
}

export const apiClient = new ApiClient()




import axios from 'axios'
import { API_CONFIG } from './constants'
import type {
  ProjectRequest,
  Project,
  GeneratorStatus,
  QueueResponse,
  Stats,
} from '@/types'

const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const api = {
  generate: async (request: ProjectRequest): Promise<{ project_id: string; status: string; message: string }> => {
    const response = await apiClient.post('/api/v1/generate', request)
    return response.data
  },

  getProject: async (projectId: string): Promise<Project> => {
    const response = await apiClient.get(`/api/v1/project/${projectId}`)
    return response.data
  },

  getStatus: async (): Promise<GeneratorStatus> => {
    const response = await apiClient.get('/api/v1/status')
    return response.data
  },

  getQueue: async (): Promise<QueueResponse> => {
    const response = await apiClient.get('/api/v1/queue')
    return response.data
  },

  getStats: async (): Promise<Stats> => {
    const response = await apiClient.get('/api/v1/stats')
    return response.data
  },

  getProjects: async (params?: {
    status?: string
    author?: string
    limit?: number
    offset?: number
  }): Promise<{ projects: Project[]; total: number }> => {
    const response = await apiClient.get('/api/v1/projects', { params })
    return response.data
  },

  startGenerator: async (): Promise<{ message: string }> => {
    const response = await apiClient.post('/api/v1/start')
    return response.data
  },

  stopGenerator: async (): Promise<{ message: string }> => {
    const response = await apiClient.post('/api/v1/stop')
    return response.data
  },

  deleteProject: async (projectId: string): Promise<{ message: string }> => {
    const response = await apiClient.delete(`/api/v1/project/${projectId}`)
    return response.data
  },

  exportZip: async (projectPath: string): Promise<{ zip_path: string }> => {
    const response = await apiClient.post('/api/v1/export/zip', { project_path: projectPath })
    return response.data
  },

  exportTar: async (projectPath: string): Promise<{ tar_path: string }> => {
    const response = await apiClient.post('/api/v1/export/tar', { project_path: projectPath })
    return response.data
  },

  validate: async (projectPath: string): Promise<{ valid: boolean; errors?: string[] }> => {
    const response = await apiClient.post('/api/v1/validate', { project_path: projectPath })
    return response.data
  },

  health: async (): Promise<{ status: string }> => {
    const response = await apiClient.get('/health')
    return response.data
  },
}


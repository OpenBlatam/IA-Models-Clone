import { apiClient } from '@/lib/api-client'
import type {
  PrototypeRequest,
  PrototypeResponse,
  ProductType,
  PrototypeHistory,
  PaginatedResponse,
} from '@/types'

export const prototypeService = {
  // Generar nuevo prototipo
  generate: async (request: PrototypeRequest): Promise<PrototypeResponse> => {
    return apiClient.post<PrototypeResponse>('/api/v1/generate', request)
  },

  // Obtener tipos de productos
  getProductTypes: async (): Promise<ProductType[]> => {
    return apiClient.get<ProductType[]>('/api/v1/product-types')
  },

  // Obtener sugerencias de materiales
  getMaterialSuggestions: async (productType: string): Promise<unknown[]> => {
    return apiClient.get<unknown[]>('/api/v1/materials/suggestions', {
      product_type: productType,
    })
  },

  // Buscar materiales
  searchMaterials: async (query: string): Promise<unknown[]> => {
    return apiClient.get<unknown[]>('/api/v1/materials/search', { q: query })
  },

  // Obtener historial
  getHistory: async (
    page = 1,
    pageSize = 20
  ): Promise<PaginatedResponse<PrototypeHistory>> => {
    return apiClient.get<PaginatedResponse<PrototypeHistory>>('/api/v1/history', {
      page,
      page_size: pageSize,
    })
  },

  // Obtener prototipo por ID
  getById: async (id: string): Promise<PrototypeResponse> => {
    return apiClient.get<PrototypeResponse>(`/api/v1/history/${id}`)
  },

  // Buscar en historial
  searchHistory: async (query: string): Promise<PrototypeHistory[]> => {
    return apiClient.get<PrototypeHistory[]>('/api/v1/history/search', { q: query })
  },

  // Análisis de viabilidad
  analyzeFeasibility: async (request: PrototypeRequest): Promise<unknown> => {
    return apiClient.post('/api/v1/feasibility', request)
  },

  // Comparar prototipos
  compare: async (
    prototype1Id: string,
    prototype2Id: string
  ): Promise<unknown> => {
    return apiClient.post('/api/v1/compare', {
      prototype1_id: prototype1Id,
      prototype2_id: prototype2Id,
    })
  },

  // Análisis de costos
  analyzeCost: async (request: PrototypeRequest): Promise<unknown> => {
    return apiClient.post('/api/v1/cost-analysis', request)
  },

  // Validar materiales
  validateMaterials: async (materials: unknown[]): Promise<unknown> => {
    return apiClient.post('/api/v1/validate-materials', { materials })
  },

  // Obtener templates
  getTemplates: async (): Promise<unknown[]> => {
    return apiClient.get<unknown[]>('/api/v1/templates')
  },

  // Obtener template por ID
  getTemplate: async (id: string): Promise<unknown> => {
    return apiClient.get(`/api/v1/templates/${id}`)
  },

  // Generar diagramas
  generateDiagrams: async (prototypeId: string): Promise<unknown> => {
    return apiClient.post('/api/v1/diagrams', { prototype_id: prototypeId })
  },

  // Obtener recomendaciones
  getRecommendations: async (request: PrototypeRequest): Promise<unknown> => {
    return apiClient.post('/api/v1/recommendations', request)
  },
}




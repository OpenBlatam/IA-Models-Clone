import { apiClient } from '@/lib/api-client'
import type { AnalyticsData } from '@/types'

export const analyticsService = {
  // Obtener analytics generales
  getAnalytics: async (): Promise<AnalyticsData> => {
    return apiClient.get<AnalyticsData>('/api/v1/analytics')
  },

  // Obtener tendencias
  getTrends: async (days = 30): Promise<unknown> => {
    return apiClient.get('/api/v1/analytics/trends', { days })
  },

  // Obtener performance
  getPerformance: async (): Promise<unknown> => {
    return apiClient.get('/api/v1/analytics/performance')
  },

  // Obtener estadísticas del historial
  getHistoryStatistics: async (): Promise<unknown> => {
    return apiClient.get('/api/v1/history/statistics')
  },

  // Obtener métricas de negocio
  getBusinessMetrics: async (): Promise<unknown> => {
    return apiClient.get('/api/v1/business-metrics/dashboard')
  },

  // Obtener métricas de ML
  getMLAnalytics: async (userId?: string): Promise<unknown> => {
    if (userId) {
      return apiClient.get(`/api/v1/ml-analytics/predict/${userId}`)
    }
    return apiClient.get('/api/v1/ml-analytics/insights')
  },
}




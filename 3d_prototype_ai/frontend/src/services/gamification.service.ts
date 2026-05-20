import { apiClient } from '@/lib/api-client'
import type { GamificationStats } from '@/types'

export const gamificationService = {
  // Obtener estadísticas del usuario
  getStats: async (userId: string): Promise<GamificationStats> => {
    return apiClient.get<GamificationStats>(`/api/v1/gamification/stats/${userId}`)
  },

  // Obtener leaderboard
  getLeaderboard: async (limit = 10): Promise<GamificationStats[]> => {
    return apiClient.get<GamificationStats[]>('/api/v1/gamification/leaderboard', {
      limit,
    })
  },

  // Otorgar puntos
  awardPoints: async (userId: string, points: number, reason: string): Promise<unknown> => {
    return apiClient.post('/api/v1/gamification/award-points', {
      user_id: userId,
      points,
      reason,
    })
  },
}




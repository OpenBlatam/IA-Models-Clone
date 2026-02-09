import { apiService } from '@/services/api';
import type { PointsAction } from '../types';
import type { GamificationProgress, LeaderboardEntry } from '@/types';

export class GamificationService {
  async getProgress(userId: string): Promise<GamificationProgress> {
    const response = await apiService.getGamificationProgress(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get progress');
    }
    return response.data;
  }

  async addPoints(userId: string, action: PointsAction): Promise<GamificationProgress> {
    const response = await apiService.addPoints(userId, action.action, action.amount);
    if (!response.data) {
      throw new Error(response.error || 'Failed to add points');
    }
    return response.data;
  }

  async getLeaderboard(limit: number = 10): Promise<LeaderboardEntry[]> {
    const response = await apiService.getLeaderboard(limit);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get leaderboard');
    }
    return response.data;
  }

  async getBadges(userId: string) {
    const response = await apiService.getBadges(userId);
    if (!response.data) {
      throw new Error(response.error || 'Failed to get badges');
    }
    return response.data;
  }
}

export const gamificationService = new GamificationService();



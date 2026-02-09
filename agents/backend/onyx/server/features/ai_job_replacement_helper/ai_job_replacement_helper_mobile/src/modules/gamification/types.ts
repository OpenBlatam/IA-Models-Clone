import type { GamificationProgress, Badge, LeaderboardEntry } from '@/types';

export interface PointsAction {
  action: string;
  amount?: number;
}

export interface BadgeProgress {
  badgeId: string;
  progress: number;
  maxProgress: number;
}

export interface LevelProgress {
  currentLevel: number;
  currentXP: number;
  xpToNextLevel: number;
  progressPercentage: number;
}



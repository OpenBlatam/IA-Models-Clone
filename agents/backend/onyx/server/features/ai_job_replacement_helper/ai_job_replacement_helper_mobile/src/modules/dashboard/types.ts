import type { DashboardData, GamificationProgress, Roadmap, Notification, Challenge } from '@/types';

export interface DashboardMetrics {
  applicationsSent: number;
  interviewsCompleted: number;
  skillsLearned: number;
  daysActive: number;
}

export interface QuickAction {
  id: string;
  icon: string;
  label: string;
  route: string;
}

export interface DashboardState {
  data: DashboardData | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}



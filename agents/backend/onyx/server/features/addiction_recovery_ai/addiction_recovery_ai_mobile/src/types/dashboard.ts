import type { RiskLevel } from './constants';
import type { LogEntryResponse } from './progress';
import type { Achievement } from './gamification';
import type { Reminder } from './notifications';

// Dashboard Types
export interface DashboardResponse {
  user_id: string;
  days_sober: number;
  current_streak: number;
  progress_percentage: number;
  recent_entries: LogEntryResponse[];
  upcoming_reminders: Reminder[];
  achievements: Achievement[];
  risk_level: RiskLevel;
}


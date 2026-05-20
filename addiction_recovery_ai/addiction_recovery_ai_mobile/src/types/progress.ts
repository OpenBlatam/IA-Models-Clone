import type { Mood } from './constants';

// Progress Types
export interface LogEntryRequest {
  user_id: string;
  date: string;
  mood: Mood;
  cravings_level: number; // 0-10
  triggers_encountered?: string[];
  consumed?: boolean;
  notes?: string;
}

export interface LogEntryResponse {
  entry_id: string;
  user_id: string;
  date: string;
  mood: string;
  cravings_level: number;
  triggers_encountered: string[];
  consumed: boolean;
  notes?: string;
  logged_at: string;
}

export interface ProgressResponse {
  user_id: string;
  days_sober: number;
  total_entries: number;
  streak_days: number;
  longest_streak: number;
  progress_percentage: number;
  recent_entries: LogEntryResponse[];
}

export interface StatsResponse {
  user_id: string;
  total_days: number;
  days_sober: number;
  relapse_count: number;
  average_cravings: number;
  most_common_triggers: string[];
  trends: Record<string, any>;
}

export interface TimelineResponse {
  user_id: string;
  timeline: TimelineEvent[];
  milestones: Milestone[];
  relapses: RelapseEvent[];
}

export interface TimelineEvent {
  date: string;
  type: string;
  description: string;
  data?: Record<string, any>;
}

export interface Milestone {
  id: string;
  date: string;
  title: string;
  description: string;
  days_sober: number;
}

export interface RelapseEvent {
  id: string;
  date: string;
  notes?: string;
  triggers?: string[];
}


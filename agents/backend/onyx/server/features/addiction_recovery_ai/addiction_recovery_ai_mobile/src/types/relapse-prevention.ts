import type { Mood, RiskLevel } from './constants';

// Relapse Prevention Types
export interface CheckRelapseRiskRequest {
  user_id: string;
  current_mood?: Mood;
  stress_level?: number; // 0-10
  cravings_level?: number; // 0-10
  recent_triggers?: string[];
}

export interface RelapseRiskResponse {
  risk_level: RiskLevel;
  risk_score: number; // 0-100
  factors: string[];
  recommendations: string[];
  emergency_plan?: string;
}

export interface TriggerResponse {
  user_id: string;
  triggers: Trigger[];
}

export interface Trigger {
  id: string;
  name: string;
  frequency: number;
  severity: number;
  last_encountered?: string;
}


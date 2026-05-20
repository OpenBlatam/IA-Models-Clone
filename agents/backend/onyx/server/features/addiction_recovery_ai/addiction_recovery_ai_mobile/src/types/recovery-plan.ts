import type { AddictionType } from './constants';

// Recovery Plan Types
export interface CreateRecoveryPlanRequest {
  user_id: string;
  addiction_type: AddictionType;
  goals?: string[];
  strategies?: string[];
  start_date?: string;
}

export interface RecoveryPlanResponse {
  plan_id: string;
  user_id: string;
  addiction_type: string;
  goals: string[];
  strategies: string[];
  start_date: string;
  created_at: string;
}


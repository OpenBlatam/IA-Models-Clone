import type { AddictionType, SeverityLevel, Frequency, RiskLevel } from './constants';

// Assessment Types
export interface AssessmentRequest {
  addiction_type: AddictionType;
  severity: SeverityLevel;
  frequency: Frequency;
  duration_years?: number;
  daily_cost?: number;
  triggers?: string[];
  motivations?: string[];
  previous_attempts?: number;
  support_system?: boolean;
  medical_conditions?: string[];
  additional_info?: string;
}

export interface AssessmentResponse {
  user_id?: string;
  assessment_id: string;
  addiction_type: string;
  severity_score: number;
  risk_level: RiskLevel;
  recommendations: string[];
  next_steps: string[];
  assessed_at: string;
}


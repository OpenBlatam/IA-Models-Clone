// Support Types
export interface CoachingSessionRequest {
  user_id: string;
  message: string;
  context?: Record<string, any>;
}

export interface CoachingSessionResponse {
  session_id: string;
  message: string;
  suggestions: string[];
  encouragement: string;
  created_at: string;
}

export interface MotivationResponse {
  user_id: string;
  message: string;
  quotes: string[];
  achievements: string[];
}


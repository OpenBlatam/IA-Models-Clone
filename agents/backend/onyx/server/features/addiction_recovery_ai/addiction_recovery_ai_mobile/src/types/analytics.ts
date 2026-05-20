// Analytics Types
export interface AnalyticsResponse {
  user_id: string;
  period: string;
  metrics: Record<string, any>;
  trends: Record<string, any>;
  insights: string[];
}


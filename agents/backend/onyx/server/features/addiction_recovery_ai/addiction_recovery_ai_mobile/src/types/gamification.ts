// Gamification Types
export interface PointsResponse {
  user_id: string;
  total_points: number;
  level: number;
  achievements: Achievement[];
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon?: string;
  unlocked_at?: string;
}


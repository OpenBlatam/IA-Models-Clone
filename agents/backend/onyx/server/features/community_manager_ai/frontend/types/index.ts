export interface Post {
  post_id: string;
  content: string;
  platforms: string[];
  scheduled_time?: string;
  status: 'scheduled' | 'published' | 'cancelled';
  media_paths?: string[];
  tags?: string[];
  created_at?: string;
  published_at?: string;
}

export interface PostCreate {
  content: string;
  platforms: string[];
  scheduled_time?: string;
  media_paths?: string[];
  tags?: string[];
}

export interface Meme {
  meme_id: string;
  image_path: string;
  caption: string;
  tags: string[];
  category: string;
  created_at?: string;
}

export interface MemeCreate {
  caption?: string;
  tags?: string[];
  category?: string;
}

export interface CalendarEvent {
  id: string;
  scheduled_time: string;
  content: string;
  platforms: string[];
  status: string;
}

export interface Platform {
  platform: string;
  connected: boolean;
  connected_at?: string;
}

export interface PlatformConnect {
  platform: string;
  credentials: Record<string, string>;
}

export interface Analytics {
  platform: string;
  total_posts: number;
  total_engagement: number;
  average_engagement_rate: number;
}

export interface DashboardOverview {
  total_posts: number;
  scheduled_posts: number;
  published_posts: number;
  connected_platforms: number;
  total_engagement: number;
  average_engagement_rate: number;
}

export interface EngagementSummary {
  total_engagement: number;
  average_engagement_rate: number;
  engagement_by_platform: Record<string, number>;
  trends: Array<{
    date: string;
    engagement: number;
  }>;
}

export interface Template {
  template_id: string;
  name: string;
  content: string;
  variables?: string[];
  category?: string;
  created_at?: string;
}

export interface TemplateCreate {
  name: string;
  content: string;
  variables?: string[];
  category?: string;
}




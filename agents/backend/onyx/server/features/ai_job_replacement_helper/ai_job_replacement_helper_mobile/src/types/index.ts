// User Types
export interface User {
  id: string;
  email: string;
  username: string;
  created_at?: string;
}

export interface Session {
  session_id: string;
  user_id: string;
  expires_at: string;
  user: User;
}

// Gamification Types
export interface GamificationProgress {
  user_id: string;
  level: number;
  xp: number;
  points: number;
  xp_to_next_level: number;
  badges: Badge[];
  streak: number;
  last_activity: string;
}

export interface Badge {
  id: string;
  type: string;
  name: string;
  description: string;
  icon: string;
  unlocked_at: string;
}

export interface LeaderboardEntry {
  user_id: string;
  username: string;
  level: number;
  xp: number;
  rank: number;
}

// Steps Types
export interface Step {
  id: string;
  title: string;
  description: string;
  category: string;
  order: number;
  prerequisites: string[];
  resources: Resource[];
  started_at?: string;
  completed_at?: string;
  status: 'not_started' | 'in_progress' | 'completed';
}

export interface Roadmap {
  steps: Step[];
  total_steps: number;
  completed_steps: number;
  progress_percentage: number;
}

export interface Resource {
  type: 'article' | 'video' | 'tool' | 'template';
  title: string;
  url: string;
  description?: string;
}

// Jobs Types
export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  description: string;
  salary_range?: string;
  job_type?: string;
  posted_date?: string;
  application_url?: string;
  required_skills?: string[];
  preferred_skills?: string[];
  match_score?: number;
  match_reasons?: string[];
}

export interface JobSwipeRequest {
  job_id: string;
  action: 'like' | 'dislike' | 'save';
}

export interface JobSwipeResponse {
  success: boolean;
  action: string;
  job_id: string;
  timestamp: string;
}

// Recommendations Types
export interface SkillRecommendation {
  skill: string;
  reason: string;
  priority: 'high' | 'medium' | 'low';
  resources: Resource[];
}

export interface JobRecommendation {
  job: Job;
  match_score: number;
  match_reasons: string[];
}

// Notifications Types
export interface Notification {
  id: string;
  user_id: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  created_at: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
}

// Mentoring Types
export interface MentoringSession {
  session_id: string;
  user_id: string;
  session_type: string;
  mentor_type: string;
  created_at: string;
  messages: ChatMessage[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

// CV Analyzer Types
export interface CVAnalysis {
  overall_score: number;
  section_scores: {
    personal_info: number;
    experience: number;
    education: number;
    skills: number;
    summary: number;
  };
  strengths: string[];
  weaknesses: string[];
  ats_score: number;
  keyword_match: {
    matched: string[];
    missing: string[];
  };
  suggestions: string[];
}

// Interview Types
export interface InterviewSession {
  session_id: string;
  user_id: string;
  interview_type: string;
  job_title?: string;
  company?: string;
  questions: InterviewQuestion[];
  current_question_index: number;
  started_at: string;
}

export interface InterviewQuestion {
  id: string;
  question: string;
  type: 'technical' | 'behavioral' | 'cultural';
  answer?: string;
  feedback?: string;
}

export interface InterviewResult {
  session_id: string;
  total_questions: number;
  answered_questions: number;
  overall_score: number;
  feedback: string;
  strengths: string[];
  improvements: string[];
}

// Challenges Types
export interface Challenge {
  id: string;
  title: string;
  description: string;
  challenge_type: 'daily' | 'weekly' | 'special';
  difficulty: 'easy' | 'medium' | 'hard';
  reward_points: number;
  reward_xp: number;
  requirements: string[];
  started_at?: string;
  completed_at?: string;
  progress: number;
  status: 'available' | 'in_progress' | 'completed';
}

// Dashboard Types
export interface DashboardData {
  user_id: string;
  gamification: GamificationProgress;
  roadmap: Roadmap;
  recent_jobs: Job[];
  recommendations: {
    skills: SkillRecommendation[];
    jobs: JobRecommendation[];
    next_steps: string[];
  };
  notifications: Notification[];
  challenges: Challenge[];
  statistics: {
    applications_sent: number;
    interviews_completed: number;
    skills_learned: number;
    days_active: number;
  };
}

// Content Generator Types
export interface ContentRequest {
  job_title?: string;
  company?: string;
  user_profile?: string;
  additional_info?: string;
}

export interface GeneratedContent {
  content: string;
  suggestions?: string[];
}

// Job Alerts Types
export interface JobAlert {
  id: string;
  user_id: string;
  keywords: string[];
  location?: string;
  job_type?: string;
  salary_range?: {
    min?: number;
    max?: number;
  };
  frequency: 'daily' | 'weekly' | 'realtime';
  active: boolean;
  matches_count: number;
  created_at: string;
}

// API Response Types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}



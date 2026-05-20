export enum EventType {
  CONCERT = 'concert',
  INTERVIEW = 'interview',
  PHOTOSHOOT = 'photoshoot',
  MEETING = 'meeting',
  REHEARSAL = 'rehearsal',
  TRAVEL = 'travel',
  OTHER = 'other',
}

export enum RoutineType {
  MORNING = 'morning',
  AFTERNOON = 'afternoon',
  EVENING = 'evening',
  NIGHT = 'night',
  CUSTOM = 'custom',
}

export enum RoutineStatus {
  PENDING = 'pending',
  COMPLETED = 'completed',
  SKIPPED = 'skipped',
  OVERDUE = 'overdue',
}

export enum ProtocolCategory {
  SOCIAL_MEDIA = 'social_media',
  INTERVIEW = 'interview',
  CONCERT = 'concert',
  PHOTOSHOOT = 'photoshoot',
  TRAVEL = 'travel',
  GENERAL = 'general',
}

export enum ProtocolPriority {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
}

export enum DressCode {
  FORMAL = 'formal',
  SMART_CASUAL = 'smart_casual',
  CASUAL = 'casual',
  SPORTY = 'sporty',
  ELEGANT = 'elegant',
  STREETWEAR = 'streetwear',
}

export enum Season {
  SPRING = 'spring',
  SUMMER = 'summer',
  FALL = 'fall',
  WINTER = 'winter',
  ALL_SEASON = 'all_season',
}

export interface CalendarEvent {
  id: string;
  title: string;
  description: string;
  event_type: EventType;
  start_time: string;
  end_time: string;
  location?: string;
  attendees?: string[];
  protocol_requirements?: string[];
  wardrobe_requirements?: string;
  notes?: string;
}

export interface RoutineTask {
  id: string;
  title: string;
  description: string;
  routine_type: RoutineType;
  scheduled_time: string;
  duration_minutes: number;
  priority: number;
  days_of_week: number[];
  is_required: boolean;
  notes?: string;
  status?: RoutineStatus;
}

export interface RoutineCompletion {
  id: string;
  task_id: string;
  completed_at: string;
  status: RoutineStatus;
  notes?: string;
  actual_duration_minutes?: number;
}

export interface Protocol {
  id: string;
  title: string;
  description: string;
  category: ProtocolCategory;
  priority: ProtocolPriority;
  rules: string[];
  do_s: string[];
  dont_s: string[];
  context?: string;
  applicable_events?: string[];
  notes?: string;
}

export interface ProtocolCompliance {
  id: string;
  protocol_id: string;
  event_id?: string;
  is_compliant: boolean;
  checked_at: string;
  notes?: string;
  violations?: string[];
}

export interface WardrobeItem {
  id: string;
  name: string;
  category: string;
  color: string;
  brand?: string;
  size?: string;
  season: Season;
  dress_codes: DressCode[];
  notes?: string;
  image_url?: string;
  last_worn?: string;
  times_worn?: number;
}

export interface Outfit {
  id: string;
  name: string;
  items: string[];
  dress_code: DressCode;
  occasion: string;
  season: Season;
  notes?: string;
  image_url?: string;
  last_worn?: string;
  times_worn?: number;
}

export interface WardrobeRecommendation {
  dress_code: DressCode;
  recommended_items: string[];
  reasoning: string;
  alternatives?: string[];
}

export interface DashboardData {
  upcoming_events: {
    count: number;
    events: CalendarEvent[];
  };
  routines: {
    pending_count: number;
    completed_today: number;
    total: number;
  };
  protocols: {
    critical_count: number;
    total: number;
  };
  wardrobe: {
    total_items: number;
    total_outfits: number;
  };
}

export interface DailySummary {
  summary: string;
  events_count: number;
  pending_routines_count: number;
  recommendations: string[];
  motivation: string;
}

export interface Alert {
  id: string;
  type: 'conflict' | 'overdue' | 'overload' | 'warning';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  title: string;
  message: string;
  timestamp: string;
  related_id?: string;
}

export interface SearchRequest {
  query: string;
  filters?: Record<string, unknown>;
}

export interface Prediction {
  predicted_value: number;
  confidence: number;
  factors: string[];
}


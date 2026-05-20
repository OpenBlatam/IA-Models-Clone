// Re-export constants
export type {
  AddictionType,
  SeverityLevel,
  Frequency,
  Mood,
  RiskLevel,
} from './constants';

export {
  ADDICTION_TYPES,
  SEVERITY_LEVELS,
  FREQUENCY_OPTIONS,
  MOOD_OPTIONS,
  RISK_LEVELS,
} from './constants';

// Auth types
export type {
  CreateUserRequest,
  UserResponse,
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  ProfileResponse,
  UpdateProfileRequest,
} from './auth';

// Assessment types
export type {
  AssessmentRequest,
  AssessmentResponse,
} from './assessment';

// Progress types
export type {
  LogEntryRequest,
  LogEntryResponse,
  ProgressResponse,
  StatsResponse,
  TimelineResponse,
  TimelineEvent,
  Milestone,
  RelapseEvent,
} from './progress';

// Recovery Plan types
export type {
  CreateRecoveryPlanRequest,
  RecoveryPlanResponse,
} from './recovery-plan';

// Relapse Prevention types
export type {
  CheckRelapseRiskRequest,
  RelapseRiskResponse,
  TriggerResponse,
  Trigger,
} from './relapse-prevention';

// Support types
export type {
  CoachingSessionRequest,
  CoachingSessionResponse,
  MotivationResponse,
} from './support';

// Analytics types
export type {
  AnalyticsResponse,
} from './analytics';

// Notification types
export type {
  Notification,
  Reminder,
} from './notifications';

// Gamification types
export type {
  PointsResponse,
  Achievement,
} from './gamification';

// Dashboard types
export type {
  DashboardResponse,
} from './dashboard';

// Emergency types
export type {
  EmergencyContact,
} from './emergency';

// API types
export type {
  ApiError,
  ApiResponse,
} from './api';

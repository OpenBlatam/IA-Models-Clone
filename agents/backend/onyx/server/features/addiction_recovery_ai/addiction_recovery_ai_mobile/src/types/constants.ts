// Constants maps instead of enums (best practice)

export const ADDICTION_TYPES = {
  SMOKING: 'smoking',
  ALCOHOL: 'alcohol',
  DRUGS: 'drugs',
  GAMBLING: 'gambling',
  INTERNET: 'internet',
  OTHER: 'other',
} as const;

export const SEVERITY_LEVELS = {
  LOW: 'low',
  MODERATE: 'moderate',
  HIGH: 'high',
  SEVERE: 'severe',
} as const;

export const FREQUENCY_OPTIONS = {
  DAILY: 'daily',
  WEEKLY: 'weekly',
  MONTHLY: 'monthly',
  OCCASIONAL: 'occasional',
} as const;

export const MOOD_OPTIONS = {
  EXCELLENT: 'excellent',
  GOOD: 'good',
  NEUTRAL: 'neutral',
  POOR: 'poor',
  TERRIBLE: 'terrible',
} as const;

export const RISK_LEVELS = {
  LOW: 'low',
  MODERATE: 'moderate',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

// Type helpers from maps
export type AddictionType = typeof ADDICTION_TYPES[keyof typeof ADDICTION_TYPES];
export type SeverityLevel = typeof SEVERITY_LEVELS[keyof typeof SEVERITY_LEVELS];
export type Frequency = typeof FREQUENCY_OPTIONS[keyof typeof FREQUENCY_OPTIONS];
export type Mood = typeof MOOD_OPTIONS[keyof typeof MOOD_OPTIONS];
export type RiskLevel = typeof RISK_LEVELS[keyof typeof RISK_LEVELS];


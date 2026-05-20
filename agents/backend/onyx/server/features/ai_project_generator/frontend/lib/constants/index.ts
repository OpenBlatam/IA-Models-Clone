export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8020',
  WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8020',
  TIMEOUT: 30000,
} as const

export const REFRESH_INTERVALS = {
  DASHBOARD: 5000,
  PROJECTS: 5000,
  QUEUE: 5000,
} as const

export const TABS = {
  GENERATE: 'generate',
  QUEUE: 'queue',
  PROJECTS: 'projects',
  STATS: 'stats',
} as const

export type TabType = typeof TABS[keyof typeof TABS]

export const PROJECT_STATUS = {
  QUEUED: 'queued',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const

export type ProjectStatus = typeof PROJECT_STATUS[keyof typeof PROJECT_STATUS]

export const FILTER_TYPES = {
  ALL: 'all',
  COMPLETED: 'completed',
  FAILED: 'failed',
  PROCESSING: 'processing',
} as const

export type FilterType = typeof FILTER_TYPES[keyof typeof FILTER_TYPES]


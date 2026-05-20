export const REFRESH_INTERVAL_MS = 10000; // 10 segundos

export const API_ENDPOINTS = {
  AGENTS: "/api/continuous-agent",
  AGENT_BY_ID: (id: string) => `/api/continuous-agent/${id}`,
} as const;

export const FILTER_OPTIONS = {
  STATUS: {
    ALL: "all",
    ACTIVE: "active",
    INACTIVE: "inactive",
  },
} as const;

export const VIEW_MODES = {
  CARDS: "cards",
  TABLE: "table",
} as const;

export const SUCCESS_RATE_THRESHOLDS = {
  EXCELLENT: 90,
  GOOD: 70,
} as const;









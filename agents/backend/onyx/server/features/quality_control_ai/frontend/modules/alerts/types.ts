export interface Alert {
  level: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  source: string;
  data?: Record<string, unknown>;
}

export interface AlertStatistics {
  total: number;
  by_level: {
    info: number;
    warning: number;
    error: number;
    critical: number;
  };
  recent_critical: number;
}


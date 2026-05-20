import { SEVERITY_COLORS, ALERT_LEVEL_COLORS } from '@/config/constants';

export const getSeverityColor = (severity: string): string => {
  return SEVERITY_COLORS[severity as keyof typeof SEVERITY_COLORS] || SEVERITY_COLORS.minor;
};

export const getAlertLevelColor = (level: string): string => {
  return ALERT_LEVEL_COLORS[level as keyof typeof ALERT_LEVEL_COLORS] || ALERT_LEVEL_COLORS.info;
};


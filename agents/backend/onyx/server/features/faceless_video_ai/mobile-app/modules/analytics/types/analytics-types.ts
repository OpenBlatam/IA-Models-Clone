/**
 * Analytics module specific types
 */

import type { Analytics, QuotaInfo, Recommendations } from '@/types/api';

export interface AnalyticsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: number;
}

export interface MetricsChartProps {
  data: Array<{ label: string; value: number }>;
  type?: 'bar' | 'line' | 'pie';
}

export interface QuotaIndicatorProps {
  quota: QuotaInfo;
  showDetails?: boolean;
}


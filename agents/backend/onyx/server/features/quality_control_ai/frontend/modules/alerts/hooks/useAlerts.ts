import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { alertsApi } from '../api';
import { useQualityControlStore } from '@/lib/store';
import { ALERTS_REFRESH_INTERVAL, STATISTICS_REFRESH_INTERVAL } from '@/config/constants';

export const useAlerts = (level?: string, limit = 50) => {
  const { setAlerts, alerts: storeAlerts } = useQualityControlStore();

  const { data, isLoading, error } = useQuery({
    queryKey: ['alerts', level, limit],
    queryFn: () => alertsApi.getRecent(level, limit),
    refetchInterval: ALERTS_REFRESH_INTERVAL,
    staleTime: 2000,
    select: (data) => data,
  });

  const { data: statistics } = useQuery({
    queryKey: ['alert-statistics'],
    queryFn: () => alertsApi.getStatistics(),
    refetchInterval: STATISTICS_REFRESH_INTERVAL,
    staleTime: 3000,
    select: (data) => data,
  });

  const alerts = useMemo(() => data || storeAlerts, [data, storeAlerts]);

  return {
    alerts,
    statistics,
    isLoading,
    error,
  };
};

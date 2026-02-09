import { useCallback, useState } from 'react';
import { RefreshControlProps } from 'react-native';
import { COLORS } from '../constants/config';

interface UseRefreshControlOptions {
  onRefresh: () => Promise<void> | void;
  refreshing?: boolean;
  tintColor?: string;
  colors?: string[];
}

export interface UseRefreshControlReturn {
  refreshing: boolean;
  refresh: () => Promise<void>;
  refreshControlProps: RefreshControlProps;
}

export function useRefreshControl({
  onRefresh,
  refreshing: externalRefreshing,
  tintColor = COLORS.primary,
  colors = [COLORS.primary],
}: UseRefreshControlOptions): UseRefreshControlReturn {
  const [internalRefreshing, setInternalRefreshing] = useState(false);

  const refreshing = externalRefreshing ?? internalRefreshing;

  const handleRefresh = useCallback(async () => {
    if (externalRefreshing !== undefined) {
      await onRefresh();
      return;
    }

    setInternalRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setInternalRefreshing(false);
    }
  }, [onRefresh, externalRefreshing]);

  return {
    refreshing,
    refresh: handleRefresh,
    refreshControlProps: {
      refreshing,
      onRefresh: handleRefresh,
      tintColor,
      colors,
    },
  };
}


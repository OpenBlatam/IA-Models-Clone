import { useState, useEffect } from 'react';
import { featureFlags } from '@/utils/feature-flags';

export function useFeatureFlag(flag: string): boolean {
  const [enabled, setEnabled] = useState(featureFlags.isEnabled(flag));

  useEffect(() => {
    setEnabled(featureFlags.isEnabled(flag));
  }, [flag]);

  return enabled;
}


/**
 * Analytics helper utilities
 */

import type { QuotaInfo } from '@/types/api';

export function getQuotaUsagePercentage(quota: QuotaInfo): {
  videos: number;
  storage: number;
} {
  return {
    videos: (quota.videos_generated / quota.videos_limit) * 100,
    storage: (quota.storage_used / quota.storage_limit) * 100,
  };
}

export function isQuotaExceeded(quota: QuotaInfo): boolean {
  return (
    quota.videos_generated >= quota.videos_limit ||
    quota.storage_used >= quota.storage_limit
  );
}

export function isQuotaWarning(quota: QuotaInfo, threshold = 0.9): boolean {
  const usage = getQuotaUsagePercentage(quota);
  return usage.videos >= threshold * 100 || usage.storage >= threshold * 100;
}

export function getRemainingQuota(quota: QuotaInfo): {
  videos: number;
  storage: number;
} {
  return {
    videos: Math.max(0, quota.videos_limit - quota.videos_generated),
    storage: Math.max(0, quota.storage_limit - quota.storage_used),
  };
}


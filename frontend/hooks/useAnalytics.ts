'use client';

import { analytics } from '@/lib/analytics';

export function useAnalytics() {
  return {
    track: analytics.track.bind(analytics),
    pageView: analytics.pageView.bind(analytics),
    click: analytics.click.bind(analytics),
    error: analytics.error.bind(analytics),
  };
}


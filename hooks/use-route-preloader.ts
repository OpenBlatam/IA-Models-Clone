"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { usePerformance } from "@/components/providers/performance-provider";

const ROUTE_PRIORITIES = {
  '/dashboard': 1,
  '/dashboard/academy': 2,
  '/dashboard/videos': 2,
  '/dashboard/lessons': 2,
  '/dashboard/games': 3,
  '/dashboard/progress': 3,
  '/dashboard/settings': 4,
} as const;

export function useRoutePreloader() {
  const router = useRouter();
  const { prefetchEnabled, isSlowConnection } = usePerformance();

  useEffect(() => {
    if (!prefetchEnabled || isSlowConnection) return;

    const preloadRoutes = () => {
      const sortedRoutes = Object.entries(ROUTE_PRIORITIES)
        .sort(([, a], [, b]) => a - b)
        .map(([route]) => route);

      sortedRoutes.forEach((route, index) => {
        setTimeout(() => {
          router.prefetch(route);
        }, index * 100);
      });
    };

    const timer = setTimeout(preloadRoutes, 1000);
    return () => clearTimeout(timer);
  }, [router, prefetchEnabled, isSlowConnection]);
}

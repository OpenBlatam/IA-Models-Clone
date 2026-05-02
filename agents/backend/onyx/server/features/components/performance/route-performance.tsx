"use client";

import React, { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

interface RouteMetrics {
  route: string;
  loadTime: number;
  timestamp: number;
}

export function RoutePerformance() {
  const pathname = usePathname();
  const [metrics, setMetrics] = useState<RouteMetrics[]>([]);
  const [startTime, setStartTime] = useState<number>(0);

  useEffect(() => {
    setStartTime(performance.now());
  }, [pathname]);

  useEffect(() => {
    const handleLoad = () => {
      const loadTime = performance.now() - startTime;
      if (startTime > 0 && pathname) {
        const metric: RouteMetrics = {
          route: pathname,
          loadTime,
          timestamp: Date.now()
        };
        
        setMetrics(prev => [...prev.slice(-9), metric]);
        
        if (process.env.NODE_ENV === 'development') {
          console.log(`Route ${pathname} loaded in ${loadTime.toFixed(2)}ms`);
        }
      }
    };

    const timer = setTimeout(handleLoad, 100);
    return () => clearTimeout(timer);
  }, [pathname, startTime]);

  if (process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 bg-background/95 backdrop-blur border rounded p-2 text-xs z-50">
      <div className="font-semibold mb-1">Route Performance</div>
      {metrics.slice(-3).map((metric, i) => (
        <div key={i} className="flex justify-between gap-2">
          <span className="truncate max-w-[120px]">{metric.route}</span>
          <span>{metric.loadTime.toFixed(0)}ms</span>
        </div>
      ))}
    </div>
  );
}

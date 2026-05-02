"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface PerformanceContextType {
  isSlowConnection: boolean;
  prefetchEnabled: boolean;
  imageQuality: 'low' | 'medium' | 'high';
  preloadCriticalRoutes: () => void;
}

const PerformanceContext = createContext<PerformanceContextType>({
  isSlowConnection: false,
  prefetchEnabled: true,
  imageQuality: 'high',
  preloadCriticalRoutes: () => {},
});

const CRITICAL_ROUTES = [
  '/dashboard/academy',
  '/dashboard/videos',
  '/dashboard/lessons',
  '/dashboard/games'
];

export function PerformanceProvider({ children }: { children: React.ReactNode }) {
  const [isSlowConnection, setIsSlowConnection] = useState(false);
  const [prefetchEnabled, setPrefetchEnabled] = useState(true);
  const [imageQuality, setImageQuality] = useState<'low' | 'medium' | 'high'>('high');
  const router = useRouter();

  const preloadCriticalRoutes = () => {
    if (!prefetchEnabled) return;
    
    CRITICAL_ROUTES.forEach(route => {
      router.prefetch(route);
    });
  };

  useEffect(() => {
    if (typeof window !== 'undefined' && 'connection' in navigator) {
      const connection = (navigator as any).connection;
      const updateConnectionStatus = () => {
        const isSlow = connection.effectiveType === 'slow-2g' || 
                      connection.effectiveType === '2g' ||
                      connection.downlink < 1.5;
        
        setIsSlowConnection(isSlow);
        setPrefetchEnabled(!isSlow);
        setImageQuality(isSlow ? 'low' : connection.downlink > 10 ? 'high' : 'medium');
      };

      updateConnectionStatus();
      connection.addEventListener('change', updateConnectionStatus);
      
      return () => connection.removeEventListener('change', updateConnectionStatus);
    }

    const timer = setTimeout(preloadCriticalRoutes, 2000);
    
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {});
    }

    return () => clearTimeout(timer);
  }, [prefetchEnabled]);

  return (
    <PerformanceContext.Provider value={{ 
      isSlowConnection, 
      prefetchEnabled, 
      imageQuality,
      preloadCriticalRoutes 
    }}>
      {children}
    </PerformanceContext.Provider>
  );
}

export const usePerformance = () => useContext(PerformanceContext);

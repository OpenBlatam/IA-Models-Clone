"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { ErrorBoundary } from "@/components/error-boundary";
import { toast } from "sonner";

interface StabilityContextType {
  isOnline: boolean;
  connectionQuality: 'good' | 'poor' | 'offline';
  retryFailedRequests: () => void;
  reportError: (error: Error, context?: Record<string, any>) => void;
}

const StabilityContext = createContext<StabilityContextType>({
  isOnline: true,
  connectionQuality: 'good',
  retryFailedRequests: () => {},
  reportError: () => {}
});

export function StabilityProvider({ children }: { children: React.ReactNode }) {
  const [isOnline, setIsOnline] = useState(true);
  const [connectionQuality, setConnectionQuality] = useState<'good' | 'poor' | 'offline'>('good');
  const [failedRequests, setFailedRequests] = useState<Array<() => Promise<void>>>([]);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setConnectionQuality('good');
      toast.success('Conexión restaurada');
      
      if (failedRequests.length > 0) {
        retryFailedRequests();
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      setConnectionQuality('offline');
      toast.error('Sin conexión a internet');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [failedRequests.length]);

  const retryFailedRequests = async () => {
    const requests = [...failedRequests];
    setFailedRequests([]);

    for (const request of requests) {
      try {
        await request();
      } catch (error) {
        console.error('Failed to retry request:', error);
        setFailedRequests(prev => [...prev, request]);
      }
    }
  };

  const reportError = (error: Error, context?: Record<string, any>) => {
    console.error('Stability Provider Error:', {
      message: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });

    if (error.message.includes('fetch') || error.message.includes('network')) {
      if (!isOnline) {
        toast.error('Sin conexión. Los cambios se guardarán cuando se restaure la conexión.');
      } else if (connectionQuality === 'poor') {
        toast.warning('Conexión lenta detectada. Algunos elementos pueden tardar en cargar.');
      } else {
        toast.error('Error de conexión. Reintentando automáticamente...');
      }
    } else {
      toast.error('Ha ocurrido un error. Por favor, inténtalo de nuevo.');
    }
  };

  return (
    <StabilityContext.Provider value={{ 
      isOnline, 
      connectionQuality, 
      retryFailedRequests,
      reportError
    }}>
      <ErrorBoundary
        onError={(error, errorInfo) => {
          reportError(error, { errorInfo: errorInfo.componentStack });
        }}
        showRetry={true}
      >
        {children}
      </ErrorBoundary>
    </StabilityContext.Provider>
  );
}

export const useStability = () => useContext(StabilityContext);

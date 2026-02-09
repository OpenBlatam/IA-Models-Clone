'use client';

import { useEffect } from 'react';
import Dashboard from '@/components/Dashboard';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { useRobotStore } from '@/lib/store/robotStore';
import { toast } from '@/lib/utils/toast';

export default function Home() {
  const { fetchStatus, connectWebSocket, disconnectWebSocket } = useRobotStore();

  useEffect(() => {
    // Fetch initial status with better error handling
    fetchStatus().catch((error) => {
      console.error('Failed to fetch initial status:', error);
      toast.error('No se pudo conectar con el backend. Verifica que esté corriendo.');
    });

    // Connect WebSocket for real-time chat
    connectWebSocket().catch((error) => {
      console.warn('WebSocket connection failed:', error);
      toast.warning('WebSocket no disponible, usando REST API');
    });

    // Set up polling for status updates
    const statusInterval = setInterval(() => {
      fetchStatus().catch((error) => {
        console.error('Status polling error:', error);
      });
    }, 2000);

    // Cleanup
    return () => {
      clearInterval(statusInterval);
      disconnectWebSocket();
    };
  }, [fetchStatus, connectWebSocket, disconnectWebSocket]);

  return (
    <ErrorBoundary>
      <main className="min-h-screen bg-white">
        <Dashboard />
      </main>
    </ErrorBoundary>
  );
}


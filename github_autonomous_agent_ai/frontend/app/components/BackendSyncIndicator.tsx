/**
 * Componente para mostrar el estado de sincronización con el backend.
 * 
 * Features:
 * - Indicador visual del estado de conexión
 * - Soporte para WebSocket y polling
 * - Formato de tiempo relativo
 * - Accesibilidad completa
 */

'use client';

import { useMemo } from 'react';
import { useBackendTasks } from '../hooks/useBackendTasks';
import { useWebSocketSync } from '../hooks/useWebSocketSync';
import { cn } from '../utils/cn';

type SyncStatus = 'error' | 'connected' | 'syncing' | 'disconnected';

interface StatusConfig {
  color: string;
  text: string;
  ariaLabel: string;
}

export function BackendSyncIndicator() {
  const { isSyncing, lastSync, error } = useBackendTasks({ enabled: true });
  const { connected } = useWebSocketSync({ enabled: true });

  const statusConfig = useMemo<StatusConfig>(() => {
    if (error) {
      return {
        color: 'bg-red-500',
        text: 'Error de sincronización',
        ariaLabel: 'Estado: Error de sincronización',
      };
    }
    if (connected) {
      return {
        color: 'bg-green-500',
        text: 'Conectado (WebSocket)',
        ariaLabel: 'Estado: Conectado mediante WebSocket',
      };
    }
    if (isSyncing) {
      return {
        color: 'bg-yellow-500',
        text: 'Sincronizando...',
        ariaLabel: 'Estado: Sincronizando',
      };
    }
    if (lastSync) {
      const timeText = formatTime(lastSync);
      return {
        color: 'bg-gray-400',
        text: `Última sync: ${timeText}`,
        ariaLabel: `Estado: Última sincronización ${timeText}`,
      };
    }
    return {
      color: 'bg-gray-400',
      text: 'Desconectado',
      ariaLabel: 'Estado: Desconectado',
    };
  }, [error, connected, isSyncing, lastSync]);

  return (
    <div 
      className="flex items-center gap-2 text-sm text-gray-600"
      role="status"
      aria-live="polite"
      aria-label={statusConfig.ariaLabel}
    >
      <div 
        className={cn(
          "w-2 h-2 rounded-full",
          statusConfig.color,
          connected && "animate-pulse"
        )}
        aria-hidden="true"
      />
      <span>{statusConfig.text}</span>
    </div>
  );
}

/**
 * Formatea una fecha como tiempo relativo en español
 */
function formatTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  
  if (seconds < 60) return `hace ${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `hace ${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `hace ${hours}h`;
}




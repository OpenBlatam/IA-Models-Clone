'use client';

/**
 * API connection status indicator component.
 * Displays the current API health status with visual feedback.
 */

import { useApiHealth } from '@/lib/hooks/use-api-health';
import { Wifi, WifiOff, RefreshCw, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ApiStatusProps {
  /**
   * Show detailed status information.
   */
  showDetails?: boolean;
  /**
   * Custom className for styling.
   */
  className?: string;
  /**
   * Position of the status indicator.
   */
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

/**
 * API connection status indicator.
 * Shows real-time API health status with automatic updates.
 */
export function ApiStatus({
  showDetails = false,
  className,
  position = 'top-right',
}: ApiStatusProps) {
  const { isHealthy, isLoading, lastChecked, message, refreshHealth } =
    useApiHealth({
      enabled: true,
      refetchInterval: 30000, // Check every 30 seconds
    });

  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  };

  if (!showDetails) {
    return (
      <div
        className={cn(
          'fixed z-50 flex items-center gap-2',
          positionClasses[position],
          className
        )}
      >
        <button
          onClick={refreshHealth}
          disabled={isLoading}
          className={cn(
            'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
            'bg-white/10 backdrop-blur-lg border border-white/20',
            'hover:bg-white/20 disabled:opacity-50',
            isHealthy
              ? 'text-green-400'
              : 'text-red-400'
          )}
          title={message}
        >
          {isLoading ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : isHealthy ? (
            <Wifi className="w-4 h-4" />
          ) : (
            <WifiOff className="w-4 h-4" />
          )}
          <span className="text-sm font-medium">
            {isLoading ? 'Checking...' : isHealthy ? 'Connected' : 'Disconnected'}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'fixed z-50',
        positionClasses[position],
        className
      )}
    >
      <div
        className={cn(
          'bg-white/10 backdrop-blur-lg rounded-lg p-4 border',
          'min-w-[200px]',
          isHealthy ? 'border-green-500/50' : 'border-red-500/50'
        )}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {isLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin text-gray-400" />
            ) : isHealthy ? (
              <Wifi className="w-4 h-4 text-green-400" />
            ) : (
              <WifiOff className="w-4 h-4 text-red-400" />
            )}
            <span
              className={cn(
                'text-sm font-semibold',
                isHealthy ? 'text-green-400' : 'text-red-400'
              )}
            >
              API Status
            </span>
          </div>
          <button
            onClick={refreshHealth}
            disabled={isLoading}
            className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            title="Refresh status"
          >
            <RefreshCw
              className={cn('w-4 h-4', isLoading && 'animate-spin')}
            />
          </button>
        </div>

        <div className="space-y-1">
          <p
            className={cn(
              'text-xs',
              isHealthy ? 'text-green-300' : 'text-red-300'
            )}
          >
            {message}
          </p>
          {lastChecked && (
            <p className="text-xs text-gray-400">
              Last checked:{' '}
              {new Date(lastChecked).toLocaleTimeString()}
            </p>
          )}
          {!isHealthy && (
            <div className="flex items-start gap-2 mt-2 p-2 bg-red-500/10 rounded border border-red-500/20">
              <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-red-300">
                Unable to connect to the API. Please check your connection and
                ensure the API server is running.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


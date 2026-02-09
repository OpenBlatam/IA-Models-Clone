'use client';

import { motion } from 'framer-motion';
import { Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import StatusIndicator from './StatusIndicator';

interface ConnectionStatusProps {
  connected: boolean;
  latency?: number;
  signalStrength?: number;
  className?: string;
  showDetails?: boolean;
}

export default function ConnectionStatus({
  connected,
  latency,
  signalStrength,
  className,
  showDetails = false,
}: ConnectionStatusProps) {
  const getSignalColor = (strength?: number) => {
    if (!strength) return 'text-tesla-gray-dark';
    if (strength >= 75) return 'text-[#10b981]';
    if (strength >= 50) return 'text-[#f59e0b]';
    return 'text-[#ef4444]';
  };

  const getLatencyColor = (latency?: number) => {
    if (!latency) return 'text-tesla-gray-dark';
    if (latency < 50) return 'text-[#10b981]';
    if (latency < 100) return 'text-[#f59e0b]';
    return 'text-[#ef4444]';
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        'flex items-center gap-tesla-sm px-tesla-md py-tesla-sm rounded-lg border',
        connected
          ? 'bg-[#d1fae5] border-[#10b981]'
          : 'bg-[#fee2e2] border-[#ef4444]',
        className
      )}
    >
      <StatusIndicator
        status={connected ? 'online' : 'offline'}
        size="md"
        showPulse={connected}
      />
      <div className="flex-1">
        <div className="flex items-center gap-2">
          {connected ? (
            <Wifi className={cn('w-4 h-4', getSignalColor(signalStrength))} />
          ) : (
            <WifiOff className="w-4 h-4 text-[#ef4444]" />
          )}
          <span
            className={cn(
              'text-sm font-medium',
              connected ? 'text-[#065f46]' : 'text-[#991b1b]'
            )}
          >
            {connected ? 'Conectado' : 'Desconectado'}
          </span>
        </div>
        {showDetails && connected && (
          <div className="flex items-center gap-4 mt-1 text-xs text-tesla-gray-dark">
            {latency !== undefined && (
              <span className={getLatencyColor(latency)}>
                {latency}ms
              </span>
            )}
            {signalStrength !== undefined && (
              <span className={getSignalColor(signalStrength)}>
                {signalStrength}%
              </span>
            )}
          </div>
        )}
      </div>
      {!connected && (
        <AlertCircle className="w-4 h-4 text-[#ef4444]" />
      )}
    </motion.div>
  );
}

